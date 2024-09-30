from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from utils import scaled_dot_product_attention


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    @staticmethod
    def _loss_fn(log_probs):
        return log_probs.mean(0)

    def _get_params(self):
        return {k: v.detach() for k, v in self.named_parameters()}

    def _compute_loss(self, params, sample):
        batch = sample.unsqueeze(0)
        log_prob = functional_call(self, (params,), (batch,))
        loss = self._loss_fn(log_prob)
        return loss

    def per_sample_grad(self, samples):
        """
        Return per sample gradients, shape (batch_size, num_params), in the form of a dict
        """
        compute_grad = grad(self._compute_loss)
        compute_sample_grad = vmap(compute_grad, in_dims=(None, 0))

        return compute_sample_grad(self._get_params(), samples)

    @torch.no_grad()
    def update_params(self, updates_flatten, lr):
        idx = 0
        for _, v in self.named_parameters():
            updates = updates_flatten[idx : idx + v.numel()].view(v.shape)
            v.data -= lr * updates
            idx += v.numel()


class NADE(BaseModel):
    def __init__(self, n, hidden_dim, device="cpu", *args, **kwargs):
        super().__init__()

        self.register_parameter("W", nn.Parameter(torch.randn(hidden_dim, n)))
        self.register_parameter("c", nn.Parameter(torch.zeros(hidden_dim)))
        self.register_parameter("V", nn.Parameter(torch.randn(n, hidden_dim)))
        self.register_parameter("b", nn.Parameter(torch.zeros(n)))

        self.n = n
        self.hidden_dim = hidden_dim
        self.device = device

        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.V)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, hidden_dim={self.hidden_dim})"

    def _forward(self, x):
        logits_list = []
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i] + torch.einsum("h,bh->b", self.V[i, :], h_i)
            logits_list.append(logits)

        return torch.stack(logits_list, dim=1)

    def forward(self, x):
        logits = self._forward(x)
        log_prob = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")

        return log_prob.sum(-1)

    @torch.no_grad()
    def sample(self, batch_size):
        x = torch.zeros(batch_size, self.n, dtype=torch.float, device=self.device)
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i] + torch.einsum("h,bh->b", self.V[i, :], h_i)
            x[:, i] = torch.bernoulli(torch.sigmoid(logits))

        return x


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias, exclusive):
        super().__init__(in_channels * n, out_channels * n, bias)

        if exclusive:
            mask = torch.tril(torch.ones(n, n), diagonal=-1)
        else:
            mask = torch.tril(torch.ones(n, n))

        mask = torch.cat([mask] * in_channels, dim=1)
        mask = torch.cat([mask] * out_channels, dim=0)
        self.register_buffer("mask", mask)
        self.weight.data *= self.mask
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(BaseModel):
    def __init__(self, n, num_channels, device="cpu", *args, **kwargs):
        super().__init__()

        layer = []
        if num_channels == 0:  # no hidden layer, FVSBN
            layer.append(MaskedLinear(1, 1, n, bias=True, exclusive=True))
        else:  # one hidden layer seems to be enough
            layer.append(MaskedLinear(1, num_channels, n, bias=True, exclusive=True))
            layer.append(nn.Sigmoid())
            layer.append(MaskedLinear(num_channels, 1, n, bias=True, exclusive=False))
        self.net = nn.Sequential(*layer)

        self.n = n
        self.num_channels = num_channels
        self.device = device

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, num_channels={self.num_channels})"

    def _forward(self, x):
        return self.net(x)

    def forward(self, x):
        logits = self._forward(x)
        log_prob = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")

        return log_prob.sum(-1)

    @torch.no_grad()
    def sample(self, batch_size):
        x = torch.zeros(batch_size, self.n, dtype=torch.float, device=self.device)
        for i in range(self.n):
            logits = self._forward(x)[:, i]
            x[:, i] = torch.bernoulli(torch.sigmoid(logits))

        return x


@dataclass
class TransformerConfig:
    phy_dim: int = 2
    max_len: int = 20
    emb_dim: int = 32
    mlp_dim: int = 128
    num_heads: int = 2
    num_layers: int = 1
    use_bias: bool = False
    device: str = "cpu"


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.num_heads == 0
        self.W_in = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.use_bias)
        self.W_out = nn.Linear(config.emb_dim, config.emb_dim, bias=config.use_bias)
        self.emb_dim = config.emb_dim
        self.num_heads = config.num_heads
        self.d_k = config.emb_dim // config.num_heads

    def forward(self, x):
        B, T, C = x.size()  # batch size, length, embedding dimension
        assert C == self.emb_dim

        Q, K, V = self.W_in(x).split(self.emb_dim, dim=2)  # (B, T, C)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        # y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, num_heads, T, d_k)
        y = scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, num_heads, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.W_out(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.emb_dim, config.mlp_dim, bias=config.use_bias)
        self.l2 = nn.Linear(config.mlp_dim, config.emb_dim, bias=config.use_bias)
        self.activation = nn.ReLU()
        # self.activation = nn.GELU()

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim, bias=config.use_bias)
        self.mha = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.emb_dim, bias=config.use_bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class TransformerARModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.phy_dim, config.emb_dim),
                wpe=nn.Embedding(config.max_len, config.emb_dim),
                blocks=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
                # layer_norm=nn.LayerNorm(config.emb_dim, bias=config.use_bias),
                linear=nn.Linear(config.emb_dim, config.phy_dim, bias=False),
            )
        )
        self.transformer.wte.weight = self.transformer.linear.weight  # weight-tying

        # initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward(self, x, target=None):
        # x: (x_0, x_1, ..., x_{T-1}), shape (B, T)
        # target: (x_1, x_2, ..., x_T), shape (B, T)
        # device = x.device
        B, T = x.size()  # batch size, length
        assert T <= self.config.max_len
        pos = torch.arange(0, T, dtype=torch.long, device=self.config.device)

        # forward pass
        tok_emb = self.transformer.wte(x)  # (B, T, emb_dim)
        pos_emb = self.transformer.wpe(pos)  # (T, emb_dim)
        x = tok_emb + pos_emb
        for block in self.transformer.blocks:
            x = block(x)
        # x = self.transformer.layer_norm(x)

        if target is not None:
            logits = self.transformer.linear(x)  # (B, T, phy_dim)
            loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")  # (B, T)
        else:
            logits = self.transformer.linear(x[:, [-1], :])  # (B, 1, phy_dim)
            loss = None

        return logits, loss

    @torch.no_grad()
    def sample(self, batch_size):
        # use auxiliary variable x_0=0 to sample
        x = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=self.config.device)
        for _ in range(self.config.max_len):
            logits, _ = self._forward(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)

        return x[:, 1:]

    @staticmethod
    def shift_inputs(x):
        # add an auxiliary variable x_0=0 and return [x_0, x_1, ..., x_{N-1}]
        device = x.device
        B, T = x.size()
        aux_x = torch.zeros(size=(B, 1), dtype=torch.long, device=device)

        return torch.cat((aux_x, x[:, :-1]), dim=1)

    def forward(self, x):
        B, T = x.size()
        assert T == self.config.max_len
        _, loss = self._forward(self.shift_inputs(x), x)  # (B, T)

        return -loss.sum(-1)
