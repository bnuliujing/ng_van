import math

import torch


def cholesky_solve(O_mat, F_vec, lambd=1e-3):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = O^T R = F` by Cholesky decomposition

    see arXiv:2310.17556, Algorithm 1
    """
    N, _ = O_mat.size()
    W = O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device)
    L = torch.linalg.cholesky(W)
    Q = torch.linalg.inv(L) @ O_mat

    return (F_vec - Q.T @ Q @ F_vec) / lambd


def cholesky_solve_fast(O_mat, F_vec, lambd=1e-3):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = O^T R = F` by Cholesky decomposition

    The computation Q is inlined

    see arXiv:2310.17556, Algorithm 1
    """
    N, _ = O_mat.size()
    W = O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device)
    L = torch.linalg.cholesky(W)
    QTQF = O_mat.T @ torch.cholesky_solve(O_mat, L) @ F_vec

    return (F_vec - QTQF) / lambd


def svd_solve(O_mat, F_vec, lambd=1e-3):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = F` by svd

    First compute `O O^T = U @ Sigma^2 @ U^T`, then `V = O^T U Sigma^{-1}`
    
    see arXiv:2310.17556, Appendix C
    """
    Sigma2, U = torch.linalg.eigh(O_mat @ O_mat.T)
    V = O_mat.T @ (1.0 / torch.sqrt(Sigma2) * U)  # V = O^T U Sigma^{-1}

    return (V * (1.0 / (Sigma2 + lambd))) @ V.T @ F_vec + (F_vec - V @ V.T @ F_vec) / lambd


def minsr_solve(O_mat, R_vec, lambd=1e-3, r_pinv=1e-12, a_pinv=0, soft=True):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = O^T R` by minSR

    `dtheta = (O^T O + lambda I)^-1 O^T R = O^T (O O^T + lambda I)^-1 R`

    Compute `O O^T = U @ D @ U^T`, so `(O O^T)^-1 = U @ D^-1 @ U^T`

    see arXiv:2108.03409 Section 2.5, and arXiv:2302.01941 MinSR solution
    """
    N, _ = O_mat.size()
    D, U = torch.linalg.eigh(O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device))
    threshold = D.max().abs() * r_pinv + a_pinv
    if soft:
        D_inv = 1 / (D * (1 + (threshold / torch.abs(D)) ** 6))
    else:
        D_inv = torch.where(torch.abs(D) >= threshold, 1 / D, torch.tensor(0.0, device=O_mat.device, dtype=O_mat.dtype))
    T_inv = (U * D_inv) @ U.T

    return O_mat.T @ T_inv @ R_vec


def dec2bin(x, length):
    mask = 2 ** torch.arange(length - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).int()


def bin2dec(b, length):
    mask = 2 ** torch.arange(length - 1, -1, -1).to(b.device, torch.int)
    return torch.sum(mask * b.int(), -1)


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return (torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1


def patches_to_spins(patches, patch_size):
    # convert patched samples to spin samples
    # input: (bs, N // patch_size) in {0, ..., 2^patch_size - 1}
    # output: (bs, N) in {-1, +1}
    bs, n_patches = patches.size()
    x = dec2bin(patches[:, :patch_size], patch_size).view(bs, -1)
    for i in range(1, n_patches):
        x_next = dec2bin(patches[:, i * patch_size : (i + 1) * patch_size], patch_size).view(bs, -1)
        x = torch.cat([x, x_next], dim=1)
    return x


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
# this function is the same as F.scaled_dot_product_attention
# but is more efficient for per-sample gradients
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
