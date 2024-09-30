import os

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from args import args
from ham import RRGInstance, SKModel
from model import MADE, NADE, TransformerARModel, TransformerConfig
from train import train_ng, train_sgd


def main():
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() and args.gpu > -1
    args.device = f"cuda:{args.gpu}" if use_cuda else "cpu"

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args.use_tb:
        writer = SummaryWriter(log_dir=args.path)
    else:
        writer = None

    with open(os.path.join(args.path, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    if args.nn == "nade":
        model = NADE(**vars(args)).to(args.device)
    elif args.nn == "made":
        model = MADE(**vars(args)).to(args.device)
    elif args.nn == "transformer":
        assert args.n % args.patch_size == 0, "n must be divisible by patch_size"
        assert args.phy_dim == 2**args.patch_size, "phy_dim must be 2^patch_size"
        config = TransformerConfig(
            phy_dim=args.phy_dim,
            max_len=args.n // args.patch_size,
            emb_dim=args.emb_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            use_bias=args.use_bias,
            device=args.device,
        )
        model = TransformerARModel(config).to(args.device)
    else:
        raise ValueError(f"Unknown neural network {args.model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if not args.nat_grad else None

    if args.ham == "sk":
        ham = SKModel(**vars(args))
    elif args.ham == "rrg":
        ham = RRGInstance(**vars(args))
    else:
        raise ValueError(f"Hamiltonian {args.ham} not supported")

    num_betas = (args.beta_final - args.beta_init) / args.beta_interval + 1
    beta_list = np.linspace(args.beta_init, args.beta_final, round(num_betas))

    for beta in beta_list:
        if args.nat_grad:
            train_ng(args, model, ham, beta, writer)
        else:
            train_sgd(args, model, optimizer, ham, beta, writer)

    if args.use_tb:
        writer.close()


if __name__ == "__main__":
    main()
