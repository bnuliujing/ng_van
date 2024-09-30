import math

import torch
from tqdm import tqdm

from utils import minsr_solve, patches_to_spins


def train_sgd(args, model, optimizer, ham, beta, writer):
    pbar = tqdm(range(args.epochs))
    for n_iter in pbar:
        optimizer.zero_grad()
        x = model.sample(args.batch_size)
        log_probs = model(x)
        if args.nn == "transformer" and args.patch_size > 1:  # convert patches to spins
            s = patches_to_spins(x, args.patch_size) * 2.0 - 1.0
        else:
            s = x * 2.0 - 1.0
        with torch.no_grad():
            energy = ham.energy(s)
            loss = log_probs / beta + energy
        loss_reinforce = torch.mean(log_probs * (loss - loss.mean()))
        loss_reinforce.backward()
        optimizer.step()

        with torch.no_grad():
            free_energy_ = loss.mean().item() / args.n
            free_energy_std_ = loss.std().item() / args.n
            energy_ = energy.mean().item() / args.n
            entropy_ = -1.0 * log_probs.mean().item() / args.n
            pbar.set_description(f"beta: {beta:.2f}, f: {free_energy_:.8g}, f_std: {free_energy_std_:.8g}")

            if args.use_tb:
                writer.add_scalar(f"beta{beta:.2f}/free_energy", free_energy_, n_iter)
                writer.add_scalar(f"beta{beta:.2f}/free_energy_std", free_energy_std_, n_iter)
                writer.add_scalar(f"beta{beta:.2f}/energy", energy_, n_iter)
                writer.add_scalar(f"beta{beta:.2f}/entropy", entropy_, n_iter)


def train_ng(args, model, ham, beta, writer):
    pbar = tqdm(range(args.epochs))
    for n_iter in pbar:
        with torch.no_grad():
            x = model.sample(args.batch_size)
            log_probs = model(x)
            if args.nn == "transformer" and args.patch_size > 1:  # convert patches to spins
                s = patches_to_spins(x, args.patch_size) * 2.0 - 1.0
            else:
                s = x * 2.0 - 1.0
            energy = ham.energy(s)
            loss = log_probs / beta + energy  # Reward, (N_s,)

        grads = model.per_sample_grad(x)  # d logP(x_i) / d theta_j, dict
        grads_flatten = torch.cat([torch.flatten(v, start_dim=1) for v in grads.values()], dim=1)  # N x M
        O_mat = grads_flatten / math.sqrt(args.batch_size)
        R_vec = (loss - loss.mean()) / math.sqrt(args.batch_size)
        O_mat, R_vec = O_mat.double(), R_vec.double()
        dtheta_flatten = minsr_solve(O_mat, R_vec, lambd=args.lambd)
        if args.adaptive_lr:
            lr = math.sqrt(2 * args.lr / (torch.dot(O_mat.T @ R_vec, dtheta_flatten)))
        else:
            lr = args.lr
        model.update_params(dtheta_flatten.float(), lr)

        with torch.no_grad():
            free_energy_ = loss.mean().item() / args.n
            free_energy_std_ = loss.std().item() / args.n
            energy_ = energy.mean().item() / args.n
            entropy_ = -1.0 * log_probs.mean().item() / args.n
            pbar.set_description(f"beta: {beta:.2f}, f: {free_energy_:.8g}, f_std: {free_energy_std_:.8g}")

        if args.use_tb:
            writer.add_scalar(f"beta{beta:.2f}/free_energy", free_energy_, n_iter)
            writer.add_scalar(f"beta{beta:.2f}/free_energy_std", free_energy_std_, n_iter)
            writer.add_scalar(f"beta{beta:.2f}/energy", energy_, n_iter)
            writer.add_scalar(f"beta{beta:.2f}/entropy", entropy_, n_iter)
