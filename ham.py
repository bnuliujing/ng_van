import math

import networkx as nx
import numpy as np
import torch


class Hamiltonian1D:
    def energy(self, s):
        assert s.shape[1] == self.n
        return -0.5 * torch.sum((s @ self.J) * s, dim=1)


class SKModel(Hamiltonian1D):
    def __init__(self, n=20, seed=1, device="cpu", *args, **kwargs):
        rng = np.random.default_rng(seed)
        self.J_np = rng.normal(size=(n, n)) / math.sqrt(n)
        self.J_np = np.triu(self.J_np, k=1)
        self.J_np = self.J_np + self.J_np.T
        self.J = torch.from_numpy(self.J_np).float().to(device)
        self.n = n
        self.seed = seed

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self.n}, seed={self.seed}, mean(J)={self.J_np.mean():.4g}, std(J)={self.J_np.std():.4g})"


class RRGInstance(Hamiltonian1D):
    def __init__(self, n=20, d=3, seed=1, device="cpu", *args, **kwargs):
        rng = np.random.default_rng(seed)
        graph = nx.random_regular_graph(d=d, n=n, seed=seed)
        weights = rng.integers(2, size=len(graph.edges)) * 2 - 1
        for (u, v), w in zip(graph.edges(), weights):
            graph[u][v]["weight"] = w
        adj_matrix = nx.adjacency_matrix(graph)
        self.J_np = np.triu(adj_matrix.toarray(), k=1).astype(float)
        self.J_np = self.J_np + self.J_np.T
        self.J = torch.from_numpy(self.J_np).float().to(device)
        self.n = n
        self.d = d
        self.seed = seed

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self.n}, d={self.d}, seed={self.seed}, mean(J)={self.J_np.mean():.4g}, std(J)={self.J_np.std():.4g})"
