from typing import Optional
import torch
from HyperbolicEmbeddings.hyperbolic_manifold.manifold import Manifold
import geoopt


class EuclideanManifold(Manifold):

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.D = dim
        self.geoopt = geoopt.Euclidean(ndim=1)

    @property
    def origin(self) -> torch.Tensor:
        return torch.zeros(self.D)

    def metric_tensor(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        return torch.eye(D).repeat(N, 1, 1)  # [N, D, D]

    def distance_squared(self, X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
        if diag:
            diff = X-Y  # [N, D]
        else:
            diff = Y.unsqueeze(0) - X.unsqueeze(1)  # [N, M, D]
        return (diff**2).sum(-1)  # [N] / [N, M]

    def exp(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return X + U

    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return Y - X

    def parallel_transport(self, X: torch.Tensor, Y: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return U

    def tangent_space_projection_matrix(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        return torch.eye(D).repeat(N, 1, 1)  # [N, D, D]

    def proj(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return U

    def are_valid_points(self, X: torch.Tensor) -> bool:
        return X.shape[1] == self.D

    def are_valid_tangent_space_vectors(self, X: torch.Tensor, U: torch.Tensor) -> bool:
        return X.shape[1] == self.D and U.shape[1] == self.D
