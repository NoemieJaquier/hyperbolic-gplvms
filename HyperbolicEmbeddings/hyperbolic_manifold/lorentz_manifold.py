import torch
import geoopt
import HyperbolicEmbeddings.hyperbolic_manifold.lorentz as lorentz
from HyperbolicEmbeddings.hyperbolic_manifold.manifold import Manifold


class LorentzManifold(Manifold):

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.D = dim+1
        self.geoopt = geoopt.Lorentz()

    @property
    def origin(self) -> torch.Tensor:
        return lorentz.origin(self.dim)

    def metric_tensor(self, X: torch.Tensor) -> torch.Tensor:
        return lorentz.metric_tensor(X)

    def distance(self, X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
        return lorentz.distance(X, Y, diag)

    def distance_squared(self, X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
        return lorentz.distance_squared(X, Y, diag)

    def exp(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return lorentz.exp(X, U)

    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return lorentz.log(X, Y)

    def parallel_transport(self, X: torch.Tensor, Y: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return lorentz.parallel_transport(X, Y, U)

    def tangent_space_projection_matrix(self, X: torch.Tensor) -> torch.Tensor:
        return lorentz.tangent_space_projection_matrix(X)

    def proj(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return lorentz.proj(X, U)

    def are_valid_points(self, X: torch.Tensor) -> torch.Tensor:
        return lorentz.are_valid_points(X)

    def are_valid_tangent_space_vectors(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return lorentz.are_valid_tangent_space_vectors(X, U)
