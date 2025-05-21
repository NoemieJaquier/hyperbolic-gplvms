from abc import ABC, abstractmethod
import torch
import geoopt


class Manifold(ABC):

    dim: int
    """dimension of the manifold."""

    D: int
    """D: number of coordinates used to represent points on the manifold."""

    geoopt: geoopt.Manifold
    """corresponding geoopt manifold. (Needed for geoopt RiemannianAdam)"""

    @property
    @abstractmethod
    def origin(self) -> torch.Tensor:
        """o: [D] origin point on the manifold"""
        ...

    @abstractmethod
    def metric_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -
        X: [N, D]  N manifold points

        Returns
        -
        G: [N, D, D] N metric tensors
        """
        ...

    def inner_product(self, X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
        """
        <x,y>_G

        Parameters
        -
        X: [N, D]  N manifold points or tangent space vectors
        Y: [M, D]  M manifold points or tangent space vectors
        diag       True requires that N must equal M to return the diagonal of the inner product matrix

        Returns
        -
        inner_product: [N, M] if not diag else [N]
        """
        G = self.metric_tensor(X)  # [N, D, D]
        if not diag:
            return X @ G[0] @ Y.T  # [N, M] = [N, D] x [D, D] x [D, M]
        assert X.shape[0] == Y.shape[0]
        return torch.bmm(torch.bmm(X.unsqueeze(1), G), Y.unsqueeze(2)).squeeze(1, 2)  # [N] = [N, 1, D] x [N, D, D] x [N, D, 1]

    def norm(self, U: torch.Tensor) -> torch.Tensor:
        """
        u_norm = sqrt(<u, u>_G)

        Parameters
        -
        U: [N, D]  N tangent space vectors

        Returns
        -
        u_norm: [N]  norm of each tangent space vector
        """
        return torch.sqrt(self.inner_product(U, U, diag=True))

    def distance(self, X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
        """
        d(x, y) = sqrt(d^2(x,y))

        Parameters
        -
        X: [N, D]  N manifold points
        Y: [M, D]  M manifold points

        Returns
        -
        distance: [N, M] if not diag else [N]
        """
        return torch.sqrt(self.distance_squared(X, Y, diag))

    @abstractmethod
    def distance_squared(self, X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
        """
        d^2(x, y) = sqrt(d^2(x,y))

        Parameters
        -
        X: [N, D]  N manifold points
        Y: [M, D]  M manifold points

        Returns
        -
        distance: [N, M] if not diag else [N]
        """
        ...

    @abstractmethod
    def exp(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -
        X: [N, D]  N manifold points
        U: [N, D]  N tangent space vectors on X

        Returns
        -
        exp_x_u: [N, D] where exp_x_u[i] = Exp_{x_i}(u_i)
        """
        ...

    @abstractmethod
    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -
        X: [N, D]  N manifold points
        Y: [N, D]  N manifold points

        Returns
        -
        log_x_y: [N, D] where log_x_y[i] = Log_{x_i}(y_i)
        """
        ...

    @abstractmethod
    def parallel_transport(self, X: torch.Tensor, Y: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -
        X: [N, D]  N manifold points
        Y: [N, D]  N manifold points
        U: [N, D]  N tangent space vectors at X

        Returns
        -
        V: [N, D] where V[i] = P_{x_i->y_i}(u_i)
        """
        ...

    @abstractmethod
    def tangent_space_projection_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        P_x = G + x@x.T

        Parameters
        -
        X: [N, D]  N manifold points

        Returns
        -
        P: [N, D, D] N projection matrices where P[i] = P_{x_i}
        """
        ...

    @abstractmethod
    def proj(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        u_proj = P_x @ u

        Parameters
        -
        X: [N, D]   N manifold points
        U: [N, D]   N points in the Euclidean space surrounding the manifold  

        Returns
        -
        U_proj: [N, D]  points U projected onto the tangent space of X 
        """
        ...

    @abstractmethod
    def are_valid_points(self, X: torch.Tensor) -> torch.Tensor:
        """
        x_0 > 0 and <x, x>_L = -1

        Parameters
        -
        X: [N, D] N potential manifold points

        Returns
        -
        valid: [bool] True if all X are valid manifold points
        """
        ...

    @abstractmethod
    def are_valid_tangent_space_vectors(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        <x, u>_L = 0

        Parameters
        -
        X: [N, D]   valid manifold points
        U: [N, D]   potential tangent space vectors at X

        Returns
        -
        valid: [bool] True if all U are valid tangent space vectors at X
        """
        ...
