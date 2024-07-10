import torch
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.kernels import Kernel
from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import lorentz_distance_torch


def stress(distances: torch.Tensor, graph_distances: torch.Tensor) -> torch.Tensor:
    """
    Computes the stress
    """
    distances_diff = graph_distances - distances
    i, j = torch.triu_indices(*distances_diff.shape, offset=1)

    return torch.pow(torch.triu(distances_diff, diagonal=1), 2)[i, j]


class ZeroAddedLossTermExactMLL(AddedLossTerm):
    """
    This class can be used as a template to define an extra loss for an exact marginal likelihood.
    In this case, the latent variable is given as a parameter of the loss function during its call.
    Additional parameters can be passed in the initialization of the loss term.
    """
    def __init__(self):  # param):  # Uncomment to add initial parameters
        super().__init__()
        # self.param = param

    def loss(self, x, **kwargs):
        return 0.0 * torch.sum(x)


class ZeroAddedLossTermApproximateMLL(AddedLossTerm):
    """
    This class can be used as a template to define an extra loss for an approximate marginal likelihood.
    In this case, the latent variable must be updated before the call of the loss function.
    Additional parameters can be passed in the initialization of the loss term.
    """
    def __init__(self, x):
        super().__init__()
        self.x = x

    def loss(self):
        return 0.0 * torch.sum(self.x)


class StressLossTermExactMLL(AddedLossTerm):
    """
    Class to compute cost regularizer (coming from a prior on latent variables) based on the stress loss (Eq. 10 in [1])

    [1] C. Cruceru. "Computationally Tractable Riemannian Manifolds for Graph Embeddings." AAAI, 2021

    Parameters
    ----------
    graph_distances: Graph distance matrix for all pairs of data points
    loss_scale: Constant to control magnitude of stress loss
    """
    def __init__(self, graph_distance_matrix, distance_function, loss_scale=1.0):
        super().__init__()
        self.graph_distances = graph_distance_matrix  # As this does not change for the optimization, it is a parameter
        self.distance = distance_function
        self.loss_scale = loss_scale

    def loss(self, x, **kwargs):
        """
        Implements stress loss according to Eq. 10 in [1].

        Parameters
        ----------
        x: Optimization variable (i.e. latent variables in GPLVM)
        kwargs: Additional arguments for loss function

        Returns
        -------
        stress_loss: Sum over the squared difference between the graph and manifold distances as defined in [1]
                     The negative stress is return to comply with GPytorch that maximizes added terms.
        """
        distances = self.distance(x, x)
        stress_loss = self.loss_scale * torch.mean(stress(distances, self.graph_distances))

        return - stress_loss


class HyperbolicStressLossTermExactMLL(StressLossTermExactMLL):
    """
    Class to compute cost regularizer (coming from a prior on latent variables) based on the stress loss (Eq. 10 in [1])

    [1] C. Cruceru. "Computationally Tractable Riemannian Manifolds for Graph Embeddings." AAAI, 2021

    Parameters
    ----------
    graph_distances: Graph distance matrix for all pairs of data points
    loss_scale: Constant to control magnitude of stress loss
    """
    def __init__(self, graph_distance_matrix, loss_scale=1.0):
        super().__init__(graph_distance_matrix, lorentz_distance_torch, loss_scale)


class EuclideanStressLossTermExactMLL(StressLossTermExactMLL):
    """
    Class to compute cost regularizer (coming from a prior on latent variables) based on the stress loss (Eq. 10 in [1])

    [1] C. Cruceru. "Computationally Tractable Riemannian Manifolds for Graph Embeddings." AAAI, 2021

    Parameters
    ----------
    graph_distances: Graph distance matrix for all pairs of data points
    loss_scale: Constant to control magnitude of stress loss
    """
    def __init__(self, graph_distance_matrix, loss_scale=1.0):

        super().__init__(graph_distance_matrix, Kernel().covar_dist, loss_scale)

