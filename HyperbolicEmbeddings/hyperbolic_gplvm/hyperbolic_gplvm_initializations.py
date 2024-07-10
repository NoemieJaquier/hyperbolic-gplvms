import torch
from geoopt.manifolds import Lorentz
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam

from HyperbolicEmbeddings.hyperbolic_distributions.hyperbolic_wrapped_normal import LorentzWrappedNormal
from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import exp_map_mu0
from HyperbolicEmbeddings.losses.graph_based_loss import HyperbolicStressLossTermExactMLL
from HyperbolicEmbeddings.gplvm.gplvm_initializations import euclidean_stress_loss_initialization


def hyperbolic_tangent_pca_initialization(data, latent_dim):
    """
    Compute the initialization of the latent variables using PCA. The initial latent variables obtained with Euclidean
    PCA are transformed to elements on the tangent space at the origin and then projected on the hyperbolic manifold
    with the exponential map.

    Parameters
    ----------
    :param data: observations  [nb_data x dimension]
    :param latent_dim: dimension of the latent space

    Return
    ------
    :return initial latent variables [nb_data x latent dimension]
    """
    U, S, V = torch.pca_lowrank(data, q=latent_dim)
    reduced_data = 0.1 * torch.matmul(data, V[:, :latent_dim])
    augmented_reduced_data = torch.hstack((torch.zeros_like(reduced_data[:, 0])[:, None], reduced_data))
    return torch.nn.Parameter(exp_map_mu0(augmented_reduced_data))


def hyperbolic_stress_loss_initialization(data: torch.Tensor, latent_dim: int, graph_distances: torch.Tensor,
                                          n_steps=1000, verbose: bool = True, preinit_type: str = 'Euclidean') \
        -> torch.Tensor:
    """
    Initialize hyperbolic latent variables to minimize the stress with respect to associated graph distances by
    minimizing the stress loss (Eq. 10 in [1])

    [1] C. Cruceru. "Computationally Tractable Riemannian Manifolds for Graph Embeddings." AAAI, 2021


    Parameters
    ----------
    data: observations in the original space
    latent_dim: dimension of the latent space
    graph_distances: Graph distance matrix for all pairs of data points
    n_steps: number of steps for carrying the optimization
    verbose: If True, print the evolution of the stress loss during the optimization
    preinit_type: pre-initialization approach (based on Euclidean stress loss or random)

    Returns
    -------
    Hyperbolic latent variables minimizing the stress loss

    """

    if preinit_type == 'Euclidean':
        X_eucl = euclidean_stress_loss_initialization(data, latent_dim, graph_distances, verbose=False)
        X_eucl = torch.hstack((torch.zeros_like(X_eucl[:, 0])[:, None], X_eucl))
        X_eucl = X_eucl - X_eucl.mean(dim=0)
        X = exp_map_mu0(X_eucl)
    else:
        sample_mean = torch.zeros(latent_dim + 1)  # Mean on the Lorentz model -> Q+1
        sample_mean[0] = 1.
        sample_scale = 0.1 * torch.ones(latent_dim)  # On the tangent space of the mean on the Lorentz model -> Q
        sample_distribution = LorentzWrappedNormal(sample_mean, sample_scale)
        X = torch.nn.Parameter(sample_distribution.sample((1, data.shape[0])).squeeze())

    X = ManifoldParameter(X, manifold=Lorentz())

    stress = HyperbolicStressLossTermExactMLL(graph_distances)

    optim = RiemannianAdam([{"params": X}], lr=0.01)

    for step in range(n_steps):
        optim.zero_grad()
        loss = -stress.loss(X)
        loss.backward()
        optim.step()
        if verbose and not step % 100:
            print(f"Iter {step + 1}/{n_steps}: {loss.item()}")
    if verbose:
        print(f"Iter {step + 1}/{n_steps}: {loss.item()}")

    return X
