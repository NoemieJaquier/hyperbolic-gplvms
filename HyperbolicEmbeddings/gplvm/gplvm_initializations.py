import torch
from torch.optim.adam import Adam

from HyperbolicEmbeddings.losses.graph_based_loss import EuclideanStressLossTermExactMLL


def pca_initialization(data: torch.Tensor, latent_dim: int):
    """
    Compute the initialization of the latent variables using PCA.

    Parameters
    ----------
    :param data: observations  [nb_data x dimension]
    :param latent_dim: dimension of the latent space

    Return
    ------
    :return initial latent variables [nb_data x latent dimension]
    """
    U, S, V = torch.pca_lowrank(data, q=latent_dim)
    return torch.nn.Parameter(torch.matmul(data, V[:, :latent_dim]))


def euclidean_stress_loss_initialization(data: torch.Tensor, latent_dim: int, graph_distances: torch.Tensor,
                               n_steps=1000, verbose: bool = True, preinit_type: str = 'PCA') -> torch.Tensor:
    """
    Initialize Euclidean latent variables to minimize the stress with respect to associated graph distances by
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
    Euclidean latent variables minimizing the stress loss

    """

    if preinit_type == 'PCA':
        X = pca_initialization(data, latent_dim)
    else:
        X = torch.nn.Parameter(torch.randn(data.shape[0], latent_dim))

    stress = EuclideanStressLossTermExactMLL(graph_distances)

    optim = Adam([{"params": X}], lr=0.01)

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

