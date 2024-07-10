import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.colors as pltc
from mayavi import mlab
from tqdm import trange
import urllib.request
import tarfile

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

from geoopt.optim.radam import RiemannianAdam

from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_gplvm_initializations import \
    hyperbolic_tangent_pca_initialization
from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_gplvm_models import MapExactHyperbolicGPLVM
from HyperbolicEmbeddings.gplvm.gplvm_exact_marginal_log_likelihood import GPLVMExactMarginalLogLikelihood
from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import lorentz_to_poincare
from HyperbolicEmbeddings.gplvm.gplvm_optimization import fit_gplvm_torch

torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    # Setting manual seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)

    # Set up training data
    # We use the canonical multi-phase oilflow dataset used in Titsias & Lawrence, 2010 that consists of 1000,
    # 12 dimensional observations belonging to three known classes corresponding to different phases of oilflow.
    url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    urllib.request.urlretrieve(url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')

    Y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
    labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
    labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)

    # Remove some data
    Y = Y[:200]
    labels = labels[:200]

    # Model parameters
    N = len(Y)
    latent_dim = 3

    # Initialize the latent variables
    X_init = hyperbolic_tangent_pca_initialization(Y, latent_dim)

    # Model
    model = MapExactHyperbolicGPLVM(Y, latent_dim, X_init=X_init, batch_params=False)

    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)

    # Declaring the objective to be optimised along with optimiser
    mll = GPLVMExactMarginalLogLikelihood(model.likelihood, model)

    # Plot initial latent variables
    if latent_dim == 2:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.X).detach().numpy()

        plt.figure(figsize=(8, 8))
        # Plot Poincaré disk
        ax = plt.gca()
        circle = plt.Circle(np.array([0, 0]), radius=1., color='black', fill=False)
        ax.add_patch(circle)
        plt.axis('off')

        # Plot points
        colors = ['r', 'b', 'g']
        for i, label in enumerate(np.unique(labels)):
            X_i = x_poincare[labels == label]
            plt.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i]], label=label)
        plt.show()

    # If the latent space is H3, we plot the embedding in the Poincaré ball
    elif latent_dim == 3:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.X).detach().numpy()

        # Mayavi plot of the Poincare ball
        num_pts = 200
        u = np.linspace(0, 2 * np.pi, num_pts)
        v = np.linspace(0, np.pi, num_pts)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
        mlab.clf()
        mlab.mesh(x, y, z, color=(0.7, 0.7, 0.7), opacity=0.1)
        mlab.points3d(0, 0, 0, color=pltc.to_rgb('black'), scale_factor=0.05)  # Plot center

        # Plot points
        colors = ['r', 'b', 'g']
        for i, label in enumerate(np.unique(labels)):
            X_i = x_poincare[labels == label]
            mlab.points3d(X_i[:, 0], X_i[:, 1], X_i[:, 2], color=pltc.to_rgb(colors[i]), scale_factor=0.05)

        mlab.show()

    # Train the model in a single batch with automatic convergence checks
    mll.train()
    fit_gplvm_torch(mll, optimizer_cls=RiemannianAdam, options={"maxiter": 1000})
    mll.eval()

    # Test evaluation
    model.eval()
    posterior = model(model.X.X)
    error = (posterior.mean.T - Y).detach().numpy()

    # Plot results
    # If the latent space is H2, we plot the embedding in the Poincaré disk
    if latent_dim == 2:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.X).detach().numpy()

        plt.figure(figsize=(8, 8))
        # Plot Poincaré disk
        ax = plt.gca()
        circle = plt.Circle(np.array([0, 0]), radius=1., color='black', fill=False)
        ax.add_patch(circle)
        plt.axis('off')

        # Plot points
        colors = ['r', 'b', 'g']
        for i, label in enumerate(np.unique(labels)):
            X_i = x_poincare[labels == label]
            plt.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i]], label=label)

        plt.show()

    # If the latent space is H3, we plot the embedding in the Poincaré ball
    elif latent_dim == 3:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.X).detach().numpy()

        # Mayavi plot of the Poincare ball
        num_pts = 200
        u = np.linspace(0, 2 * np.pi, num_pts)
        v = np.linspace(0, np.pi, num_pts)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
        mlab.clf()
        mlab.mesh(x, y, z, color=(0.7, 0.7, 0.7), opacity=0.1)

        # Plot points
        colors = ['r', 'b', 'g']
        for i, label in enumerate(np.unique(labels)):
            X_i = x_poincare[labels == label]
            mlab.points3d(X_i[:, 0], X_i[:, 1], X_i[:, 2], color=pltc.to_rgb(colors[i]), scale_factor=0.05)

        mlab.show()
