import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.colors as pltc
from mayavi import mlab
from tqdm import trange
import urllib.request
import tarfile

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from geoopt.optim.radam import RiemannianAdam

from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_gplvm_initializations import \
    hyperbolic_tangent_pca_initialization
from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_bayesian_gplvm_models import HyperbolicVariationalBayesianGPLVM
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
    Y = Y[:500]
    labels = labels[:500]

    # Model parameters
    # While we need to specify the dimensionality of the latent variables at the outset, one of the advantages of the
    # Bayesian framework is that by using a ARD kernel we can prune dimensions corresponding to small inverse
    # lengthscales.
    N = len(Y)
    data_dim = Y.shape[1]
    latent_dim = 3  # H2
    n_inducing = 25
    pca = False

    # Initialize the latent variables
    X_init = hyperbolic_tangent_pca_initialization(Y, latent_dim)

    # Model
    model = HyperbolicVariationalBayesianGPLVM(Y, latent_dim, n_inducing, X_init=X_init)

    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)

    # Declaring the objective to be optimised along with optimiser
    # (see models/latent_variable.py for how the additional loss terms are accounted for)
    mll = VariationalELBO(likelihood, model, num_data=len(Y))

    # Plot initialization
    # If the latent space is H2, we plot the embedding in the Poincaré disk
    if latent_dim == 2:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.q_mu).detach().numpy()

        plt.figure(figsize=(8, 8))
        # Plot Poincaré disk
        ax = plt.gca()
        circle = plt.Circle(np.array([0, 0]), radius=1., color='black', fill=False)
        ax.add_patch(circle)

        # Plot points
        colors = ['r', 'b', 'g']
        for i, label in enumerate(np.unique(labels)):
            X_i = x_poincare[labels == label]
            plt.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i]], label=label)

    # If the latent space is H3, we plot the embedding in the Poincaré ball
    elif latent_dim == 3:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.q_mu).detach().numpy()

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
    fit_gplvm_torch(mll, optimizer_cls=RiemannianAdam)  #, options={"maxiter": 1000, "disp": True, "lr": 0.01})
    mll.eval()

    # Plot results
    # If the latent space is H2, we plot the embedding in the Poincaré disk
    if latent_dim == 2:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.q_mu).detach().numpy()

        plt.figure(figsize=(8, 8))
        # Plot Poincaré disk
        ax = plt.gca()
        circle = plt.Circle(np.array([0, 0]), radius=1., color='black', fill=False)
        ax.add_patch(circle)

        # Plot points
        colors = ['r', 'b', 'g']
        for i, label in enumerate(np.unique(labels)):
            X_i = x_poincare[labels == label]
            plt.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i]], label=label)

    # If the latent space is H3, we plot the embedding in the Poincaré ball
    elif latent_dim == 3:
        # From Lorentz to Poincaré
        x_poincare = lorentz_to_poincare(model.X.q_mu).detach().numpy()

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

    print('End')
