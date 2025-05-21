from pathlib import Path

import os
import numpy as np
import torch
from argparse import ArgumentParser

from geoopt.optim.radam import RiemannianAdam
import gpytorch.priors.torch_priors as torch_priors
from gpytorch.kernels import ScaleKernel, RBFKernel

from HyperbolicEmbeddings.taxonomy_utils.data_loaders import load_taxonomy_data
from HyperbolicEmbeddings.gplvm.gplvm_initializations import euclidean_stress_loss_initialization
from HyperbolicEmbeddings.gplvm.gplvm_models import MapExactGPLVM, BackConstrainedExactGPLVM
from HyperbolicEmbeddings.gplvm.gplvm_exact_marginal_log_likelihood import GPLVMExactMarginalLogLikelihood
from HyperbolicEmbeddings.gplvm.gplvm_optimization import fit_gplvm_torch
from HyperbolicEmbeddings.kernels.kernels_graph import GraphMaternKernel
from HyperbolicEmbeddings.losses.graph_based_loss import ZeroAddedLossTermExactMLL, EuclideanStressLossTermExactMLL
from HyperbolicEmbeddings.plot_utils.plots_euclidean_gplvms import plot_euclidean_gplvm_2d, plot_euclidean_gplvm_3d
from HyperbolicEmbeddings.plot_utils.plot_general import plot_distance_matrix
from HyperbolicEmbeddings.taxonomy_utils.taxonomy_functions import reorder_taxonomy_data, reorder_distance_matrix
from HyperbolicEmbeddings.taxonomy_utils.grasp_data_utils import HAND_GRASPS_NAMES


torch.set_default_dtype(torch.float64)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent.resolve()


def main(latent_dim, model_type, loss_type='Zero', loss_scale=0.0, plot_on=True, load_on=True,
         outputscale_prior=None, lengthscale_prior=None):

    # Setting manual seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)

    # Select dataset
    dataset = "grasps"

    # Paths
    MODEL_PATH = ROOT_DIR / (dataset + '_saved_models_gplvm')
    MODEL_PATH.mkdir(exist_ok=True)

    # Load data
    training_data, data_mean, adjacency_matrix, graph_distances, nodes_names, indices_in_graph, nodes_legend_for_plot, \
    color_function, max_manifold_distance, joint_names = load_taxonomy_data(dataset)

    # Model name
    model_name = 'euclidean_gplvm_' + model_type + '_dim' + str(latent_dim) + '_' + dataset + '_'
    model_name += loss_type
    if loss_type != 'Zero':
        model_name += str(loss_scale)

    # Model parameters
    N = len(training_data)

    # Initialization
    print('Initialization ...')
    X_init = euclidean_stress_loss_initialization(training_data, latent_dim, graph_distances)

    # Model definition
    if model_type == 'MAP':
        # Priors
        if outputscale_prior is not None:
            kernel_outputscale_prior = torch_priors.GammaPrior(*outputscale_prior)
        else:
            kernel_outputscale_prior = None

        if lengthscale_prior is not None:
            kernel_lengthscale_prior = torch_priors.GammaPrior(*lengthscale_prior)
        else:
            kernel_lengthscale_prior = None

        # Model
        model = MapExactGPLVM(training_data, latent_dim, X_init=X_init, batch_params=False,
                              kernel_lengthscale_prior=kernel_lengthscale_prior,
                              kernel_outputscale_prior=kernel_outputscale_prior)
    elif model_type == 'BC':
        # Prior on latent variable
        if outputscale_prior is not None:
            kernel_outputscale_prior = torch_priors.GammaPrior(*outputscale_prior)
        else:
            kernel_outputscale_prior = None

        if lengthscale_prior is not None:
            kernel_lengthscale_prior = torch_priors.GammaPrior(*lengthscale_prior)
        else:
            kernel_lengthscale_prior = None

        # Back constraints
        data_kernel = ScaleKernel(RBFKernel())
        classes_kernel = ScaleKernel(GraphMaternKernel(adjacency_matrix, nu=2.5))

        if loss_type == 'Stress':
            data_kernel.base_kernel.lengthscale = 1.8
            classes_kernel.base_kernel.lengthscale = 1.5

            data_kernel.outputscale = 2.0
            classes_kernel.outputscale = 1.0

            taxonomy_bc = True

        elif loss_type == 'Zero':
            data_kernel.base_kernel.lengthscale = 1.8
            classes_kernel.base_kernel.lengthscale = 1.5
            data_kernel.outputscale = 2.0
            classes_kernel.outputscale = 1.0

            taxonomy_bc = False

        # Kernel
        model = BackConstrainedExactGPLVM(training_data, latent_dim, indices_in_graph,
                                          data_kernel=data_kernel, classes_kernel=classes_kernel,
                                          kernel_lengthscale_prior=kernel_lengthscale_prior,
                                          kernel_outputscale_prior=kernel_outputscale_prior,
                                          X_init=X_init, batch_params=False,
                                          taxonomy_based_back_constraints=taxonomy_bc)

    # Add an extra loss term for the model (can be seen as an additional prior on the latent variable)
    if loss_type == 'Zero' or None:
        print("Loss_type = Zero")
        added_loss = ZeroAddedLossTermExactMLL()
    elif loss_type == 'Stress':
        print("Loss_type = Stress")
        added_loss = EuclideanStressLossTermExactMLL(graph_distances, loss_scale)
    else:
        raise NotImplementedError

    model.add_loss_term(added_loss)

    # Declaring the objective to be optimised along with optimiser
    mll = GPLVMExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()

    if load_on:
        print('Model loading ...')
        # Load parameters from file
        mll.load_state_dict(torch.load(MODEL_PATH / model_name))
    else:
        # Train the model in a single batch
        print('Optimization ...')
        fit_gplvm_torch(mll, optimizer_cls=RiemannianAdam, model_path=MODEL_PATH / model_name
                        , options={"maxiter": 1000, "disp": True, "lr": 0.01})

    mll.eval()

    # Plot results
    if plot_on:
        x_latent = model.X()
        x_latent_np = x_latent.detach().numpy()

        # Get colors
        x_colors = []
        for n in range(N):
            x_colors.append(color_function(nodes_names[n]))

        # Compute geodesics
        geodesics = []
        geodesic_idx = [[0, 30],  # ET -> PE
                        [90, 35],  # WT -> PD
                        [70, 84],  # SD -> Tr
                        [65, 13],  # Ri -> IE
                        [62, 30],  # Qu -> PE
                        [80, 5],  # TP -> FH
                        [50, 40],  # PF -> PS
                        ]
        nb_points_geodesic = 50
        for idx in geodesic_idx:
            t = torch.linspace(0., 1., nb_points_geodesic)[:, None]
            geodesic = x_latent[idx[0]] + t * (x_latent[idx[1]] - x_latent[idx[0]])
            geodesics.append(geodesic.detach().numpy())

        # If the latent space is H2, we plot the embedding in the Poincaré disk
        if latent_dim == 2:
            # Plot euclidean latent space
            plot_euclidean_gplvm_2d(x_latent_np, x_colors, geodesics=geodesics, show=True)

        # If the latent space is H3, we plot the embedding in the Poincaré ball
        elif latent_dim == 3:
            # Plot hyperbolic latent space
            plot_euclidean_gplvm_3d(x_latent_np, x_colors, geodesics=geodesics, show=True)

        # Plot error between distances in the latent space and original graph distances
        x_latent_ordered = torch.vstack(reorder_taxonomy_data(x_latent, nodes_names, HAND_GRASPS_NAMES))
        x_colors_ordered = reorder_taxonomy_data(x_colors, nodes_names, HAND_GRASPS_NAMES)
        distances_latent = model.covar_module.covar_dist(x_latent_ordered, x_latent_ordered).detach().numpy()
        graph_distances_ordered = reorder_distance_matrix(graph_distances, nodes_names, HAND_GRASPS_NAMES)
        plot_distance_matrix(np.abs(distances_latent - graph_distances_ordered), x_colors=x_colors_ordered, show=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", dest="latent_dim", type=int, default=3,
                        help="Set the latent dim.")
    parser.add_argument("--model_type", dest="model_type", default="BC",
                        help="Set the model type. Options: MAP, BC.")
    parser.add_argument("--loss_type", dest="loss_type", default="Stress",
                        help="Set the loss type. Options: Zero, Stress, Distortion.")
    parser.add_argument("--loss_scale", dest="loss_scale", type=float, default=3000.0,
                        help="Set the loss scale.")
    parser.add_argument("--load_on", dest="load_on", type=bool, default=False,
                        help="If True, load exisiting model.")
    parser.add_argument("--plot_on", dest="plot_on", type=bool, default=True,
                        help="If True, generate plots.")

    args = parser.parse_args()

    latent_dim = args.latent_dim
    model_type = args.model_type
    loss_type = args.loss_type
    loss_scale = args.loss_scale
    load_on = args.load_on
    plot_on = args.plot_on

    main(latent_dim, model_type, loss_type, loss_scale, plot_on, load_on)
