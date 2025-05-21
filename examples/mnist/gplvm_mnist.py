from pathlib import Path

import os
import numpy as np
import torch
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from geoopt.optim.radam import RiemannianAdam
import gpytorch.priors.torch_priors as torch_priors
from gpytorch.kernels import ScaleKernel, RBFKernel

from HyperbolicEmbeddings.datasets_utils.data_loaders import MnistDataset, mnist_color_function
from HyperbolicEmbeddings.utils.normalization import centering
from HyperbolicEmbeddings.gplvm.gplvm_initializations import pca_initialization
from HyperbolicEmbeddings.gplvm.gplvm_models import MapExactGPLVM, BackConstrainedExactGPLVM
from HyperbolicEmbeddings.gplvm.gplvm_exact_marginal_log_likelihood import GPLVMExactMarginalLogLikelihood
from HyperbolicEmbeddings.gplvm.gplvm_optimization import fit_gplvm_torch_with_logger
from HyperbolicEmbeddings.losses.graph_based_loss import ZeroAddedLossTermExactMLL
from HyperbolicEmbeddings.plot_utils.plots_euclidean_gplvms import plot_euclidean_gplvm_2d, plot_euclidean_gplvm_3d
from HyperbolicEmbeddings.plot_utils.plot_general import plot_distance_matrix
from HyperbolicEmbeddings.utils.logger import compose_loggers, ConsoleLogger, MemoryLogger

from HyperbolicEmbeddings.euclidean_manifold.euclidean_manifold import EuclideanManifold
from HyperbolicEmbeddings.pullback_metric.get_pullback_metric import get_analytic_pullback_metric_euclidean
from HyperbolicEmbeddings.pullback_metric.get_geodesic import get_geodesic
from HyperbolicEmbeddings.pullback_metric.get_geodesic_stochman import get_geodesic_on_discrete_learned_manifold
from HyperbolicEmbeddings.plot_utils.plots_pullback import get_pullback_metric_volume_around_point, plot_2D_metric_volume_around_point, get_bounding_box_points
from HyperbolicEmbeddings.plot_utils.plot_mnist import plot_predictions


torch.set_default_dtype(torch.float64)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent.resolve()
MODEL_PATH = ROOT_DIR / ('mnist_saved_models_gplvm')
MODEL_PATH.mkdir(exist_ok=True)


def train_gplvm(latent_dim, model_type, init_type="PCA",
         plot_on=False, load_on=False, outputscale_prior=None, lengthscale_prior=None):

    # Setting manual seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)

    # Load data
    batch_size = 100
    dataset_loader = MnistDataset(batch_size)

    # Load a balanced subset of the MNIST dataset
    classes = [0, 1, 2, 3, 6, 9]
    data_per_class = 100
    training_data, nodes_names = dataset_loader.get_dynamic_binary_mnist_subset(train=True, data_per_class=data_per_class, classes=classes)
    color_function = mnist_color_function
    nodes_names_np = nodes_names.detach().numpy()
    nodes_legend_for_plot = [str(nodes_names_np[i]) for i in range(nodes_names_np.shape[0])]

    # Center
    training_data, data_mean = centering(training_data)
    data_std = training_data.std(dim=0).max()
    training_data = training_data / data_std

    # Model parameters
    N = len(training_data)

    # Model name
    model_name = 'euclidean_gplvm_' + model_type + '_dim' + str(latent_dim) + \
        '_mnist_' + ' '.join(str(c) for c in classes).replace(" ", "") + '_' + str(data_per_class) 

    # Initialization
    if init_type == 'PCA':
        X_init = pca_initialization(training_data, latent_dim)
    else:
        X_init = None

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
        data_kernel.base_kernel.lengthscale = 1.5  # TODO tune
        data_kernel.outputscale = 1.0

        taxonomy_bc = False

        # Kernel
        model = BackConstrainedExactGPLVM(training_data, latent_dim, None,
                                          data_kernel=data_kernel,
                                          kernel_lengthscale_prior=kernel_lengthscale_prior,
                                          kernel_outputscale_prior=kernel_outputscale_prior,
                                          X_init=X_init, batch_params=False,
                                          taxonomy_based_back_constraints=taxonomy_bc)

    # Add an extra loss term for the model (can be seen as an additional prior on the latent variable)
    added_loss = ZeroAddedLossTermExactMLL()
    model.add_loss_term(added_loss)

    # Declaring the objective to be optimised along with optimiser
    mll = GPLVMExactMarginalLogLikelihood(model.likelihood, model)

    # Plot initial latent variables
    if plot_on:
        x_latent_init = model.X()
        x_latent_init_np = model.X().detach().numpy()
        # Get colors
        x_colors = []
        for n in range(N):
            x_colors.append(color_function(nodes_names[n]))

        # Plot the latent space
        if latent_dim == 2:
            plot_euclidean_gplvm_2d(x_latent_init_np, x_colors, show=True)
        elif latent_dim == 3:
            plot_euclidean_gplvm_3d(x_latent_init_np, x_colors, show=True)

        # Plot distances in the latent space
        distances_latent_init = model.covar_module.covar_dist(x_latent_init, x_latent_init).detach().numpy()
        plot_distance_matrix(distances_latent_init, show=True)

    # Training / loading the model
    mll.train()
    # If the model exists, load it
    if load_on and os.path.isfile(MODEL_PATH / model_name / "model_best_state_dict.pth"):
        print('Model loading ...')
        mll.load_state_dict(torch.load(MODEL_PATH / model_name / "model_best_state_dict.pth")) 
        print("Loaded saved model.")
    # Otherwise train the model in a single batch with automatic convergence checks 
    else:
        # Hyperparameters and logger
        maxiter = 1000
        lr = 0.05
        model_path = MODEL_PATH / model_name
        model_path.mkdir(exist_ok=True)
        logger = compose_loggers(ConsoleLogger(n_logs=maxiter), MemoryLogger(model_path))
        # Training
        print('Model training ...')
        try:
            fit_gplvm_torch_with_logger(mll, optimizer_cls=RiemannianAdam, logger=logger, options={"maxiter": maxiter, "lr": lr}, stop_crit_options={"maxiter": maxiter})

            logger.print_full_iteration('Best Iteration', logger.best_step, logger.best_loss, logger.best_parameters)
            mll.load_state_dict(logger.best_state_dict)
            print("Best iteration was achieved at step " + str(logger.best_step) + " with the best loss at " + str(logger.best_loss))
        except KeyboardInterrupt:
            pass
    mll.eval()

    # Test evaluation
    mll.model.eval()
    posterior = mll.model(mll.model.X())
    error = np.abs((posterior.mean.T - training_data).detach().numpy())
    print("Error mean and std: ", np.mean(error), np.std(error))


    # Plot results
    if plot_on:
        x_latent = mll.model.X()
        x_latent_np = x_latent.detach().numpy()

        # Plot trained latent space 
        if latent_dim == 2:
            plot_euclidean_gplvm_2d(x_latent_np, x_colors, show=True)

        elif latent_dim == 3:
            plot_euclidean_gplvm_3d(x_latent_np, x_colors, show=True)

        # Plot distances in the latent space
        distances_latent = model.covar_module.covar_dist(x_latent, x_latent).detach().numpy()

        max_distance = np.max(distances_latent)
        plot_distance_matrix(distances_latent, max_distance=max_distance, show=True)
    
    return training_data, data_mean, data_std, nodes_names_np, mll, model_name


def pullback_model_and_geodesic(model, model_name, training_data, data_mean, data_std, nodes_names_np, geodesic_idx=[349, 418]):

    X_eucl = model.X().detach()
    x_colors = [mnist_color_function(nodes_names_np[n]) for n in range(len(training_data))]
    latent_dim = X_eucl.shape[1]
    euclidean = EuclideanManifold(dim=latent_dim)

    # Pullback functions
    def euclidean_pullback_metric_function(X_new: torch.Tensor) -> torch.Tensor:
        return get_analytic_pullback_metric_euclidean(X_new, X_eucl, training_data, model.likelihood, model.covar_module)

    def get_pullback_metric_volume(X_grid: torch.Tensor) -> torch.Tensor:
        pullback_metrics = euclidean_pullback_metric_function(X_grid)
        return torch.Tensor([G.det().sqrt().item() for G in pullback_metrics])

    # Compute Euclidean and Euclidean pullback geodesics
    N_eval = 30
    initial_geodesic_euclidean = get_geodesic(X_eucl[geodesic_idx[0]], X_eucl[geodesic_idx[1]], N_eval, euclidean)
    geodesic_euclidean_pullback = get_geodesic_on_discrete_learned_manifold(initial_geodesic_euclidean, X_eucl, euclidean_pullback_metric_function,
                                                            n_eval=N_eval, grid_size=32,
                                                            num_nodes_to_optimize=N_eval)

    # Compute uncertainty along the prediction obtained by decoding the geodesics
    initial_predictions = model(initial_geodesic_euclidean).mean.T
    pullback_predictions = model(geodesic_euclidean_pullback).mean.T
    initial_predictions_stddev = model(initial_geodesic_euclidean).stddev.T
    pullback_predictions_stddev = model(geodesic_euclidean_pullback).stddev.T
    initial_squared_variances = initial_predictions_stddev.detach().cpu().numpy()**2
    pullback_squared_variances = pullback_predictions_stddev.detach().cpu().numpy()**2
    print('Euclidean prediction uncertainty: ', np.mean(initial_squared_variances), ' pm ', np.std(initial_squared_variances))
    print('Pullback prediction uncertainty: ', np.mean(pullback_squared_variances), ' pm ', np.std(pullback_squared_variances))

    # Plotting data
    X_grid, metric_volumes = None, None
    if latent_dim == 2:
        X_grid = get_bounding_box_points(X_eucl, grid_size=120)  # , padding_percent=0.05)
        metric_volumes = get_pullback_metric_volume(X_grid)
    plotting_data = (X_grid, metric_volumes, X_eucl, training_data, initial_geodesic_euclidean, geodesic_euclidean_pullback,
                     initial_predictions, pullback_predictions,
                     x_colors, data_mean, data_std)  
    # torch.save(plotting_data, MODEL_PATH / model_name / (str(latent_dim) + 'D_euclidean.pth'))  
    
    # Plot pullback and predictions
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 0.2, 0.8])
    ax_latent_space = fig.add_subplot(gs[:, 0])
    fig_initial_predictions = fig.add_subfigure(gs[0, 2])
    fig_pullback_predictions = fig.add_subfigure(gs[1, 2])
    axs_initial_predictions = fig_initial_predictions.subplots(2, 5)
    axs_pullback_predictions = fig_pullback_predictions.subplots(2, 5)
    
    latent_dim = X_eucl.shape[1]

    ax_cbar = plot_2D_metric_volume_around_point(ax_latent_space, X_grid, metric_volumes,
                                                 color_threshold=14000,
                                                 hide_frame=True,
                                                 fontsize=12)
    ax_latent_space.scatter(X_eucl[:, 0], X_eucl[:, 1], c=x_colors, s=8)
    ax_latent_space.plot(initial_geodesic_euclidean[:, 0], initial_geodesic_euclidean[:, 1],
                         color='black', linestyle='--', linewidth=2, label='euclidean geodesic')
    ax_latent_space.plot(geodesic_euclidean_pullback[:, 0], geodesic_euclidean_pullback[:, 1],
                         color='black', linestyle='-', linewidth=2, label='pullback geodesic')
    ax_cbar.set_yticklabels([0, "2k", "4k", "6k", "8k", "10k", "12k", ">14k"])

    # Plot predictions along the geodesic
    plot_predictions(axs_initial_predictions, initial_predictions, data_mean, data_std, 12, yoffset=-0.1)
    plot_predictions(axs_pullback_predictions, pullback_predictions, data_mean, data_std, 12, yoffset=0.05)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", dest="latent_dim", type=int, default=2,
                        help="Set the latent dim.")
    parser.add_argument("--model_type", dest="model_type", default="MAP",
                        help="Set the model type. Options: MAP, BC.")
    parser.add_argument("--init_type", dest="init_type", type=str, default="PCA",
                        help="Set the GPLVM initialization. Options: Random, PCA.")
    parser.add_argument("--plot_on", dest="plot_on", type=bool, default=False,
                        help="If True, generate plots.")
    parser.add_argument("--load_on", dest="load_on", type=bool, default=True,
                        help="If True, load existing model.")

    args = parser.parse_args()

    latent_dim = args.latent_dim
    model_type = args.model_type
    init_type = args.init_type
    plot_on = args.plot_on
    load_on = args.load_on

    training_data, data_mean, data_std, nodes_names_np, mll, model_name = train_gplvm(latent_dim, model_type, init_type, plot_on, load_on)

    pullback_model_and_geodesic(mll.model, model_name, training_data, data_mean, data_std, nodes_names_np, geodesic_idx=[349, 418])
