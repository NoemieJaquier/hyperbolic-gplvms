import json
from pathlib import Path

import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from argparse import ArgumentParser

import gpytorch.priors.torch_priors as torch_priors
from gpytorch.kernels import ScaleKernel, RBFKernel

from geoopt.optim.radam import RiemannianAdam

from HyperbolicEmbeddings.datasets_utils.data_loaders import MnistDataset, mnist_color_function
from HyperbolicEmbeddings.utils.normalization import centering
from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_gplvm_initializations import \
    hyperbolic_tangent_pca_initialization
from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_gplvm_models import MapExactHyperbolicGPLVM, \
    BackConstrainedHyperbolicExactGPLVM
from HyperbolicEmbeddings.gplvm.gplvm_exact_marginal_log_likelihood import GPLVMExactMarginalLogLikelihood
from HyperbolicEmbeddings.gplvm.gplvm_optimization import fit_gplvm_torch, fit_gplvm_torch_with_logger
from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import lorentz_to_poincare, lorentz_distance_torch
from HyperbolicEmbeddings.losses.graph_based_loss import ZeroAddedLossTermExactMLL
from HyperbolicEmbeddings.plot_utils.plots_hyperbolic_gplvms import plot_hyperbolic_gplvm_2d, plot_hyperbolic_gplvm_3d
from HyperbolicEmbeddings.plot_utils.plot_general import plot_distance_matrix
from HyperbolicEmbeddings.utils.logger import compose_loggers, ConsoleLogger, MemoryLogger

from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_manifold import LorentzManifold
import HyperbolicEmbeddings.hyperbolic_manifold.poincare as poincare
import HyperbolicEmbeddings.hyperbolic_manifold.lorentz as lorentz
from HyperbolicEmbeddings.pullback_metric.get_pullback_metric import get_analytic_pullback_metric_lorentz
from HyperbolicEmbeddings.pullback_metric.get_geodesic import get_geodesic,  optimize_geodesic_on_learned_manifold
from HyperbolicEmbeddings.plot_utils.plots_pullback import get_pullback_metric_volume_around_point, plot_2D_metric_volume_around_point
from HyperbolicEmbeddings.kernels.kernels_hyperbolic import LorentzGaussianKernel
from HyperbolicEmbeddings.plot_utils.plot_mnist import plot_predictions

torch.set_default_dtype(torch.float64)
# gpytorch.settings.cholesky_jitter._global_double_value = 1e-4
# gpytorch.settings.cholesky_jitter._global_float_value = 1e-4

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent.resolve()
MODEL_PATH = ROOT_DIR / ('mnist_saved_models_gplvm')
MODEL_PATH.mkdir(exist_ok=True)


def train_gphlvm(latent_dim, model_type, init_type="PCA",
         plot_on=False, load_on=False, outputscale_prior=None, lengthscale_prior=None):

    # Setting manual seed for reproducibility
    seed = 73
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    model_name = 'hyperbolic_gplvm_' + model_type + '_dim' + str(latent_dim) + \
        '_mnist_' + ' '.join(str(c) for c in classes).replace(" ", "") + '_' + str(data_per_class)

    # Initialization
    if init_type == 'PCA':
        X_init = hyperbolic_tangent_pca_initialization(training_data, latent_dim)
    else:
        X_init = None

    # Kernel type
    if latent_dim == 2:
        kernel_type="GeometricKernel"  #  Use the rejection sampling kernel from geometric_kernels, which we use to train the model in our paper. Use this one to exactly reproduce our results. Change to kernel_type="FastHyperbolic" for faster training
    elif latent_dim == 3:
        kernel_type="SlowHyperbolic"  #  Use HyperbolicRiemannianGaussianKernel. This is our old implementation, with which we trained the model of our ICML paper.
    # For any dimension, you can also use this type to train the models faster.
    # kernel_type="FastHyperbolic"  #  Use LorentzGaussianKernel. This is our new implementation, which we use to compute pullback metric. 

    # Model definition
    if model_type == 'MAP':
        # Priors
        if outputscale_prior is not None:
            hyperbolic_kernel_outputscale_prior = torch_priors.GammaPrior(*outputscale_prior)
        else:
            hyperbolic_kernel_outputscale_prior = torch_priors.GammaPrior(5.0, 0.8)
        if lengthscale_prior is not None:
            hyperbolic_kernel_lengthscale_prior = torch_priors.GammaPrior(*lengthscale_prior)
        else:
            hyperbolic_kernel_lengthscale_prior = None

        # Model
        model = MapExactHyperbolicGPLVM(training_data, latent_dim, X_init=X_init, batch_params=False,
                                        kernel_lengthscale_prior=hyperbolic_kernel_lengthscale_prior,
                                        kernel_outputscale_prior=hyperbolic_kernel_outputscale_prior, 
                                        kernel_type=kernel_type
                                        )

    elif model_type == 'BC':
        # Prior on latent variable
        if outputscale_prior is not None:
            hyperbolic_kernel_outputscale_prior = torch_priors.GammaPrior(*outputscale_prior)
        else:
            hyperbolic_kernel_outputscale_prior = None

        if lengthscale_prior is not None:
            hyperbolic_kernel_lengthscale_prior = torch_priors.GammaPrior(*lengthscale_prior)
        else:
            hyperbolic_kernel_lengthscale_prior = torch_priors.GammaPrior(2.0, 2.0)

        # Back constraints
        data_kernel = ScaleKernel(RBFKernel())

        data_kernel.base_kernel.lengthscale = 5.0  # TODO tune 
        data_kernel.outputscale = 1.0

        taxonomy_bc = False

        # Model
        model = BackConstrainedHyperbolicExactGPLVM(training_data, latent_dim, None,
                                                    kernel_lengthscale_prior=hyperbolic_kernel_lengthscale_prior,
                                                    kernel_outputscale_prior=hyperbolic_kernel_outputscale_prior,
                                                    X_init=X_init, batch_params=False,
                                                    taxonomy_based_back_constraints=taxonomy_bc,
                                                    kernel_type=kernel_type
                                                    )

    # Add an extra loss term for the model (can be seen as an additional prior on the latent variable)
    added_loss = ZeroAddedLossTermExactMLL()
    model.add_loss_term(added_loss)

    # Declaring the objective to be optimised along with optimiser
    mll = GPLVMExactMarginalLogLikelihood(model.likelihood, model)

    # Plot initial latent variables
    if plot_on:
        x_latent_init = model.X()
        # From Lorentz to Poincaré
        x_poincare_init = lorentz_to_poincare(x_latent_init).detach().numpy()
        # Get colors
        x_colors = []
        for n in range(N):
            x_colors.append(color_function(nodes_names[n]))

        # If the latent space is H2, we plot the embedding in the Poincaré disk
        if latent_dim == 2:
            # Plot hyperbolic latent space
            plot_hyperbolic_gplvm_2d(x_poincare_init, x_colors, show=True)
    
        # If the latent space is H3, we plot the embedding in the Poincaré ball
        elif latent_dim == 3:
            # Plot hyperbolic latent space
            plot_hyperbolic_gplvm_3d(x_poincare_init, x_colors, show=True)
    
        # Plot distances in the latent space
        distances_latent_init = lorentz_distance_torch(x_latent_init, x_latent_init).detach().numpy()
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
        maxiter = 500
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
        # From Lorentz to Poincaré
        x_latent = mll.model.X()
        x_poincare = lorentz_to_poincare(x_latent).detach().numpy()

        # If the latent space is H2, we plot the embedding in the Poincaré disk
        if latent_dim == 2:
            # Plot hyperbolic latent space
            plot_hyperbolic_gplvm_2d(x_poincare, x_colors, show=True)

        # If the latent space is H3, we plot the embedding in the Poincaré ball
        elif latent_dim == 3:
            # Plot hyperbolic latent space
            plot_hyperbolic_gplvm_3d(x_poincare, x_colors, show=True)

        # Plot distances in the latent space
        distances_latent = lorentz_distance_torch(x_latent, x_latent).detach().numpy()
        max_distance = np.max(distances_latent)
        plot_distance_matrix(distances_latent, max_distance=max_distance, show=True)
        
    return training_data, data_mean, data_std, nodes_names_np, mll, model_name


def pullback_model_and_geodesic(model, model_name, training_data, data_mean, data_std, nodes_names_np, geodesic_idx=[349, 418]):

    X_lorentz = model.X().detach()
    X_poincare = poincare.from_lorentz(X_lorentz)
    latent_dim = X_lorentz.shape[1]-1
    hyperbolic = LorentzManifold(dim=latent_dim)
    x_colors = [mnist_color_function(nodes_names_np[n]) for n in range(len(training_data))]
    
    # Setup the kernel -- we use this kernel for computing the analytical derivatives instead of the kernel from GeometricKernels or the SlowHyperbolic used during training. All kernels return the same value.
    torch.manual_seed(123)
    covar_module = ScaleKernel(LorentzGaussianKernel(dim=latent_dim, nb_points_integral=3000))
    covar_module.outputscale = model.covar_module.outputscale
    covar_module.base_kernel.lengthscale = model.covar_module.base_kernel.lengthscale
    model.covar_module = covar_module

    # Pullback functions
    def lorentz_pullback_metric_function(X_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        G = get_analytic_pullback_metric_lorentz(X_new, X_lorentz, training_data, model.likelihood, covar_module)  # [M, 4, 4]
        P_proj = lorentz.tangent_space_projection_matrix(X_new)  # [M, 4, 4]
        GL_tangent = torch.bmm(torch.bmm(P_proj, G), P_proj.permute(0, 2, 1))  # [M, 4, 4]
        return GL_tangent

    def get_pullback_metric_volume(X_grid_poincare: torch.Tensor) -> torch.Tensor:
        X_grid_lorentz = lorentz.from_poincare(X_grid_poincare)
        pullback_metrics = lorentz_pullback_metric_function(X_grid_lorentz)
        return lorentz.matrix_volume(pullback_metrics)

    # Compute hyperbolic and hyperbolic pullback geodesics
    N_eval = 30
    initial_geodesic_lorentz = get_geodesic(X_lorentz[geodesic_idx[0]], X_lorentz[geodesic_idx[1]], N_eval, hyperbolic)
    geodesic_lorentz_pullback = optimize_geodesic_on_learned_manifold(initial_geodesic_lorentz.clone(), lorentz_pullback_metric_function, hyperbolic,
                                                                      n_steps=200, learning_rate=5e-3, spline_energy_weight=100.)
    geodesic_pullback_poincare = poincare.from_lorentz(geodesic_lorentz_pullback)
    initial_geodesic_poincare = poincare.from_lorentz(initial_geodesic_lorentz)

    # Compute uncertainty along the prediction obtained by decoding the geodesics
    initial_predictions = model(initial_geodesic_lorentz).mean.T
    pullback_predictions = model(geodesic_lorentz_pullback).mean.T
    initial_predictions_stddev = model(initial_geodesic_lorentz).stddev.T
    pullback_predictions_stddev = model(geodesic_lorentz_pullback).stddev.T
    initial_squared_variances = initial_predictions_stddev.detach().cpu().numpy()**2
    pullback_squared_variances = pullback_predictions_stddev.detach().cpu().numpy()**2
    print('Hyperbolic prediction uncertainty: ', np.mean(initial_squared_variances), ' pm ', np.std(initial_squared_variances))
    print('Pullback prediction uncertainty: ', np.mean(pullback_squared_variances), ' pm ', np.std(pullback_squared_variances))

    # Plotting data
    X_grid, metric_volumes = None, None
    if latent_dim == 2:
        X_grid, metric_volumes = get_pullback_metric_volume_around_point(torch.zeros(
            2), get_pullback_metric_volume, grid_size=110, radius=1, restrict_to_unit_circle=True)
    plotting_data = (X_grid, metric_volumes, X_poincare, initial_geodesic_poincare, geodesic_pullback_poincare,
                     initial_predictions, pullback_predictions, x_colors, data_mean, data_std)
    # torch.save(plotting_data, MODEL_PATH / model_name / (str(latent_dim) + 'D_hyperbolic.pth'))

    # Plot pullback and predictions
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 0.2, 0.8])
    ax_latent_space = fig.add_subplot(gs[:, 0])
    fig_initial_predictions = fig.add_subfigure(gs[0, 2])
    fig_pullback_predictions = fig.add_subfigure(gs[1, 2])
    axs_initial_predictions = fig_initial_predictions.subplots(2, 5)
    axs_pullback_predictions = fig_pullback_predictions.subplots(2, 5)
    
    latent_dim = X_poincare.shape[1]

    ax_cbar = plot_2D_metric_volume_around_point(ax_latent_space, X_grid, metric_volumes,
                                                 color_threshold=30000,
                                                 hide_frame=True,
                                                 fontsize=12)
    ax_latent_space.scatter(X_poincare[:, 0], X_poincare[:, 1], c=x_colors, s=8)
    ax_latent_space.plot(initial_geodesic_poincare[:, 0], initial_geodesic_poincare[:, 1],
                         color='black', linestyle='--', linewidth=3, label='hyperbolic geodesic')
    ax_latent_space.plot(geodesic_pullback_poincare[:, 0], geodesic_pullback_poincare[:, 1],
                         color='black', linestyle='-', linewidth=3, label='pullback geodesic')
    ax_cbar.set_yticklabels([0, "5k", "10k", "15k", "20k", "25k", ">30k"])

    # Plot predictions along the geodesic
    plot_predictions(axs_initial_predictions, initial_predictions, data_mean, data_std, 12, yoffset=-0.1)
    plot_predictions(axs_pullback_predictions, pullback_predictions, data_mean, data_std, 12, yoffset=0.05)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", dest="latent_dim", type=int, default=2,
                        help="Set the latent dim (H2 -> 2, H3 -> 3).")
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

    training_data, data_mean, data_std, nodes_names_np, mll, model_name = train_gphlvm(latent_dim, model_type, init_type, plot_on, load_on)

    pullback_model_and_geodesic(mll.model, model_name, training_data, data_mean, data_std, nodes_names_np, geodesic_idx=[349, 418])
