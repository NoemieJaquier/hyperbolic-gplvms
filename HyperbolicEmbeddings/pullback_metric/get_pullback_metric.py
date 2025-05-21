import torch
import gpytorch

from HyperbolicEmbeddings.kernels.derivatives.analytic_derivatives_euclidean_rbf_kernel import analytic_diff_x_euclidean_rbf_kernel_batched, analytic_diff_xy_euclidean_rbf_kernel
from HyperbolicEmbeddings.kernels.derivatives.torch_derivatives import diff_1st_kernel, diff_2nd_kernel
from HyperbolicEmbeddings.pullback_metric.lorentz_pullback_metric_2D import LorentzPullbackMetric2D
from HyperbolicEmbeddings.pullback_metric.lorentz_pullback_metric_3D import LorentzPullbackMetric3D
from HyperbolicEmbeddings.pullback_metric.pullback_metric_tensor_derivatives import analytic_jacobian_mean_cov, analytic_pullback_metric_tensor
from HyperbolicEmbeddings.pullback_metric.get_train_train_inverse_covariance import get_train_train_inverse_covariance


def get_pullback_metric(X_new: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                        likelihood: gpytorch.likelihoods.Likelihood, kernel_module: gpytorch.kernels.Kernel) -> torch.Tensor:
    """
    This function computes the pullback metric tensor at each point x in X_new using pytorch. 
    Since pytorch is very slow at computing the necessary derivatives it is recommended to use the analytic versions instead.

    GL = mu_J^T@mu_J + D_y*Sigma_J

    Parameters
    -
    X_new: [M, D_x]  points on the Lorentz manifold (assumed to require grad)
    X: NxD_x, Y: NxD_y, likelihood, kernel_module trained GPLVM model

    Returns
    -
    GL: torch.shape([M, D_x, D_x])   pullback metric tensor at each point x in X_new
    """
    k_train_inverse = get_train_train_inverse_covariance(X, likelihood, kernel_module).to_dense().detach()  # [N, N]

    # first derivative
    kernel_xX = kernel_module(X_new, X).to_dense()  # [M, N]
    diff_x_k = diff_1st_kernel(kernel_xX, X_new)  # [M, N, D_x]

    # second derivative
    X_new1 = X_new.clone()
    X_new2 = X_new.clone()
    kernel_xx = kernel_module(X_new1, X_new2, diag=True)  # [M]
    diff_xy_k = diff_2nd_kernel(diff_1st_kernel(kernel_xx, X_new1), X_new2)  # [M, D_x, D_x]

    jacobian_mean, jacobian_cov = analytic_jacobian_mean_cov(diff_x_k, diff_xy_k, k_train_inverse, Y)  # [M, D_y, D_x], [M, D_x, D_x]
    return analytic_pullback_metric_tensor(jacobian_mean, jacobian_cov)  # [M, 4, 4]


def get_analytic_pullback_metric_euclidean(X_new: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                                           likelihood: gpytorch.likelihoods.Likelihood, kernel_module: gpytorch.kernels.Kernel) -> torch.Tensor:
    """
    analytical version of get_pullback_metric() for the euclidean manifold.
    """
    k_train_inverse = get_train_train_inverse_covariance(X, likelihood, kernel_module).to_dense().detach()  # [N, N]
    tau, kappa = kernel_module.outputscale, 2*kernel_module.base_kernel.lengthscale.squeeze()**2

    diff_x_k = analytic_diff_x_euclidean_rbf_kernel_batched(X_new, X, tau, kappa)  # [M, N, D_x]
    diff_xy_k = analytic_diff_xy_euclidean_rbf_kernel(X_new, X_new, tau, kappa)  # [M, D_x, D_x]
    jacobian_mean, jacobian_cov = analytic_jacobian_mean_cov(diff_x_k, diff_xy_k, k_train_inverse, Y)  # [M, D_y, D_x], [M, D_x, D_x]
    return analytic_pullback_metric_tensor(jacobian_mean, jacobian_cov)  # [M, D_x, D_x]


def get_analytic_pullback_metric_lorentz(X_new: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                                         likelihood: gpytorch.likelihoods.Likelihood, kernel_module: gpytorch.kernels.Kernel) -> torch.Tensor:
    """
    The get_pullback_metric() function does not work at all for the 3 dimensional Lorentz manifold due to numerical issues.
    For the 2 dimensional Lorentz manifold it works but is slow and extremely memory intensive since pytorch does not
    sum over the integral sample dimension L in an efficient way.
    This function computes the analytic derivatives for the 2D and 3D Lorentz manifold.
    """
    dim = X_new.shape[1] - 1
    if dim == 2:
        return LorentzPullbackMetric2D.apply(X_new, X, Y, likelihood, kernel_module)  # [M, 3, 3]
    if dim == 3:
        return LorentzPullbackMetric3D.apply(X_new, X, Y, likelihood, kernel_module)  # [M, 4, 4]
    raise ValueError(f"pullback metric supports only 2- and 3-dimensional Lorentz Manifold. Given was: {dim}-dimensional")
