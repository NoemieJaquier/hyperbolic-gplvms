import torch
import gpytorch
from torch.autograd.function import FunctionCtx
from HyperbolicEmbeddings.kernels.derivatives.analytic_derivatives_lorentz_heat_kernel_3D import analytic_diff_x_3D_hyperbolic_heat_kernel_batched, analytic_diff_xx_3D_hyperbolic_heat_kernel_batched, analytic_diff_xy_3D_hyperbolic_heat_kernel, analytic_third_derivative_3D_hyperbolic_heat_kernel
from HyperbolicEmbeddings.pullback_metric.pullback_metric_tensor_derivatives import analytic_diff_x_jacobian_mean_cov, analytic_diff_x_pullback_metric_tensor, analytic_jacobian_mean_cov, analytic_pullback_metric_tensor
from HyperbolicEmbeddings.pullback_metric.get_train_train_inverse_covariance import get_train_train_inverse_covariance


class LorentzPullbackMetric3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, X_new: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                likelihood: gpytorch.likelihoods.Likelihood, kernel_module: gpytorch.kernels.Kernel) -> torch.Tensor:
        """
        GL = mu_J^T@mu_J + D_y*Sigma_J

        Parameters
        -
        X_new: torch.shape([M, 4])   points on the Lorentz manifold
        X: Nx4, Y: NxD_y, likelihood, kernel_module trained GPLVM model building blocks

        Returns
        -
        GL: torch.shape([M, 4, 4])   pullback metric tensor at each point x in X_new
        """
        k_train_inverse = get_train_train_inverse_covariance(X, likelihood, kernel_module).to_dense()  # [N, N]
        tau, kappa = kernel_module.outputscale, 2*kernel_module.base_kernel.lengthscale.squeeze()**2

        diff_x_k = analytic_diff_x_3D_hyperbolic_heat_kernel_batched(X_new, X, tau, kappa)  # [M, N, 4]
        diff_xy_k = analytic_diff_xy_3D_hyperbolic_heat_kernel(X_new, X_new, tau, kappa)  # [M, 4, 4]
        jacobian_mean, jacobian_cov = analytic_jacobian_mean_cov(diff_x_k, diff_xy_k, k_train_inverse, Y)  # [M, D_y, 4], [M, 4, 4]
        GL = analytic_pullback_metric_tensor(jacobian_mean, jacobian_cov)  # [M, 4, 4]
        ctx.save_for_backward(X_new, X, Y, tau, kappa, diff_x_k, k_train_inverse, jacobian_mean)
        return GL

    @staticmethod
    def backward(ctx: FunctionCtx, diff_GL_E: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -
        diff_GL_E: torch.shape([M, 4, 4])

        Returns
        -
        diff_x_E: torch.shape([M, 4])
        """
        X_new, X, Y, tau, kappa, diff_x_k, k_train_inverse, jacobian_mean = ctx.saved_tensors

        diff_xx_k = analytic_diff_xx_3D_hyperbolic_heat_kernel_batched(X_new, X, tau, kappa)  # [M, N, 4, 4]
        diff_xyx_k, diff_xyy_k = analytic_third_derivative_3D_hyperbolic_heat_kernel(X_new, X_new, tau, kappa)  # [M, 4, 4, 4], [M, 4, 4, 4]
        diff_x_jacobian_mean, diff_x_jacobian_cov = analytic_diff_x_jacobian_mean_cov(
            diff_x_k, diff_xx_k, diff_xyx_k, diff_xyy_k, k_train_inverse, Y)  # [M, D_y, 4, 4], [M, 4, 4, 4]
        diff_x_GL = analytic_diff_x_pullback_metric_tensor(jacobian_mean, diff_x_jacobian_mean, diff_x_jacobian_cov)  # [M, 4, 4, 4]

        diff_x_E = (diff_x_GL * diff_GL_E.unsqueeze(-1)).sum(dim=[1, 2])  # [M, D_x]
        return diff_x_E, None, None, None, None
