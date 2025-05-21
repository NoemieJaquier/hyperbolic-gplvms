import torch
import gpytorch
from torch.autograd.function import FunctionCtx
from HyperbolicEmbeddings.pullback_metric.pullback_metric_tensor_derivatives import analytic_diff_x_jacobian_mean_cov, analytic_diff_x_pullback_metric_tensor, analytic_jacobian_mean_cov, analytic_pullback_metric_tensor
from HyperbolicEmbeddings.pullback_metric.get_train_train_inverse_covariance import get_train_train_inverse_covariance
from HyperbolicEmbeddings.kernels.derivatives.analytic_derivatives_lorentz_heat_kernel_2D import analytic_diff_x_2D_lorentz_heat_kernel, analytic_diff_xx_2D_lorentz_heat_kernel, analytic_diff_xy_2D_lorentz_heat_kernel, analytic_third_derivative_2D_lorentz_heat_kernel


class LorentzPullbackMetric2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, X_new: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                likelihood: gpytorch.likelihoods.Likelihood, kernel_module: gpytorch.kernels.Kernel) -> torch.Tensor:
        """
        GL = mu_J^T@mu_J + D_y*Sigma_J

        Parameters
        -
        X_new: torch.shape([M, 3])   points on the Lorentz manifold
        X: Nx3, Y: NxD_y, likelihood, kernel_module trained GPLVM model building blocks

        Returns
        -
        GL: torch.shape([M, 3, 3])   pullback metric tensor at each point x in X_new
        """
        k_train_inverse = get_train_train_inverse_covariance(X, likelihood, kernel_module).to_dense()  # [N, N]
        tau, lenthscale = kernel_module.outputscale, kernel_module.base_kernel.lengthscale.squeeze()  # [1], [1]
        samples_circle, samples_std_gaussian = kernel_module.base_kernel.samples_circle, kernel_module.base_kernel.samples_std_gaussian
        samples_trunc_gaussian = torch.abs(samples_std_gaussian) / lenthscale  # [L]

        M, N = X_new.shape[0], X.shape[0]
        diff_x_k, diff_xy_k = torch.zeros(M, N, 3),  torch.zeros(M, 3, 3)
        batch_size = 5000
        for i in range(0, M, batch_size):
            X_batch = X_new[i:i+batch_size]
            diff_x_k[i:i+batch_size] = analytic_diff_x_2D_lorentz_heat_kernel(X_batch,
                                                                              X, tau, samples_circle, samples_trunc_gaussian)  # [batch_size, N, 3]
            diff_xy_k[i:i+batch_size] = analytic_diff_xy_2D_lorentz_heat_kernel(X_batch,
                                                                                X_batch, tau, samples_circle, samples_trunc_gaussian)  # [batch_size, 3, 3]

        jacobian_mean, jacobian_cov = analytic_jacobian_mean_cov(diff_x_k, diff_xy_k, k_train_inverse, Y)  # [M, D_y, 3], [M, 3, 3]
        GL = analytic_pullback_metric_tensor(jacobian_mean, jacobian_cov)  # [M, 3, 3]
        ctx.save_for_backward(X_new, X, Y, tau, samples_circle, samples_trunc_gaussian, diff_x_k, k_train_inverse, jacobian_mean)
        return GL

    @staticmethod
    def backward(ctx: FunctionCtx, diff_GL_E: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -
        diff_GL_E: torch.shape([M, 3, 3])

        Returns
        -
        diff_x_E: torch.shape([M, 3])
        """
        X_new, X, Y, tau, samples_circle, samples_trunc_gaussian, diff_x_k, k_train_inverse, jacobian_mean = ctx.saved_tensors

        diff_xx_k = analytic_diff_xx_2D_lorentz_heat_kernel(X_new, X, tau, samples_circle, samples_trunc_gaussian)  # [M, N, 3, 3]
        diff_xyx_k, diff_xyy_k = analytic_third_derivative_2D_lorentz_heat_kernel(
            X_new, X_new, tau, samples_circle, samples_trunc_gaussian)  # [M, 3, 3, 3], [M, 3, 3, 3]
        diff_x_jacobian_mean, diff_x_jacobian_cov = analytic_diff_x_jacobian_mean_cov(
            diff_x_k, diff_xx_k, diff_xyx_k, diff_xyy_k, k_train_inverse, Y)  # [M, D_y, 3, 3], [M, 3, 3, 3]
        diff_x_GL = analytic_diff_x_pullback_metric_tensor(jacobian_mean, diff_x_jacobian_mean, diff_x_jacobian_cov)  # [M, 3, 3, 3]

        diff_x_E = (diff_x_GL * diff_GL_E.unsqueeze(-1)).sum(dim=[1, 2])  # [M, 3]
        return diff_x_E, None, None, None, None
