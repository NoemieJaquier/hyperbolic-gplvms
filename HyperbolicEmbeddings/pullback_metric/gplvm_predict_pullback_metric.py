import torch
from typing import Tuple, Literal
import gpytorch
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import MatmulLinearOperator
from linear_operator import to_dense
from functools import lru_cache
from torch.autograd import grad

@lru_cache()
def get_train_train_inverse_covariance(x: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood,
                                       kernel_module: gpytorch.kernels.Kernel):
    gp_prior = MultivariateNormal(torch.zeros(x.shape[0]), kernel_module(x))
    train_train_covar = likelihood(gp_prior).lazy_covariance_matrix
    with gpytorch.settings.max_root_decomposition_size(3000):
        train_train_covar_inv_root = to_dense(train_train_covar.root_inv_decomposition().root)
    return MatmulLinearOperator(train_train_covar_inv_root, train_train_covar_inv_root.transpose(-1, -2))


def get_pullback_metric_function(x: torch.Tensor, y: torch.Tensor, likelihood: Likelihood, kernel: Kernel):
    for params in [*kernel.parameters(), *likelihood.parameters()]:
        params.requires_grad_(False)
    return lambda X_new: predict_pullback_metric(X_new, x, y, likelihood, kernel)


def predict_pullback_metric(x_grid: torch.Tensor, x: torch.Tensor, y: torch.Tensor, likelihood: Likelihood,
                            kernel: Kernel) -> torch.Tensor:
    """
    Returns the expected pullback metric according to (Tosi14: Metrics for Probabilistic Geometries).
    E[G] = E[J]^T @ E[J] + D_y*Sigma_J

    Parameters
    -
    X_grid: torch.Shape([M, D_x])
    X: torch.shape([N, D_x])
    Y: torch.shape([N, D_y])

    Returns
    -
    G: torch.Shape([M, D_x, D_x])
    """
    D_y = y.shape[1]
    jacobian_mean, jacobian_cov = predict_jacobian(x_grid, x, y, likelihood, kernel)
    return torch.bmm(jacobian_mean.permute(0, 2, 1), jacobian_mean) + D_y * jacobian_cov


def predict_jacobian(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor, likelihood: Likelihood, kernel: Kernel) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Predicts the transpose of the Jacobian at given new inputs. 

    Parameters
    -
    X_new: torch.shape([M, D_x])
    X: torch.shape([N, D_x])
    Y: torch.shape([N, D_y])

    Returns
    -
    jacobian_mean: torch.shape([M, D_y, D_x])
    jacobian_cov: torch.shape([M, D_x, D_x])
    """
    M = x_new.shape[0]
    train_train_inverse = get_train_train_inverse_covariance(x, likelihood, kernel).to_dense()
    # train_train_inverse = torch.linalg.inv(kernel(X).to_dense() + likelihood.noise**2 * torch.eye(X.shape[0]))

    # partial_test_train, partial_test_test = torch_derivatives(kernel, x_new, x)
    partial_test_train, partial_test_test = analytic_SE_derivatives(kernel, x_new, x)

    covar_correction = torch.bmm(partial_test_train, train_train_inverse.expand(M, *train_train_inverse.shape))
    jacobian_mean = torch.bmm(covar_correction, y.unsqueeze(0).repeat(M, 1, 1))
    jacobian_cov = partial_test_test - torch.bmm(covar_correction, partial_test_train.permute(0, 2, 1))
    jacobian_cov = (jacobian_cov + jacobian_cov.permute(0, 2, 1)) / 2

    return jacobian_mean.permute(0, 2, 1), jacobian_cov


def torch_derivatives(kernel: Kernel, x_new: torch.Tensor, x: torch.Tensor):
    def f(x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        return kernel(x.unsqueeze(0), x_prime.unsqueeze(0)).to_dense().squeeze()

    def jacobian(x: torch.Tensor) -> torch.Tensor:
        jacobian_components = [grad(f(x, x_prime), x, create_graph=True)[0] for x_prime in x.detach()]
        return torch.stack(jacobian_components, dim=1)

    def hessian(x: torch.Tensor) -> torch.Tensor:
        grad_k_x = grad(f(x, x.detach()), x, create_graph=True)
        hessian_components = [grad(grad_k_x[0][i], x, create_graph=True)[0] for i in range(len(x))]
        return torch.stack(hessian_components)

    def iterate_derivative(func, derivative_type: Literal['jacobian', 'hessian']) -> torch.Tensor:
        derivatives = []
        X_iterate = x_new.clone().requires_grad_(True)
        for i, x in enumerate(X_iterate):
            print(f"{derivative_type} derivative", i)
            derivatives.append(func(x))
        return torch.stack(derivatives, dim=0)

    partial_test_train = iterate_derivative(jacobian, 'jacobian')
    partial_test_test = iterate_derivative(hessian, 'hessian')
    return partial_test_train, partial_test_test


def analytic_SE_derivatives(kernel: Kernel, x_new: torch.Tensor, x: torch.Tensor):
    lengthscale = kernel.base_kernel.lengthscale.squeeze()
    all_diffs = (x.unsqueeze(0) - x_new.unsqueeze(1)) / lengthscale ** 2
    partial_test_train = (kernel(x_new, x).to_dense().unsqueeze(-1) * all_diffs).permute(0, 2, 1)

    kernel_evaluations_on_new_input = kernel(x_new, diag=True)
    diagonal_lengthscales = lengthscale**-2 * torch.eye(x_new.shape[1])
    partial_test_test = diagonal_lengthscales.unsqueeze(0) * kernel_evaluations_on_new_input.view(-1, 1, 1)

    return partial_test_train, partial_test_test
