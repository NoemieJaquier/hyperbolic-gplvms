import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import MatmulLinearOperator, LinearOperator
from functools import lru_cache
from linear_operator import to_dense


@lru_cache()
def get_train_train_inverse_covariance(X: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel_module: gpytorch.kernels.Kernel) -> LinearOperator:
    """[k(X, X) + noise]^{-1}"""
    gp_prior = MultivariateNormal(torch.zeros(X.shape[0]), kernel_module(X))
    train_train_covar = likelihood(gp_prior).lazy_covariance_matrix
    with gpytorch.settings.max_root_decomposition_size(3000):
        train_train_covar_inv_root = to_dense(train_train_covar.root_inv_decomposition().root)
    return MatmulLinearOperator(train_train_covar_inv_root, train_train_covar_inv_root.transpose(-1, -2))
