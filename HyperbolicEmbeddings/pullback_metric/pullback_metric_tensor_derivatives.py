import torch


def analytic_jacobian_mean_cov(diff_x_k: torch.Tensor, diff_xy_k: torch.Tensor, k_train_inverse: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    mu_J = [d/dx* k(x*, X)] @ K_inv @ Y	
    Sigma_J = [d^2/dx*^2 k(x*, x*)] - [d/dx* k(x*, X)] @ K_inv @ [d/dx* k(X, x*)]
    with K_inv = [k(X,X) + noise]^{-1} 

    Parameters
    -
    diff_x_k: torch.shape([M, N, D_x])     derivative of the hyperbolic heat kernel w.r.t. x
    diff_xy_k: torch.shape([M, D_x, D_x])  second derivative of the hyperbolic heat kernel w.r.t. x and y
    k_train_inverse: torch.shape([N, N])   inverse of the training covariance matrix k(X, X) + noise
    Y: torch.shape([N, D_y])               training data

    Returns
    -
    jacobian_mean: torch.shape([M, D_y, D_x])  mean of the jacobian
    jacobian_cov: torch.shape([M, D_x, D_x])     covariance of the jacobian
    """
    M = diff_x_k.shape[0]
    covar_correction = torch.bmm(diff_x_k.permute(0, 2, 1), k_train_inverse.expand(M, -1, -1))  # [M, D_x, N]
    jacobian_mean = torch.bmm(covar_correction, Y.expand(M, -1, -1)).permute(0, 2, 1)  # [M, D_y, D_x]
    jacobian_cov = diff_xy_k - torch.bmm(covar_correction, diff_x_k)  # [M, D_x, D_x]
    return jacobian_mean, jacobian_cov


def analytic_diff_x_jacobian_mean_cov(diff_x_k: torch.Tensor, diff_xx_k: torch.Tensor, diff_xyx_k: torch.Tensor, diff_xyy_k: torch.Tensor, k_train_inverse: torch.Tensor, Y: torch.Tensor):
    """
    d/dx* mu_J = [d^2/dx*^2 k(x*, X)] @ K_inv @ Y	
    d/dx* Sigma_J = [d^3/dx*^3 k(x*, x*)] - [d^2/dx*^2 k(x*, X)] @ K_inv @ [d/dx* k(X, x*)] - [d/dx* k(x*, X)] @ K_inv @ [d^2/dx*^2 k(X, x*)]
    with K_inv = [k(X,X) + noise]^{-1} 

    Parameters
    -
    diff_x_k: torch.shape([M, N, D_x])        derivative of the hyperbolic heat kernel w.r.t. x
    diff_xx_k: torch.shape([M, N, D_x, D_x])    second derivative of the hyperbolic heat kernel w.r.t. x and x
    diff_xyx_k: torch.shape([M, D_x, D_x, D_x])   third derivative of the hyperbolic heat kernel w.r.t. x, y and x
    diff_xyy_k: torch.shape([M, D_x, D_x, D_x])   third derivative of the hyperbolic heat kernel w.r.t. x, y and y
    k_train_inverse: torch.shape([N, N])    inverse of the training covariance matrix k(X, X) + noise
    Y: torch.shape([N, D_y])                training data

    Returns
    -
    diff_x_jacobian_mean: torch.shape([M, D_y, D_x, D_x])  derivative of the mean of the jacobian w.r.t. x
    """
    M, D_x = diff_xx_k.shape[0], diff_xx_k.shape[2]
    covar_correction = batch_mm(diff_xx_k.permute(0, 2, 3, 1), k_train_inverse.expand(M, D_x, -1, -1))  # [M, D_x, D_x, N]
    diff_x_jacobian_mean = batch_mm(covar_correction, Y.expand(M, D_x, -1, -1)).permute(0, 3, 1, 2)  # [M, D_y, D_x, D_x]

    diff_x_jacobian_cov = torch.zeros(M, D_x, D_x, D_x)  # [M, D_x, D_x, D_x]
    dk_xx = diff_xyx_k + diff_xyy_k  # [M, D_x, D_x, D_x]
    for j in range(D_x):
        dkxX_kXX_kXx = torch.bmm(covar_correction[:, :, j, :], diff_x_k)  # [M, D_x, D_x]
        diff_x_jacobian_cov[:, :, :, j] = dk_xx[:, :, :, j] - dkxX_kXX_kXx - dkxX_kXX_kXx.permute(0, 2, 1)

    return diff_x_jacobian_mean, diff_x_jacobian_cov


def analytic_pullback_metric_tensor(jacobian_mean: torch.Tensor, jacobian_cov: torch.Tensor) -> torch.Tensor:
    """
    G_pullback=mu_J^T@mu_J + D_y*Sigma_J

    Parameters
    -
    jacobian_mean: torch.shape([M, D_y, D_x])  mean of the jacobian
    jacobian_cov: torch.shape([M, D_x, D_x])   covariance of the jacobian

    Returns
    -
    G_pullback: torch.shape([M, D_x, D_x])   pullback metric tensor 
    """
    D_y = jacobian_mean.shape[1]
    G_pullback = torch.bmm(jacobian_mean.permute(0, 2, 1), jacobian_mean) + D_y * jacobian_cov  # [M, 4, 4]
    return G_pullback


def analytic_diff_x_pullback_metric_tensor(jacobian_mean: torch.Tensor, diff_x_jacobian_mean: torch.Tensor, diff_x_jacobian_cov: torch.Tensor) -> torch.Tensor:
    """
    d/dx G_pullback = [d/dx mu_J^T]@mu_J + mu_J^T@[d/dx mu_J] + D_y*[d/dx Sigma_J]

    Parameters
    -
    jacobian_mean: torch.shape([M, D_y, D_x]) mean of the jacobian
    diff_x_jacobian_mean: torch.shape([M, D_y, D_x, D_x]) derivative of the mean of the jacobian w.r.t. x
    diff_x_jacobian_cov:  torch.shape([M, D_x, D_x, D_x])   derivative of the covariance of the jacobian w.r.t. x

    Returns
    -
    diff_x_G_pullback: torch.shape([M, D_x, D_x, D_x])    where for each point x=X[i] diff_x_G_pullback[i, :, :, j]= d/dx_ij G_pullback[i]
    """
    M, D_y, D_x = jacobian_mean.shape
    diff_x_G_pullback = torch.zeros(M, D_x, D_x, D_x)
    for j in range(D_x):
        mT_dm = torch.bmm(jacobian_mean.permute(0, 2, 1), diff_x_jacobian_mean[:, :, :, j])  # [M, D_x, D_x] = [M, D_x, D_y] @ [M, D_y, D_x]
        diff_x_G_pullback[:, :, :, j] = mT_dm + mT_dm.permute(0, 2, 1) + D_y * diff_x_jacobian_cov[:, :, :, j]
    return diff_x_G_pullback


def batch_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    matrix multiplication like torch.bmm but works with multiple batch dimensions, e.g.

    Parameters
    -
    A: torch.shape[N_1, ..., N_k, L, M]
    B: torch.shape[N_1, ..., N_k, M, P]

    Returns
    -
    result: torch.shape[N_1, ..., N_k, L, P] where result[i_1, ..., i_k] = A[i_1, ..., i_k] @ B[i_1, ..., i_k]
    """
    batch_shape = A.shape[:-2]
    result = torch.bmm(A.reshape(-1, *A.shape[-2:]), B.reshape(-1, *B.shape[-2:]))
    return result.view(*batch_shape, *result.shape[-2:])
