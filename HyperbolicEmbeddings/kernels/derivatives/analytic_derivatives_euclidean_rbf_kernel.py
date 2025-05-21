import torch


def analytic_euclidean_rbf_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor, diag=False) -> torch.Tensor:
    diffs = X-Y if diag else X.unsqueeze(1) - Y.unsqueeze(0)
    rho_squared = torch.sum(diffs**2, dim=-1)
    return tau * torch.exp(-rho_squared / kappa)  # [M]


def analytic_diff_x_euclidean_rbf_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, D]
    Y: [N, D]

    Returns
    diff_x_k: [N, D]
    """
    scalar_factors = analytic_euclidean_rbf_kernel(X, Y, tau, kappa, diag=True).unsqueeze(-1)  # [M, 1]
    return 2 * (Y-X) / kappa * scalar_factors  # [M, D_x]


def analytic_diff_x_euclidean_rbf_kernel_batched(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    N = Y.shape[0]
    # TODO: check if this can be vectorized directly into the analytic_diff_x_3D_hyperbolic_heat_kernel implementation
    return torch.stack([analytic_diff_x_euclidean_rbf_kernel(x.repeat(N, 1), Y, tau, kappa) for x in X])  # [M, N, D_x]


def analytic_diff_xy_euclidean_rbf_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    M, D_x = X.shape
    scalar_factors = analytic_euclidean_rbf_kernel(X, Y, tau, kappa, diag=True)[:, None, None]  # [M, 1, 1]
    I = torch.eye(D_x).repeat(M, 1, 1)  # [M, D_x, D_x]
    diff_yx = Y - X  # [M, D_x]
    outer_diff = torch.bmm(diff_yx[:, :, None], diff_yx[:, None, :])  # [M, D_x, D_x]
    return (2/kappa*I - 4/kappa**2*outer_diff) * scalar_factors
