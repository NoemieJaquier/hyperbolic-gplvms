import torch


def analytic_diff_x_poincare_from_lorentz(X_lorentz: torch.Tensor) -> torch.Tensor:
    """
    let f(x) = p be the lorentz to poincare mapping

    Parameters
    X: [M, D+1] points on the Lorentz model

    Returns
    diff_x_p: [M, D, D+1] where diff_x_p[i] = d/dx f(X[i])
    """
    M, D = X_lorentz.shape[0], X_lorentz.shape[1]-1
    one_over_one_plus_x = 1 / (1 + X_lorentz[:, 0])  # [M]

    diff_x_p = torch.zeros(M, D, D+1, dtype=X_lorentz.dtype)
    i = torch.arange(D)
    diff_x_p[:, i, i+1] = one_over_one_plus_x[:, None]  # [M, D]
    diff_x_p[:, :, 0] = - X_lorentz[:, 1:] * one_over_one_plus_x[:, None]**2  # [M, D]
    return diff_x_p  # [M, D, D+1]


def analytic_diff_xx_poincare_from_lorentz(X_lorentz: torch.Tensor) -> torch.Tensor:
    """
    let f(x) = p be the lorentz to poincare mapping

    Parameters
    -
    X: [M, D+1] points on the Lorentz model

    Returns
    -
    diff_xx_p: [M, D, D+1, D+1] where diff_xx_p[i] = d^2/dx^2 f(X[i])
    """
    M, D = X_lorentz.shape[0], X_lorentz.shape[1]-1
    diff_xx_p = torch.zeros(M, D, D+1, D+1, dtype=X_lorentz.dtype)
    one_over_one_plus_x = 1 / (1 + X_lorentz[:, 0])  # [M]
    i = torch.arange(D)
    diff_xx_p[:, i, 1+i, 0] = - one_over_one_plus_x[:, None]**2  # [M, D]
    diff_xx_p[:, i, 0, 1+i] = - one_over_one_plus_x[:, None]**2  # [M, D]
    diff_xx_p[:, :, 0, 0] = 2*X_lorentz[:, 1:] * one_over_one_plus_x[:, None]**3  # [M, D]
    return diff_xx_p


def analytic_diff_x_poincare_outer_product(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, D] points in or on the unit circle
    Y: [M, D] points in or on the unit circle

    Returns
    -
    diff_x_outer_product: [N, M, D] where diff_x_p[i, j] = d/dx outer_product(X[i], Y[j])
    """
    denom_1 = (torch.sum(X**2, dim=1, keepdim=True) - 1).unsqueeze(1)  # [N, 1, D]
    X_sub_Y = X.unsqueeze(1) - Y.unsqueeze(0)  # [N, M, D]
    denom_2 = torch.sum(X_sub_Y**2, dim=2, keepdim=True)  # [N, M, 1]
    return X.unsqueeze(1)/denom_1 - X_sub_Y/denom_2   # [N, M, D]


def analytic_diff_xx_poincare_outer_product(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, D] points in or on the unit circle
    Y: [M, D] points in or on the unit circle

    Returns
    -
    diff_xx_outer_product: [N, M, D, D] where diff_xx_p[i, j] = d^2/dx^2 outer_product(X[i], Y[j])
    """
    N, M, D = X.shape[0], Y.shape[0], Y.shape[1]
    denom_1 = (torch.sum(X**2, dim=1, keepdim=True) - 1)[:, :, None, None]  # [N, 1, 1, 1]
    I = torch.eye(D).repeat(N, M, 1, 1)  # [N, M, D, D]
    XX_T = 2 * (X.unsqueeze(2) * X.unsqueeze(1)).unsqueeze(1)  # [N, 1, D, D]
    X_sub_Y = X.unsqueeze(1) - Y.unsqueeze(0)  # [N, M, D]
    X_sub_Y_X_sub_Y_T = 2 * X_sub_Y.unsqueeze(3) * X_sub_Y.unsqueeze(2)  # [N, M, D, D]
    denom_2 = torch.sum(X_sub_Y**2, dim=2, keepdim=True).unsqueeze(3)  # [N, M, 1, 1]
    return I/denom_1 - XX_T/denom_1**2 - I/denom_2 + X_sub_Y_X_sub_Y_T / denom_2**2  # [N, M, D, D]
