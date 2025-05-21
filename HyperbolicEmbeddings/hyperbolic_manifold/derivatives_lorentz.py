import torch
import HyperbolicEmbeddings.hyperbolic_manifold.lorentz as lorentz


def analytic_diff_x_lorentz_logmap(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    computes the derivative of the logarithmic map d/dx_i Log_{x_i}(y_i) for each pair (x_i, y_i) in X, Y.  

    Parameters
    -
    X: [N, D]   points on the lorentz manifold
    Y: [N, D]   points on the lorentz manifold

    Returns
    -
    diff_x_logmap: [N, D, D] where diff_x_logmap[i] = d/dx_i Log_{x_i}(y_i)
    """
    N, D = X.shape
    G, u, rho, s = lorentz.common_ops(X, Y, operations='Gurs')
    I = torch.eye(D).repeat(N, 1, 1)

    y_plus_ux = (Y + u.unsqueeze(-1) * X).unsqueeze(1)  # [M, 1, D]
    Gy = torch.bmm(G, Y.unsqueeze(2))  # [M, D, 1]
    diff_x_logmap = (- 1/s**2 - u*rho/s**3)[:, None, None] * torch.bmm(Gy, y_plus_ux)  # [M, D, D]
    diff_x_logmap += (rho/s)[:, None, None] * torch.bmm(Gy, X.unsqueeze(1))  # [M, D, D]
    diff_x_logmap += (u*rho/s)[:, None, None] * I  # [M, D, D]
    return diff_x_logmap.permute(0, 2, 1)


def analytic_diff_y_lorentz_logmap(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    computes the derivative of the logarithmic map d/dy_i Log_{x_i}(y_i) for each pair (x_i, y_i) in X, Y.  

    Parameters
    -
    X: [M, D]   points on the lorentz manifold
    Y: [M, D]   points on the lorentz manifold

    Returns
    -
    diff_y_logmap: [M, D, D] where diff_y_logmap[i] = d/dy_i Log_{x_i}(y_i)
    """
    M, D = X.shape
    G, u, rho, s = lorentz.common_ops(X, Y, operations='Gurs')
    I = torch.eye(D).repeat(M, 1, 1)

    y_plus_ux = (Y + u.unsqueeze(-1) * X).unsqueeze(1)  # [M, 1, D]
    Gx = torch.bmm(G, X.unsqueeze(2))  # [M, D, 1]
    diff_y_logmap = (-1/s**2 - u*rho/s**3)[:, None, None] * torch.bmm(Gx, y_plus_ux)  # [M, D, D]
    diff_y_logmap += (rho/s)[:, None, None] * (I + torch.bmm(Gx, X.unsqueeze(1)))  # [M, D, D]
    return diff_y_logmap.permute(0, 2, 1)
