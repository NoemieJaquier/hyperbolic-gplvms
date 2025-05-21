import torch
import HyperbolicEmbeddings.hyperbolic_manifold.lorentz as lorentz


def analytic_diff_x_3D_hyperbolic_heat_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    let kernel(x,y) = tau * rho / sinh(rho) * exp(- rho**2 / kappa) be the hyperbolic heat kernel with
    rho = <x,y>_L the lorentzian inner product of the inputs x and y then this function computes
    the analytic first derivative d/dx kernel(x,y).

    Parameters
    -
    X: torch.shape([M, 4])   points on the lorentz manifold
    Y: torch.shape([M, 4])   points on the lorentz manifold
    tau: torch.shape([1])    scalar outputscale
    kappa: torch.shape([1])  scalar lengthscale

    Returns
    diff_x_k: torch.shape([M, 4])
    """
    G, u, rho, s = lorentz.common_ops(X, Y, operations='Gurs')
    g = analytic_g(u, kappa, rho, s)  # [M]
    scalar_factors = g * tau * torch.exp(-rho**2/kappa)  # [M]
    diff_x_k = scalar_factors.unsqueeze(-1) * torch.bmm(G, Y.unsqueeze(-1)).squeeze()
    return diff_x_k


def analytic_diff_x_3D_hyperbolic_heat_kernel_batched(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: torch.shape([M, 4])   points on the lorentz manifold
    Y: torch.shape([N, 4])   points on the lorentz manifold
    tau: torch.shape([1])    scalar outputscale
    kappa: torch.shape([1])  scalar lengthscale

    returns
    -
    diff_x_k: torch.shape([M, N, 4]) where diff_x_k[i, j] = d/dx kernel(X[i], Y[j])
    """
    N = Y.shape[0]
    # TODO: check if this can be vectorized directly into the analytic_diff_x_3D_hyperbolic_heat_kernel implementation
    return torch.stack([analytic_diff_x_3D_hyperbolic_heat_kernel(x.repeat(N, 1), Y, tau, kappa) for x in X])  # [M, N, 4]


def analytic_diff_xy_3D_hyperbolic_heat_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    let kernel(x,y) = tau * rho / sinh(rho) * exp(- rho**2 / kappa) be the hyperbolic heat kernel with
    rho = <x,y>_L the lorentzian inner product of the inputs x and y then this function computes
    the analytic second derivative d²/dydx kernel(x,y).

    Parameters
    -
    X: torch.shape([M, 4])   points on the lorentz manifold
    Y: torch.shape([M, 4])   points on the lorentz manifold
    tau: torch.shape([1])    scalar outputscale
    kappa: torch.shape([1])  scalar lengthscale

    Returns
    -
    diff_xy_k: torch.shape([M, 4, 4])
    """
    G, u, rho, s = lorentz.common_ops(X, Y, operations='Gurs')
    g = analytic_g(u, kappa, rho, s)  # [M]
    h = analytic_h(u, kappa, rho, s)  # [M]

    Gy, xG = torch.bmm(G, Y.unsqueeze(-1)), torch.bmm(G, X.unsqueeze(-1)).permute(0, 2, 1)
    GyxG = torch.bmm(Gy, xG)  # [M, 4, 4]
    scalar_factors = tau * torch.exp(-rho**2/kappa)
    diff_xy_k = (h[:, None, None] * GyxG + g[:, None, None]*G) * scalar_factors[:, None, None]

    return diff_xy_k


def analytic_diff_xx_3D_hyperbolic_heat_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    let kernel(x,y) = tau * rho / sinh(rho) * exp(- rho**2 / kappa) be the hyperbolic heat kernel with
    rho = <x,y>_L the lorentzian inner product of the inputs x and y then this function computes
    the analytic second derivative d²/dxdx kernel(x,y).

    Parameters
    -
    X: torch.shape([M, 4])   points on the lorentz manifold
    Y: torch.shape([M, 4])   points on the lorentz manifold
    tau: torch.shape([1])    scalar outputscale
    kappa: torch.shape([1])  scalar lengthscale

    Returns
    -
    diff_xx_k: torch.shape([M, 4, 4])
    """
    G, u, rho, s = lorentz.common_ops(X, Y, operations='Gurs')
    h = analytic_h(u, kappa, rho, s)  # [M]

    Gy = torch.bmm(G, Y.unsqueeze(-1))
    GyyG = torch.bmm(Gy, Gy.permute(0, 2, 1))  # [M, 4, 4]
    scalar_factors = tau * torch.exp(-rho**2/kappa)
    diff_xx_k = h[:, None, None] * GyyG * scalar_factors[:, None, None]

    return diff_xx_k


def analytic_diff_xx_3D_hyperbolic_heat_kernel_batched(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: torch.shape([M, 4])   points on the lorentz manifold
    Y: torch.shape([N, 4])   points on the lorentz manifold
    tau: torch.shape([1])    scalar outputscale
    kappa: torch.shape([1])  scalar lengthscale

    returns
    -
    diff_xx_k: torch.shape([M, N, 4, 4]) where diff_xx_k[i, j] = d/dxx kernel(X[i], Y[j])
    """
    N = Y.shape[0]
    # TODO: check if this can be vectorized directly into the analytic_diff_xx_3D_hyperbolic_heat_kernel implementation
    return torch.stack([analytic_diff_xx_3D_hyperbolic_heat_kernel(x.repeat(N, 1), Y, tau, kappa) for x in X])  # [M, N, 4, 4]


def analytic_third_derivative_3D_hyperbolic_heat_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    The second derivative d²/dydx kernel(x,y) is a 4x4 matrix. Compute the derivative for each matrix element with respect to x or y
    results in two 4x4x4 tensors. One for d/dy d²/dydx kernel(x,y) and one for d/dx d²/dydx kernel(x,y).

    Parameters
    -
    X: torch.shape([M, 4])   points on the lorentz manifold
    Y: torch.shape([M, 4])   points on the lorentz manifold
    tau: torch.shape([1])    scalar outputscale
    kappa: torch.shape([1])  scalar lengthscale

    Returns
    -
    diff_xyx_k: torch.shape([M, 4, 4, 4])
    diff_xyy_k: torch.shape([M, 4, 4, 4])
    """
    G, u, rho, s = lorentz.common_ops(X, Y, operations='Gurs')
    M = u.shape[0]
    h = analytic_h(u, kappa, rho, s)  # [M]
    z = analytic_z(u, kappa, rho, s)  # [M]

    def get_third_derivative(X: torch.Tensor, Y: torch.Tensor):
        diff_xyy_k = torch.zeros(M, 4, 4, 4)
        Gy, xG = torch.bmm(G, Y.unsqueeze(-1)), torch.bmm(G, X.unsqueeze(-1)).permute(0, 2, 1)  # [M, 4, 1], [M, 1, 4]
        GyxG = torch.bmm(Gy, xG)  # [M, 4, 4]
        zGyxG_hG = z[:, None, None] * GyxG + h[:, None, None]*G  # [M, 4, 4]
        hxG = h[:, None, None] * xG  # [M, 1, 4]
        scalar_factors = tau * torch.exp(-rho**2/kappa)  # [M]
        for j in range(4):
            e_j = torch.zeros(M, 4, 1)
            e_j[:, j] = 1
            delta = -1 if j == 0 else 1
            x_j = X[:, j, None, None]  # [M, 1, 1]
            diff_xyy_k[:, :, j, :] = (zGyxG_hG * x_j + torch.bmm(e_j, hxG)) * delta * scalar_factors[:, None, None]
        return diff_xyy_k

    diff_xyx_k = get_third_derivative(Y, X).permute(0, 2, 1, 3)
    diff_xyy_k = get_third_derivative(X, Y)
    return diff_xyx_k, diff_xyy_k


def analytic_g(u: torch.Tensor, kappa: torch.Tensor, rho: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    g = 2*rho**2 / (kappa*s**2) - (s + u*rho) / s**3
    return torch.where(rho < 1e-4, analytic_g_limit(kappa), g)


def analytic_g_diff(u: torch.Tensor, kappa: torch.Tensor, rho: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    g_diff = 3*u**2*rho / s**5 + 3*u/s**4 - rho/s**3 - 4*u*rho**2 / (kappa*s**4) - 4*rho / (kappa*s**3)
    return torch.where(rho < 1e-4, analytic_g_diff_limit(kappa), g_diff)


def analytic_g_diff2(u: torch.Tensor, kappa: torch.Tensor, rho: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """second derivative of g"""
    return -15*u**3*rho / s**7 - 15*u**2/s**6 + 9*u*rho/s**5 + 4/s**4 + 16*u**2*rho**2 / (kappa*s**6) + 20*u*rho/(kappa*s**5) + (4-4*rho**2)/(kappa*s**4)


def analytic_g_limit(kappa: torch.Tensor) -> torch.Tensor:
    return 2/kappa + 1/3.


def analytic_g_diff_limit(kappa: torch.Tensor) -> torch.Tensor:
    return 4/(3*kappa) + 4/15.


def analytic_g_diff2_limit(kappa: torch.Tensor) -> torch.Tensor:
    return 68/(45*kappa) + 12/35.


def analytic_h(u: torch.Tensor, kappa: torch.Tensor, rho: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """h = g_diff + g*2*rho / (kappa*s)"""
    h = ((2*u**2 + 1) * rho + 3*u*s) / s**5 - (6*u*rho**2 + + 2*rho*s) / (kappa*s**4) - 4*rho / (kappa*s**3) + 4*rho**3 / (kappa**2*s**3)
    return torch.where(rho < 1e-4, analytic_h_limit(kappa), h)


def analytic_h_diff(u: torch.Tensor, kappa: torch.Tensor, rho: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """first derivative of h."""
    g = analytic_g(u, kappa, rho, s)
    g_diff = analytic_g_diff(u, kappa, rho, s)
    g_diff2 = analytic_g_diff2(u, kappa, rho, s)
    h_diff = g_diff2 + g_diff*2*rho / (kappa*s) - 2*g / (kappa*s**2) - 2*g*u*rho / (kappa*s**3)
    return torch.where(rho < 1e-4, analytic_h_diff_limit(kappa), h_diff)


def analytic_h_limit(kappa: torch.Tensor) -> torch.Tensor:
    """h = g_diff + g*2*rho / (kappa*s)"""
    return 4/kappa**2 + 6/(3*kappa) + 4/15.


def analytic_h_diff_limit(kappa: torch.Tensor) -> torch.Tensor:
    return 4/kappa**2 + 34/(15*kappa) + 12/35.


def analytic_z(u: torch.Tensor, kappa: torch.Tensor, rho: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """z = h_diff + h*2*rho / (kappa*s)."""
    h = analytic_h(u, kappa, rho, s)
    h_diff = analytic_h_diff(u, kappa, rho, s)
    z = h_diff + h*2*rho / (kappa*s)
    return torch.where(rho < 1e-4, analytic_z_limit(kappa), z)


def analytic_z_limit(kappa: torch.Tensor) -> torch.Tensor:
    return 8 / kappa**3 + 8 / kappa**2 + 14/(5*kappa) + 12/35.
