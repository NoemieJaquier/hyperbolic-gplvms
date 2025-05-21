import torch
import HyperbolicEmbeddings.hyperbolic_manifold.poincare as poincare
from HyperbolicEmbeddings.hyperbolic_manifold.derivatives_poincare import analytic_diff_x_poincare_from_lorentz, analytic_diff_x_poincare_outer_product, analytic_diff_xx_poincare_from_lorentz, analytic_diff_xx_poincare_outer_product
from HyperbolicEmbeddings.kernels.utils_kernel_hyperbolic_2d import get_normalization_factor, get_phi, get_pms_factor


def analytic_diff_x_2D_lorentz_heat_kernel(
        X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor,
        samples_circle: torch.Tensor, samples_trunc_gaussian: torch.Tensor, x_normalization: torch.Tensor = None) -> torch.Tensor:
    """
    Parameters
    -
    X: [M, 3]   points on the lorentz manifold
    Y: [N, 3]   points on the lorentz manifold
    tau: [1]    scalar outputscale
    samples_circle: [L, 2] samples from the unit circle
    samples_trunc_gaussian: [L] samples from a truncated gaussian distribution. dependent on lengthscale
    x_normalization: [3] point on the lorentz model on which the normalization constant for the kernel is based.
                         If None, the x_normalization is just set to X[0]

    Returns
    diff_x_k: torch.shape([M, N, 3])
    """
    X_poincare, Y_poincare = poincare.from_lorentz(X), poincare.from_lorentz(Y)  # [M, 2]

    diff_x_phi = __helper_analytic_diff_x_phi(X_poincare, X, samples_circle, samples_trunc_gaussian)  # [M, L, 3]
    outer_product_Y = poincare.outer_product(Y_poincare, samples_circle)  # [M, L]

    phi_Y = get_phi(outer_product_Y, samples_trunc_gaussian)  # [N, L]

    pms_factor = get_pms_factor(samples_trunc_gaussian)  # [L]
    diff_x_kernel = torch.einsum('mli,nl->mni', diff_x_phi, pms_factor.unsqueeze(0) * phi_Y.conj())  # [M, N, 3]

    x_normalization_poincare = X_poincare[0].detach() if x_normalization is None else poincare.from_lorentz(x_normalization.unsqueeze(0)).squeeze()
    normalization_factor = get_normalization_factor(x_normalization_poincare, samples_circle, pms_factor)  # [1]
    return tau * normalization_factor * diff_x_kernel.real


def analytic_diff_xx_2D_lorentz_heat_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor,
                                            samples_circle: torch.Tensor, samples_trunc_gaussian: torch.Tensor, x_normalization: torch.Tensor = None) -> torch.Tensor:
    """
    Parameters
    -
    X: [M, 3]   points on the lorentz manifold
    Y: [N, 3]   points on the lorentz manifold
    tau: [1]    scalar outputscale
    samples_circle: [L, 2] samples from the unit circle
    samples_trunc_gaussian: [L] samples from a truncated gaussian distribution. dependent on lengthscale
    x_normalization: [3] point on the lorentz model on which the normalization constant for the kernel is based.
                         If None, the x_normalization is just set to X[0]

    Returns
    -
    diff_xx_k: [M, N, 3, 3] where diff_xx_k[i, j] = d^2/dx^2 kernel(X[i], Y[j])
    """
    X_poincare, Y_poincare = poincare.from_lorentz(X), poincare.from_lorentz(Y)  # [M, 2]

    outer_product_X = poincare.outer_product(X_poincare, samples_circle)  # [M, L]
    phi_X = get_phi(outer_product_X, samples_trunc_gaussian)  # [M, L]
    diff_xp_outer_product = analytic_diff_x_poincare_outer_product(X_poincare, samples_circle)  # [M, L, 2]
    diff_xxp_outer_product = analytic_diff_xx_poincare_outer_product(X_poincare, samples_circle)  # [M, L, 2, 2]
    diff_x_xp = analytic_diff_x_poincare_from_lorentz(X)  # [M, 2, 3]
    diff_xx_xp = analytic_diff_xx_poincare_from_lorentz(X)  # [M, 2, 3, 3]
    diff_xx_phi = analytic_diff_xx_phi(phi_X, diff_xp_outer_product, diff_xxp_outer_product,
                                       diff_x_xp, diff_xx_xp, samples_trunc_gaussian)  # [M, L, 3, 3]

    outer_product_Y = poincare.outer_product(Y_poincare, samples_circle)  # [N, L]
    phi_Y = get_phi(outer_product_Y, samples_trunc_gaussian)  # [N, L]

    pms_factor = get_pms_factor(samples_trunc_gaussian)  # [1, L]
    diff_xx_kernel = torch.einsum('mlij,nl->mnij', diff_xx_phi, pms_factor.unsqueeze(0) * phi_Y.conj())  # [M, N, 3, 3]

    x_normalization_poincare = X_poincare[0].detach() if x_normalization is None else poincare.from_lorentz(x_normalization.unsqueeze(0)).squeeze()
    normalization_factor = get_normalization_factor(x_normalization_poincare, samples_circle, pms_factor)  # [1]
    return tau * normalization_factor * diff_xx_kernel.real


def analytic_diff_xy_2D_lorentz_heat_kernel(
        X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor,
        samples_circle: torch.Tensor, samples_trunc_gaussian: torch.Tensor, x_normalization: torch.Tensor = None) -> torch.Tensor:
    """
    Parameters
    -
    X: [M, 3]   points on the lorentz manifold
    Y: [M, 3]   points on the lorentz manifold
    tau: [1]    scalar outputscale
    samples_circle: [L, 2] samples from the unit circle
    samples_trunc_gaussian: [L] samples from a truncated gaussian distribution. dependent on lengthscale
    x_normalization: [3] point on the lorentz model on which the normalization constant for the kernel is based.
                         If None, the x_normalization is just set to X[0]

    Returns
    -
    diff_xy_k: [M, 3, 3] where diff_xy_k[i] = d^2/dxdy kernel(X[i], Y[i])
    """
    X_poincare, Y_poincare = poincare.from_lorentz(X), poincare.from_lorentz(Y)  # [M, 2]

    diff_x_phi = __helper_analytic_diff_x_phi(X_poincare, X, samples_circle, samples_trunc_gaussian)  # [M, L, 3]
    diff_y_phi = __helper_analytic_diff_x_phi(Y_poincare, Y, samples_circle, samples_trunc_gaussian)  # [M, L, 3]

    pms_factor = get_pms_factor(samples_trunc_gaussian)  # [L]
    diff_xy_kernel = torch.sum(
        pms_factor[None, :, None, None] *  # [1, L, 1, 1]
        diff_x_phi.unsqueeze(3) *  # [M, L, 3, 1]
        diff_y_phi.conj().unsqueeze(2),  # [M, L, 1, 3]
        dim=1)  # [M, 3, 3]

    x_normalization_poincare = X_poincare[0].detach() if x_normalization is None else poincare.from_lorentz(x_normalization.unsqueeze(0)).squeeze()
    normalization_factor = get_normalization_factor(x_normalization_poincare, samples_circle, pms_factor)  # [1]
    return tau * normalization_factor * diff_xy_kernel.real  # [M, 3, 3]


def analytic_third_derivative_2D_lorentz_heat_kernel(X: torch.Tensor, Y: torch.Tensor, tau: torch.Tensor,
                                                     samples_circle: torch.Tensor, samples_trunc_gaussian: torch.Tensor, x_normalization: torch.Tensor = None) -> torch.Tensor:
    """
    Parameters
    -
    X: [M, 3]   points on the lorentz manifold
    Y: [M, 3]   points on the lorentz manifold
    tau: [1]    scalar outputscale
    samples_circle: [L, 2] samples from the unit circle
    samples_trunc_gaussian: [L] samples from a truncated gaussian distribution. dependent on lengthscale
    x_normalization: [3] point on the lorentz model on which the normalization constant for the kernel is based.
                         If None, the x_normalization is just set to X[0]

    Returns
    -
    diff_xyx_k: [M, 3, 3, 3] where diff_xyx_k[i] = d^3/dx^2dy kernel(X[i], Y[i])
    diff_xyy_k: [M, 3, 3, 3] where diff_xyy_k[i] = d^3/dxdy^2 kernel(X[i], Y[i])
    """
    X_poincare, Y_poincare = poincare.from_lorentz(X), poincare.from_lorentz(Y)  # [M, 2]

    outer_product_X = poincare.outer_product(X_poincare, samples_circle)  # [M, L]
    phi_X = get_phi(outer_product_X, samples_trunc_gaussian)  # [M, L]
    diff_xp_outer_product = analytic_diff_x_poincare_outer_product(X_poincare, samples_circle)  # [M, L, 2]
    diff_xxp_outer_product = analytic_diff_xx_poincare_outer_product(X_poincare, samples_circle)  # [M, L, 2, 2]
    diff_x_xp = analytic_diff_x_poincare_from_lorentz(X)  # [M, 2, 3]
    diff_xx_xp = analytic_diff_xx_poincare_from_lorentz(X)  # [M, 2, 3, 3]
    diff_x_phi = analytic_diff_x_phi(phi_X, diff_xp_outer_product, diff_x_xp, samples_trunc_gaussian)  # [M, L, 3]
    diff_xx_phi = analytic_diff_xx_phi(phi_X, diff_xp_outer_product, diff_xxp_outer_product,
                                       diff_x_xp, diff_xx_xp, samples_trunc_gaussian)  # [M, L, 3, 3]

    outer_product_Y = poincare.outer_product(Y_poincare, samples_circle)  # [M, L]
    phi_Y = get_phi(outer_product_Y, samples_trunc_gaussian)  # [M, L]
    diff_yp_outer_product = analytic_diff_x_poincare_outer_product(Y_poincare, samples_circle)  # [M, L, 2]
    diff_yyp_outer_product = analytic_diff_xx_poincare_outer_product(Y_poincare, samples_circle)  # [M, L, 2, 2]
    diff_y_yp = analytic_diff_x_poincare_from_lorentz(Y)  # [M, 2, 3]
    diff_yy_yp = analytic_diff_xx_poincare_from_lorentz(Y)  # [M, 2, 3, 3]
    diff_y_phi = analytic_diff_x_phi(phi_Y, diff_yp_outer_product, diff_y_yp, samples_trunc_gaussian)  # [M, L, 3]
    diff_yy_phi = analytic_diff_xx_phi(phi_Y, diff_yp_outer_product, diff_yyp_outer_product,
                                       diff_y_yp, diff_yy_yp, samples_trunc_gaussian)  # [M, L, 3, 3]

    pms_factor = get_pms_factor(samples_trunc_gaussian)  # [L]
    x_normalization_poincare = X_poincare[0].detach() if x_normalization is None else poincare.from_lorentz(x_normalization.unsqueeze(0)).squeeze()
    normalization_factor = get_normalization_factor(x_normalization_poincare, samples_circle, pms_factor)  # [1]

    diff_xyx_kernel = torch.sum((pms_factor[None, :, None, None] * diff_xx_phi).unsqueeze(3) *
                                diff_y_phi.conj().unsqueeze(2).unsqueeze(4), dim=1)  # [M, 3, 3, 3]
    diff_xyy_kernel = torch.sum((pms_factor[None, :, None] * diff_x_phi).unsqueeze(3).unsqueeze(4) *
                                diff_yy_phi.conj().unsqueeze(2), dim=1)  # [M, 3, 3, 3]
    return tau*normalization_factor*diff_xyx_kernel.real, tau*normalization_factor*diff_xyy_kernel.real


def analytic_diff_x_phi(phi: torch.Tensor, diff_xp_outer_product: torch.Tensor, diff_x_xp: torch.Tensor, samples_trunc_gaussian: torch.Tensor) -> torch.Tensor:
    """
    e^((1+2i)<x_n, b_l>)

    Parameters
    -
    phi: [N, L]   e^((1+2i)<x_n, b_l>)
    diff_xp_outer_product: [N, L, D] derivative of the poincare outer product of X and samples_circle
    diff_x_xp: [N, D, D+1] derivative of the poincare from lorentz
    samples_trunc_gaussian: [L] random samples of a truncated Gaussian

    Returns
    -
    diff_x_phi: [N, L, D+1]
    """
    phi_coefficient = (1. + 2j * samples_trunc_gaussian).unsqueeze(0).unsqueeze(2)  # [1, L, 1]
    diff_x_outer_product = torch.bmm(diff_xp_outer_product, diff_x_xp)  # [N, L, D+1]
    return phi_coefficient * phi.unsqueeze(2) * diff_x_outer_product  # [N, L, D+1]


def analytic_diff_xx_phi(phi: torch.Tensor, diff_xp_outer_product: torch.Tensor, diff_xxp_outer_product: torch.Tensor,
                         diff_x_xp: torch.Tensor, diff_xx_xp: torch.Tensor, samples_trunc_gaussian: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    phi: [N, L]   e^((1+2i)<x_n, b_l>)
    diff_xp_outer_product: [N, L, D] derivative of the poincare outer product of X and samples_circle
    diff_xxp_outer_product: [N, L, D, D] second derivative of the poincare outer product of X and samples_circle
    diff_x_xp: [N, D, D+1] derivative of the poincare from lorentz mapping
    diff_xx_xp: [N, D, D+1, D+1] second derivative of the poincare from lorentz mapping
    samples_trunc_gaussian: [L] random samples of a truncated Gaussian

    Returns
    -
    diff_xx_phi: [N, L, D+1, D+1]
    """
    N, L, D = diff_xp_outer_product.shape
    phi_coefficient = (1. + 2j * samples_trunc_gaussian)[None, :, None, None]  # [1, L, 1, 1]
    diff_xp_outer_product_squared = diff_xp_outer_product.unsqueeze(2) * diff_xp_outer_product.unsqueeze(3)  # [N, L, D, D]
    diff_x_xp_repeated = diff_x_xp.unsqueeze(1).repeat(1, L, 1, 1)  # [N, L, D, D+1]
    part_1 = __batch_mm(__batch_mm(diff_x_xp_repeated.permute(0, 1, 3, 2), diff_xp_outer_product_squared), diff_x_xp_repeated)  # [N, L, D+1, D+1]
    part_2 = torch.bmm(diff_xp_outer_product, diff_xx_xp.reshape(N, D, -1)).reshape(N, L, D+1, D+1)  # [N, L, D+1, D+1]
    part_3 = __batch_mm(__batch_mm(diff_x_xp_repeated.permute(0, 1, 3, 2), diff_xxp_outer_product), diff_x_xp_repeated)  # [N, L, D+1, D+1]
    return phi_coefficient * phi[:, :, None, None] * (phi_coefficient * part_1 + part_2 + part_3)  # [N, L, D+1, D+1]


def __helper_analytic_diff_x_phi(X_poincare: torch.Tensor, X_lorentz: torch.Tensor, samples_circle: torch.Tensor, samples_trunc_gaussian: torch.Tensor) -> torch.Tensor:
    outer_product_X = poincare.outer_product(X_poincare, samples_circle)  # [M, L]
    phi_X = get_phi(outer_product_X, samples_trunc_gaussian)  # [M, L]
    diff_xp_outer_product = analytic_diff_x_poincare_outer_product(X_poincare, samples_circle)  # [M, L, 2]
    diff_x_xp = analytic_diff_x_poincare_from_lorentz(X_lorentz)  # [M, 2, 3]
    diff_x_phi = analytic_diff_x_phi(phi_X, diff_xp_outer_product, diff_x_xp, samples_trunc_gaussian)  # [M, L, 3]
    return diff_x_phi


def __batch_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    batch_shape = A.shape[:-2]
    result = torch.bmm(A.reshape(-1, *A.shape[-2:]), B.reshape(-1, *B.shape[-2:]))
    return result.view(*batch_shape, *result.shape[-2:])
