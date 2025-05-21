import torch
import HyperbolicEmbeddings.hyperbolic_manifold.poincare as poincare


def get_phi(outer_product: torch.Tensor, samples_trunc_gaussian: torch.Tensor) -> torch.Tensor:
    """
    e^((1+2i)<x_n, b_l>)

    Parameters
    -
    X: [N, D]  points on the poincare model
    samples_circle: [L, D] random samples on the circle
    outer_product: [N, L] poincare outer product of X and samples_circle
    samples_trunc_gaussian: [L] random samples of a truncated Gaussian

    Returns
    -
    phi: [N, L]
    """
    phi_coefficient = (1. + 2j * samples_trunc_gaussian).unsqueeze(0)  # [1, L]
    return torch.exp(phi_coefficient * outer_product)  # [N, L]


def get_pms_factor(samples_trunc_gaussian: torch.Tensor) -> torch.Tensor:
    """
    s * tanh(pi*s)

    Parameters
    -
    samples_trunc_gaussian: [L] random samples of a truncated Gaussian

    Returns
    -
    pms_factor: [L]
    """
    return samples_trunc_gaussian * torch.tanh(torch.pi * samples_trunc_gaussian)


def get_normalization_factor(x0: torch.Tensor, samples_circle: torch.Tensor, pms_factor: torch.Tensor) -> torch.Tensor:
    """
    C = 1 / \sum_l s_l * tanh(pi*s_l) * |e^(<x_0, b_l)|^2)

    Parameters
    -
    x0: [D] point on the poincare model
    samples_circle: [L, 2] random samples on the circle
    pms_factor: [L]

    Returns
    -
    normalization_factor: [1]
    """
    # TODO: add some option to define the constant x0 outside this function or remove the option to pass a custom normalization point entirely
    x0 = torch.tensor([0.25, 0.25])
    outer_product_x0 = poincare.outer_product(x0.detach().unsqueeze(0), samples_circle).squeeze()  # [L]
    exponential_X = torch.exp(2*outer_product_x0)  # [L]
    return 1 / torch.sum(pms_factor * exponential_X)  # [1]


def get_random_samples_lorentz_kernel_2D(N_samples) -> None:
    """
    Parameters
    -
    N_samples:   number of samples

    Returns
    -
    samples_circle: [N_samples, 2] samples from the unit circle
    samples_std_gaussian: [N_samples] samples from a truncated standard gaussian distribution
    """
    angle_samples = 2. * torch.pi * torch.rand(N_samples)  # [N_samples]
    samples_x, samples_y = torch.cos(angle_samples), torch.sin(angle_samples)  # [N_samples], [N_samples]
    samples_circle = torch.stack([samples_x, samples_y], dim=1)  # [N_samples, 2]
    samples_std_gaussian = torch.randn(N_samples).squeeze()  # [N_samples]
    return samples_circle, samples_std_gaussian

