import torch
import gpytorch
import HyperbolicEmbeddings.hyperbolic_manifold.lorentz as lorentz
import HyperbolicEmbeddings.hyperbolic_manifold.poincare as poincare
from HyperbolicEmbeddings.kernels.utils_kernel_hyperbolic_2d import poincare_heat_kernel_2D


def lorentz_heat_kernel_1D(X: torch.Tensor, Y: torch.Tensor, kappa: float, diag=False) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, 2] points on the Lorentz Model
    Y: [M, 2] points on the Lorentz Model
    tau:   outputscale of the kernel
    kappa: lengthscale of the kernel
    diag:     whether to return the diagonal or the whole kernel matrix

    Returns
    -------
    kernel: [N, M] if not diag else [N]
    """
    distance_squared = lorentz.distance_squared(X, Y, diag=diag)
    return torch.exp(- distance_squared / kappa)


def get_random_samples_lorentz_heat_kernel_2D(N_samples: int) -> torch.Tensor:
    """
    Parameters
    -
    N_samples:   number of samples
    lengthscale: lengthscale of the kernel

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


def lorentz_heat_kernel_2D(X: torch.Tensor, Y: torch.Tensor, samples_circle: torch.Tensor, samples_trunc_gaussian: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, 3]   points on the lorentz manifold
    Y: [M, 3]   points on the lorentz manifold
    tau: [1]    scalar outputscale
    samples_circle: [N_samples, 2] samples from the unit circle 
    samples_trunc_gaussian: [N_samples] samples from a truncated gaussian distribution. dependent on lengthscale

    Returns
    kernel: [N, M]
    """
    X_poincare = poincare.from_lorentz(X)  # [N, 2]
    Y_poincare = poincare.from_lorentz(Y)  # [M, 2]
    return poincare_heat_kernel_2D(X_poincare, Y_poincare, samples_circle, samples_trunc_gaussian)


def lorentz_heat_kernel_3D(X: torch.Tensor, Y: torch.Tensor, kappa: float, diag=False) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, 4] points on the Lorentz Model
    Y: [M, 4] points on the Lorentz Model
    diag:     whether to return the diagonal or the whole kernel matrix

    Returns
    -------
    kernel: [N, M] if not diag else [N]
    """
    distance = lorentz.distance(X, Y, diag)  # [N, M] or [N]
    base_kernel = torch.exp(-distance**2 / kappa)  # [N, M] or [N]
    scalar_factors = distance / torch.sinh(distance)  # [N, M] or [N]
    scalar_factors = torch.where(torch.isnan(scalar_factors), 1., scalar_factors)  # [N, M] or [N]
    return scalar_factors * base_kernel


class LorentzHeatKernel(gpytorch.kernels.Kernel):
    def __init__(self, dim: int, batch_shape: torch.Size = torch.Size([]), nb_points_integral=1000) -> None:
        """
        Parameters
        -
        dim: dimension of the hyperbolic manifold. Currently, only dim=1, 2, or 3 are supported.
        nb_points_integral: number of samples used for the Monte Carlo approximation in the 2D case, i.e., only relevant for dim=2
        """
        self.has_lengthscale = True
        super(LorentzHeatKernel, self).__init__(ard_num_dims=None, batch_shape=batch_shape)
        self.dim = dim
        self.nb_points_integral = nb_points_integral
        self.draw_new_2D_samples()

    def draw_new_2D_samples(self) -> None:
        """
        randomly draws new samples used for the 2D heat kernel computation
        """
        samples_circle, samples_std_gaussian = get_random_samples_lorentz_heat_kernel_2D(self.nb_points_integral)
        self.samples_circle = samples_circle
        self.samples_std_gaussian = samples_std_gaussian

    def forward(self, X, Y, diag=False, **_) -> torch.Tensor:
        """
        Parameters
        -
        X: [N, D] points on the Lorentz Model
        Y: [M, D] points on the Lorentz Model
        diag:     whether to return the diagonal or the whole kernel matrix

        Returns
        -
        kernel: [N, M] if not diag else [N]
        """
        output_shape = None
        if len(X.shape) == 3:
            output_shape = X.shape[0]
            X = X[0]
            Y = Y[0]

        if self.dim == 1:
            kernel = lorentz_heat_kernel_1D(X, Y, 2*self.lengthscale**2, diag)
        elif self.dim == 2:
            samples_trunc_gaussian = torch.abs(self.samples_std_gaussian / self.lengthscale.squeeze())
            kernel = lorentz_heat_kernel_2D(X, Y, self.samples_circle, samples_trunc_gaussian)
            if diag:
                kernel = kernel[torch.eye(X.shape[0], dtype=torch.bool)]  # TODO: properly add diag flag for lorentz_heat_kernel_2D function
        elif self.dim == 3:
            kernel = lorentz_heat_kernel_3D(X, Y, 2*self.lengthscale**2, diag)
        else:
            raise NotImplementedError

        if output_shape is not None and self.batch_shape == torch.Size():
            kernel = kernel.repeat(output_shape, 1, 1)
        return kernel.squeeze() if diag else kernel
