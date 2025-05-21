import torch
import gpytorch

from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import lorentz_distance_torch, lorentz_to_poincare
from HyperbolicEmbeddings.kernels.utils_kernel_hyperbolic_2d import get_normalization_factor, get_phi, get_pms_factor, get_random_samples_lorentz_kernel_2D
import HyperbolicEmbeddings.hyperbolic_manifold.poincare as poincare
import HyperbolicEmbeddings.hyperbolic_manifold.lorentz as lorentz

# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
# else:
#     device = 'cpu'
device = 'cpu'


class LorentzGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the hyperbolic manifold.

    This class is a faster version of the HyperbolicRiemannianGaussianKernel below. Both kernels are exactly identical if resample=True in HyperbolicRiemannianGaussianKernel.  

    Attributes
    ----------
    self.dim, dimension of the hyperbolic H^n on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral for the 2D heat kernel

    Methods
    -------
    forward(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)
    forward_dim1(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)
    forward_dim2(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)
    forward_dim3(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim: int, batch_shape: torch.Size = torch.Size([]), nb_points_integral=1000, **kwargs) -> None:
        """
        Parameters
        -
        dim: dimension of the hyperbolic manifold. Currently, only dim=1, 2, or 3 are supported.
        nb_points_integral: number of samples used for the Monte Carlo approximation in the 2D case, i.e., only relevant for dim=2
        """
        self.has_lengthscale = True
        super(LorentzGaussianKernel, self).__init__(ard_num_dims=None, batch_shape=batch_shape, **kwargs)
        self.dim = dim
        self.nb_points_integral = nb_points_integral
        self.samples_circle, self.samples_std_gaussian = get_random_samples_lorentz_kernel_2D(self.nb_points_integral)

    def lorentz_kernel_1D(self, X: torch.Tensor, Y: torch.Tensor, kappa: float, diag=False) -> torch.Tensor:
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


    def lorentz_kernel_2D(self, X: torch.Tensor, Y: torch.Tensor, samples_circle: torch.Tensor, samples_trunc_gaussian: torch.Tensor) -> torch.Tensor:
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

        outer_product_X = poincare.outer_product(X_poincare, samples_circle)  # [N, L]
        outer_product_Y = poincare.outer_product(Y_poincare, samples_circle)  # [M, L]
        phi_X = get_phi(outer_product_X, samples_trunc_gaussian)  # [N, L]
        phi_Y = get_phi(outer_product_Y, samples_trunc_gaussian)  # [M, L]
        pms_factor = get_pms_factor(samples_trunc_gaussian).unsqueeze(0)  # [1, L]
        normalization_factor = get_normalization_factor(X_poincare[0], samples_circle, pms_factor)  # [1]
        outer_product = (pms_factor * phi_X) @ phi_Y.conj().T   # [N, M]
        return normalization_factor * outer_product.real


    def lorentz_kernel_3D(self, X: torch.Tensor, Y: torch.Tensor, kappa: float, diag=False) -> torch.Tensor:
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
            kernel = self.lorentz_kernel_1D(X, Y, 2*self.lengthscale**2, diag)
        elif self.dim == 2:
            samples_trunc_gaussian = torch.abs(self.samples_std_gaussian / self.lengthscale.squeeze())
            kernel = self.lorentz_kernel_2D(X, Y, self.samples_circle, samples_trunc_gaussian)
            if diag:
                kernel = kernel[torch.eye(X.shape[0], dtype=torch.bool)]  # TODO: properly add diag flag for lorentz_heat_kernel_2D function
        elif self.dim == 3:
            kernel = self.lorentz_kernel_3D(X, Y, 2*self.lengthscale**2, diag)
        else:
            raise NotImplementedError

        if output_shape is not None and self.batch_shape == torch.Size():
            kernel = kernel.repeat(output_shape, 1, 1)
        return kernel.squeeze() if diag else kernel
    

class HyperbolicRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the hyperbolic
    manifold.

    Attributes
    ----------
    self.dim, dimension of the hyperbolic H^n on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral for the heat kernel

    Methods
    -------
    forward(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)
    forward_dim1(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)
    forward_dim2(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)
    forward_dim3(point1_in_hyperbolic, point2_in_hyperbolic, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nb_points_integral=1000, resample=True, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the hyperbolic H^n on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nb_points_integral: number of points used to compute the integral for the heat kernel
        :param resample: use new samples at each call of the function
                         We used "True" for training the models of our ICML paper.
                         "False" makes the kernel identical to LorentzGaussianKernel above.
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(HyperbolicRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Dimension of hyperbolic manifold (data dimension = self.dim + 1)
        self.dim = dim

        # Number of points for the integral computation
        self.nb_points_integral = nb_points_integral

        # New samples at each forward call? 
        self.resample = resample

        if not self.resample and self.dim == 2:
            # Draw samples
            self.samples_circle, self.samples_std_gaussian = get_random_samples_lorentz_kernel_2D(self.nb_points_integral)

    def forward_dim1(self, x1, x2):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a hyperbolic manifold H^1

        Parameters
        ----------
        :param x1: input points on the hyperbolic manifold
        :param x2: input points on the hyperbolic manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole kernel matrix, or just the diagonal? If True, we must have `x1 == x2`

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute hyperbolic distance
        distance = lorentz_distance_torch(x1, x2)

        # Kernel
        kernel = torch.exp(- torch.pow(distance, 2) / (2 * self.lengthscale ** 2))

        return kernel

    def forward_dim2(self, x1, x2):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a hyperbolic manifold H^2 following:

        p(z, w, t) ~ C \sum_{m} \rho_m \tanh(\pi \rho_m) \frac{1}{N}
                        \sum_{n} e^{(i 2 \rho_m + 1) \langle z, b_n \rangle} \overline{e^{(i 2 \rho_m + 1) \langle w,
        with b_n ~ U(\mathbb{S}^1) and \rho_m = |\tilde{\rho_m}|, \tilde{\rho_m} ~ N(0, 1/2t)
        and t = lengthscale^2 / 2

        In this case, the inner sum is approximated with a single sample b_n.
        Note that more samples of \rho_m (~1000) are required to achieve a similar approximation as for the
        forward_double_sum function.

        Parameters
        ----------
        :param x1: input points on the hyperbolic manifold
        :param x2: input points on the hyperbolic manifold

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """

        # Lorentz to Poincare
        x1_p = lorentz_to_poincare(x1)
        x2_p = lorentz_to_poincare(x2)

        if self.resample:
            # Draw samples
            self.samples_circle, self.samples_std_gaussian = get_random_samples_lorentz_kernel_2D(self.nb_points_integral)

        # Sample b uniformly on S1
        bns = self.samples_circle

        # Sample p - Getting samples from truncated normal distribution
        samples_trunc_gaussian = torch.abs(self.samples_std_gaussian / self.lengthscale).squeeze().to(device)

        # Compute phi_l coefficient and factor of the sum over pms
        pms_factor = get_pms_factor(samples_trunc_gaussian).unsqueeze(-2).unsqueeze(-2)

        outer_product_x1 = poincare.outer_product(x1_p, bns)
        outer_product_x2 = poincare.outer_product(x2_p, bns)
        phi_x1 = get_phi(outer_product_x1, samples_trunc_gaussian).unsqueeze(-2)
        phi_x2 = get_phi(outer_product_x2, samples_trunc_gaussian).unsqueeze(-3)
        phi_x1_extended = torch.cat(phi_x2.shape[-2] * [phi_x1], dim=-2)
        phi_x2_extended = torch.cat(phi_x1.shape[-3] * [phi_x2], dim=-3)
        phi_l = phi_x1_extended * torch.conj(phi_x2_extended)

        # Compute heat kernel ~ outer sum over p
        outer_sum = torch.sum(pms_factor * phi_l, -1) # / self.nb_points_integral

        # With normalization factor
        phi_l_normal = phi_x1_extended[..., 0, 0, :] * torch.conj(phi_x2_extended[..., 0, 0, :])

        if self.resample:
            normalization_factor = 1. / (torch.sum(pms_factor[..., 0, 0, :] * phi_l_normal, -1))
            # normalization_factor = 1. / (torch.sum(pms_factor[..., 0, 0, :] * phi_l_normal, -1) / self.nb_points_integral)
        else:
            normalization_factor = get_normalization_factor(x1[0], self.samples_circle, pms_factor)
        normalization_factor = normalization_factor.unsqueeze(-1).unsqueeze(-1)

        kernel = normalization_factor * outer_sum

        # Kernel
        return kernel.real

    def forward_dim3(self, x1, x2):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a hyperbolic manifold H^1

        Parameters
        ----------
        :param x1: input points on the hyperbolic manifold
        :param x2: input points on the hyperbolic manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole kernel matrix, or just the diagonal? If True, we must have `x1 == x2`

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute hyperbolic distance
        distance = lorentz_distance_torch(x1, x2)

        # Kernel (simplifying non-distance-related terms)
        kernel = torch.exp(- torch.pow(distance, 2) / (2 * self.lengthscale ** 2))
        # kernel = kernel * distance / torch.sinh(distance)  # Adding 1e-8 avoids numerical issues around d=0
        kernel = kernel * (distance+1e-8) / torch.sinh(distance+1e-8)  # Adding 1e-8 avoids numerical issues around d=0
        return kernel

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a hyperbolic manifold.

        Parameters
        ----------
        :param x1: input points on the hyperbolic manifold
        :param x2: input points on the hyperbolic manifold

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """

        # If dimension is 1, 2, or 3, compute the kernel directly
        if self.dim == 1:
            kernel = self.forward_dim1(x1, x2)
        elif self.dim == 2:
            # start = time.time()
            kernel = self.forward_dim2(x1, x2)
            # print(time.time()-start)
        elif self.dim == 3:
            kernel = self.forward_dim3(x1, x2)
        else:
            raise NotImplementedError

        # Handle diagonal case
        # TODO We should do this more efficiently
        if diag:
            kernel = torch.diagonal(kernel, 0, -2, -1)
        # Kernel
        return kernel

