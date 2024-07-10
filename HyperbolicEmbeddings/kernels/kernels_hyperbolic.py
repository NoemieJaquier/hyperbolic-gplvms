import torch
import gpytorch

from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import lorentz_distance_torch, lorentz_to_poincare

# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
# else:
#     device = 'cpu'
device = 'cpu'


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
    def __init__(self, dim, nb_points_integral=1000, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the hyperbolic H^n on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nb_points_integral: number of points used to compute the integral for the heat kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(HyperbolicRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Dimension of hyperbolic manifold (data dimension = self.dim + 1)
        self.dim = dim

        # Number of points for the integral computation
        self.nb_points_integral = nb_points_integral

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

    def inner_product(self, x1, x2, diag=False):
        """
        Computes the inner product between points in the hyperbolic manifold (Poincaré ball representation) as
        <z, b> = \frac{1}{2}\log\frac{1-|z|^2}{|z-b|^2}

        Parameters
        ----------
        :param x1: input points on the hyperbolic manifold
        :param x2: input points on the hyperbolic manifold

        Optional parameters
        -------------------
        :param diag: Should we return the matrix, or just the diagonal? If True, we must have `x1 == x2`

        Returns
        -------
        :return: inner product matrix between x1 and x2
        """
        if diag is False:
            # Expand dimensions to compute all vector-vector distances
            x1 = x1.unsqueeze(-2)
            x2 = x2.unsqueeze(-3)

            # Repeat x and y data along -2 and -3 dimensions to have b1 x ... x ndata_x x ndata_y x dim arrays
            x1 = torch.cat(x2.shape[-2] * [x1], dim=-2)
            x2 = torch.cat(x1.shape[-3] * [x2], dim=-3)

        # Difference between x1 and x2
        diff_x = x1 - x2

        # Inner product
        inner_product = 0.5 * torch.log((1. - torch.norm(x1, dim=-1)**2) / torch.norm(diff_x, dim=-1)**2)

        return inner_product

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

        Optional parameters
        -------------------
        :param diag: Should we return the whole kernel matrix, or just the diagonal? If True, we must have `x1 == x2`

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """

        # Lorentz to Poincare
        x1_p = lorentz_to_poincare(x1)
        x2_p = lorentz_to_poincare(x2)

        # Sample b uniformly on S1
        angle_samples = 2. * torch.pi * torch.rand(self.nb_points_integral)
        bns = torch.cat((torch.cos(angle_samples)[:, None], torch.sin(angle_samples)[:, None]), -1).to(device)

        # Sample p - Getting samples from truncated normal distribution
        pms = torch.abs(torch.randn(self.nb_points_integral, requires_grad=False, device=device)
                        * (1. / self.lengthscale)).squeeze().to(device)

        # Compute phi_l coefficient and factor of the sum over pms
        phi_l_coefficient = (2j * pms + 1.).unsqueeze(-2).unsqueeze(-2)
        pms_factor = (pms * torch.tanh(torch.pi * pms)).unsqueeze(-2).unsqueeze(-2)

        # ## Batch computation
        # Compute inner product and expand dimensions to compute all vector-vector operations
        inner_product_x1 = self.inner_product(x1_p, bns).unsqueeze(-2)
        inner_product_x2 = self.inner_product(x2_p, bns).unsqueeze(-3)
        # Repeat data along -2 and -3 dimensions to have b1 x ... x ndata_x x ndata_y x dim arrays
        inner_product_x1 = torch.cat(inner_product_x2.shape[-2] * [inner_product_x1], dim=-2)
        inner_product_x2 = torch.cat(inner_product_x1.shape[-3] * [inner_product_x2], dim=-3)

        # Compute terms of the inner sum over b: we here use one sample on the circle for each element of the outer sum
        exponential_x1p = torch.exp(phi_l_coefficient * inner_product_x1)
        exponential_x2p = torch.exp(phi_l_coefficient * inner_product_x2)
        phi_l = exponential_x1p * torch.conj(exponential_x2p)

        # Compute heat kernel ~ outer sum over p
        outer_sum = torch.sum(pms_factor * phi_l, -1) / self.nb_points_integral

        # With normalization factor
        phi_l_normal = exponential_x1p[..., 0, 0, :] * torch.conj(exponential_x1p[..., 0, 0, :])
        normalization_factor = 1. / (torch.sum(pms_factor[..., 0, 0, :] * phi_l_normal, -1) / self.nb_points_integral)
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

