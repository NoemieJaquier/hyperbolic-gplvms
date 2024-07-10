import torch
import torch.nn.functional

from gpytorch.models.gplvm.latent_variable import LatentVariable
from gpytorch.mlls import KLGaussianAddedLossTerm
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel
from gpytorch.priors import Prior

from HyperbolicEmbeddings.hyperbolic_distributions.hyperbolic_wrapped_normal import LorentzWrappedNormal
from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import exp_map_mu0


class HyperbolicVariationalLatentVariable(LatentVariable):
    """
    This class is used for GPLVM models to recover a variational approximation of the latent variables.
    The variational approximation will be an isotropic wrapped Gaussian distribution on the hyperbolic manifold.

    Parameters
    ----------
    :param int n: Size of the latent space.
    :param int data_dim: Dimensionality of the :math:`\\mathbf Y` values.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X` on the hyperbolic manifold
    :param ~gpytorch.priors.Prior prior_x: hyperbolic prior for :math:`\\mathbf X`
    """

    def __init__(self, n, data_dim, latent_dim, X_init, prior_x):
        super().__init__(n, latent_dim)

        self.data_dim = data_dim
        self.prior_x = prior_x

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init)
        self.q_log_sigma = 1e-2 * torch.nn.Parameter(torch.randn(n, latent_dim))
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")

    def forward(self):
        # Variational distribution over the latent variable q(x)
        q_x = LorentzWrappedNormal(self.q_mu, torch.nn.functional.softplus(self.q_log_sigma))
        x_kl = KLGaussianAddedLossTerm(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term("x_kl", x_kl)  # Update the KL term
        return q_x.rsample()


class HyperbolicBackConstraintsLatentVariable(LatentVariable):
    """
    This class is used for GPLVM models with hyperbolic latent spaces to estimate the hyperbolic latent variables X
    with back constraints.
    The latent variable x is computed as a function of the corresponding observations y in the latent space of mu0 =
    [1, 0, ..., 0] and then projected on the hyperbolic manifold using the exponential map.
    Namely, given the training targets (or observations) y_n, each dimension j of x is computed as
    x_j = exp_mu0( \sum_n w_jn k(y, y_n) )
    with k a kernel function on the observation space.

    Attributes
    ----------
    self.n (int): size of the latent space.
    self.latent_dim (int): dimensionality of latent space.
    self.train_targets (torch.Tensor): training observations  [n x dimension]
    self.targets_kernel (gpytorch.Kernel): kernel defining the relationship between the observations for the BC function

    Methods
    -------
    self.forward(self)
    self.back_constraint_function(self, targets)
    """

    def __init__(self, n, latent_dim, train_targets, targets_kernel: Kernel = None, weight_prior: Prior = None,
                 weight_init: torch.Tensor = None):
        """
        Initialization.

        Parameters
        ----------
        :param n: size of the latent space
        :param latent_dim: dimension of the latent space
        :param train_targets: training observations  [n x dimension]

        Optional parameters
        -------------------
        :param targets_kernel: kernel defining the relationship between the observations for the back constraints
        :param weight_prior: prior for the weight parameter of the back constraints
        :param weight_init: initial weight parameter
        """
        super().__init__(n, latent_dim)

        # Data
        self.train_targets = train_targets

        # Define kernels
        if targets_kernel:
            self.targets_kernel = targets_kernel
        else:
            self.targets_kernel = ScaleKernel(RBFKernel())

        # Ensure that kernel parameters are not trainable (necessary for back constraints)
        for parameter in self.targets_kernel.parameters():
            parameter.requires_grad = False

        # Define back constraints weights
        if weight_init is None:
            weight_init = torch.nn.Parameter(torch.randn(n, latent_dim))
        else:
            weight_init = torch.nn.Parameter(weight_init)
        self.register_parameter("weights", weight_init)
        if weight_prior:
            self.register_prior("prior_x", weight_prior, "weights")

    def back_constraint_function(self, targets):
        """
        Computes the back constraint function x_j = exp_mu0( \sum_n w_jn k(y, y_n) ).

        Parameters
        ----------
        :param targets: observations [nb_data x dimension]

        Returns
        -------
        :return x: latent variables [nb_data x latent dimension]
        """
        # Compute kernel matrix
        kernel_matrix = self.targets_kernel(targets, self.train_targets)

        # Get latent variable in the tangent space of mu0
        x_tangent = torch.matmul(kernel_matrix.evaluate(), self.weights)
        x_tangent_complete = torch.hstack((torch.zeros(targets.shape[0], 1), x_tangent))
        # Project onto the hyperbolic manifold
        x = exp_map_mu0(x_tangent_complete)
        return x

    def forward(self):
        """
        Computes the back constraint function for the training targets.

        Returns
        -------
        :return x: latent variables corresponding to the training targets [nb_data x latent dimension]
        """
        x = self.back_constraint_function(self.train_targets)
        return x


class HyperbolicTaxonomyBackConstraintsLatentVariable(HyperbolicBackConstraintsLatentVariable):
    """
    This class is used for GPLVM models with hyperbolic latent spaces to estimate the hyperbolic latent variables X
    with back constraints.
    The latent variable x is computed as a function of the corresponding observations y in the latent space of mu0 =
    [1, 0, ..., 0] and then projected on the hyperbolic manifold using the exponential map.
    Namely, given the training targets (or observations) y_n, each dimension j of x is computed as
    x_j = exp_mu0( \sum_n w_jn k_y(y, y_n) k_c(c, c_n) )
    with k_y and k_c kernel functions on the observation and class spaces, respectively.

    Attributes
    ----------
    self.n (int): size of the latent space.
    self.latent_dim (int): dimensionality of latent space.
    self.train_targets (torch.Tensor): training observations  [n x dimension]
    self.targets_kernel (gpytorch.Kernel): kernel defining the relationship between the observations for the BC function
    self.classes_kernel (gpytorch.Kernel): kernel defining the relationship between the data classes for the BC function

    Methods
    -------
    self.forward(self)
    self.back_constraint_function(self, targets)
    """

    def __init__(self, n, latent_dim, train_targets, train_class_idx, classes_kernel: Kernel,
                 targets_kernel: Kernel = None,  weight_prior: Prior = None, weight_init: torch.Tensor = None):
        """
        Initialization.

        Parameters
        ----------
        :param n: size of the latent space
        :param latent_dim: dimension of the latent space
        :param train_targets: training observations  [n x dimension]
        :param train_class_idx: indices of training data classes
        :param classes_kernel: kernel defining the relationship between the data classes for the back constraints

        Optional parameters
        -------------------
        :param targets_kernel: kernel defining the relationship between the observations for the back constraints
        :param weight_prior: prior for the weight parameter of the back constraints
        :param weight_init: initial weight parameter
        """

        super().__init__(n, latent_dim, train_targets, targets_kernel, weight_prior, weight_init)

        # Data
        self.train_class_idx = train_class_idx

        # Define class kernel
        self.classes_kernel = classes_kernel

        # Ensure that kernel parameters are not trainable (necessary for back constraints)
        for parameter in self.classes_kernel.parameters():
            parameter.requires_grad = False

    def back_constraint_function(self, targets, class_idx):
        """
        Computes the back constraint function x_j = exp_mu0( \sum_n w_jn k_y(y, y_n) k_c(c, c_n) ).

        Parameters
        ----------
        :param targets: observations [nb_data x dimension]
        :param class_idx: indices of current data classes

        Returns
        -------
        :return x: latent variables [nb_data x latent dimension]
        """
        # Compute the kernel matrix
        kernel_matrix = self.targets_kernel(targets, self.train_targets) * \
                        self.classes_kernel(class_idx, self.train_class_idx)

        # Get latent variable in the tangent space of mu0
        x_tangent = torch.matmul(kernel_matrix.evaluate(), self.weights)
        x_tangent_complete = torch.hstack((torch.zeros(targets.shape[0], 1), x_tangent))
        # Project onto the hyperbolic manifold
        x = exp_map_mu0(x_tangent_complete)
        return x

    def forward(self):
        """
        Computes the back constraint function for the training targets and classes.

        Returns
        -------
        :return x: latent variables corresponding to the training targets and classes [nb_data x latent dimension]
        """
        x = self.back_constraint_function(self.train_targets, self.train_class_idx)
        return x
