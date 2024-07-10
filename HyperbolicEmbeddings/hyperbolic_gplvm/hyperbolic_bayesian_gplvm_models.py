import numpy as np
import torch
import warnings

from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.means import ZeroMean
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.priors.prior import Prior
import gpytorch.priors.torch_priors as torch_priors
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel
from gpytorch.distributions import MultivariateNormal

from geoopt.manifolds import Lorentz
from geoopt import ManifoldParameter

from HyperbolicEmbeddings.kernels.kernels_hyperbolic import HyperbolicRiemannianGaussianKernel
from HyperbolicEmbeddings.hyperbolic_distributions.hyperbolic_wrapped_normal import LorentzWrappedNormal, \
    LorentzWrappedNormalPrior
from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_latent_variable import HyperbolicVariationalLatentVariable
from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_gplvm_initializations import hyperbolic_tangent_pca_initialization


class HyperbolicVariationalBayesianGPLVM(BayesianGPLVM):
    """
    This class implements a variational Bayesian Gaussian Process Latent Variable Model (GPLVM) class.
    The latent variables belong to the hyperbolic manifold and follow a wrapped Gaussian variational distribution q(X).
    Therefore, the inducing inputs also belong to the hyperbolic manifold.

    The Lorentz model of the hyperbolic manifold is used for all computations.

    Attributes
    ----------
    self.X: hyperbolic latent variables, an instance of HyperbolicEmbeddings.hyperbolic_gplvm.
                                                        hyperbolic_latent_variable.HyperbolicVariationalLatentVariable
    self.variational_strategy: The strategy that determines how the model marginalizes over the variational distribution
                          (over inducing points) to produce the approximate posterior distribution (over data),
                          an instance of gpytorch.variational._VariationalStrategy
    self.train_targets: observations [dim x n]
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: hyperbolic kernel of the GPs defining the generative mapping
    self.n: number of observations (data)
    self.batch_shape: dimension of the Euclidean observations space R^D, used as batch dimension
    self.inducing_input: hyperbolic inducing inputs in the latent space R^Q
    self.added_loss: additional loss added to the log posterior during MAP estimation

    Methods
    -------
    self.forward(self, x)
    self.train_inputs(self)
    self._get_batch_idx(self)

    """
    def __init__(self, data: torch.Tensor, latent_dim: int, n_inducing: int, X_init: torch.Tensor = None,
                 latent_prior: Prior = None, kernel_lengthscale_prior: Prior = None):
        """
        Parameters
        ----------
        :param data: observations  [n x dimension]
        :param latent_dim: dimension of the latent hyperbolic manifold H^n
               Poincaré -> data dimension = latent_dim
               Lorentz  -> data dimension = latent_dim + 1
        :param n_inducing: number of inducing variables
        :param X_init: initial latent variables, if None, initialized randomly
        :param latent_prior: prior on the latent variable
        :param kernel_lengthscale_prior: prior on the lengthscale parameter of the hyperbolic kernel
        :param k2_single_sum_computation: if True, uses single sum approximation instead of double sum when latent_dim=2
        """
        data_dim = data.shape[1]
        n = data.shape[0]
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.train_targets = data.T

        # Define prior for X
        if latent_prior is None:
            X_prior_mean = torch.zeros(n, latent_dim+1)  # Mean on the Lorentz model -> N x Q+1
            X_prior_mean[:, 0] = 1.
            X_prior_scale = 2.0 * torch.ones(n, latent_dim)  # On the tangent space of the mean on the Lorentz model -> N x Q
            prior_x = LorentzWrappedNormalPrior(X_prior_mean, X_prior_scale)
        else:
            prior_x = latent_prior

        # Define a distribution to sample initial X and initial inducing inputs
        sample_mean = torch.zeros(latent_dim + 1)  # Mean on the Lorentz model -> Q+1
        sample_mean[0] = 1.
        sample_scale = 0.1 * torch.ones(latent_dim)  # On the tangent space of the mean on the Lorentz model -> Q
        sample_distribution = LorentzWrappedNormal(sample_mean, sample_scale)

        # Initialise X
        if X_init is None:
            print('No initialization given, the latent variables are initialized randomly.')
            X_init = torch.nn.Parameter(sample_distribution.sample((1, n)).squeeze())

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        # For the hyperbolic GPLVM, the inducing inputs belong to H^n
        self.inducing_inputs = sample_distribution.sample((data_dim, n_inducing))

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # LatentVariable
        X = HyperbolicVariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # Replace torch Parameters which are on the hyperbolic space by ManifoldParameters
        q_f.inducing_points = ManifoldParameter(q_f.inducing_points, manifold=Lorentz())  # Inducing points
        X.q_mu = ManifoldParameter(X.q_mu, manifold=Lorentz())  # Mean of latent points

        super().__init__(X, q_f)

        # Model parameters
        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(batch_shape=self.batch_shape)
        outputscale_prior = torch_priors.GammaPrior(2.0, 0.5)
        self.covar_module = ScaleKernel(HyperbolicRiemannianGaussianKernel(latent_dim, nb_points_integral=1000,
                                                                           lengthscale_prior=
                                                                           kernel_lengthscale_prior,
                                                                           batch_shape=self.batch_shape
                                                                           ),
                                        outputscale_prior=outputscale_prior)

        # Initialize an added loss term
        self.added_loss = None

    def forward(self, x):
        """
        In prior mode (self.train()): returns the prior of the GPLVM on the observation corresponding to the
        latent variable(s) x.
        In posterior mode (self.eval()): returns the posterior of the GPLVM on the observation corresponding to the
        latent variable(s) x.

        Parameters
        ----------
        :param x: point(s) in the latent space [n x latent dimension]

        Return
        ------
        :return prior or posterior of the GPLVM at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        dist = MultivariateNormal(mean_x, covar_x)

        if self.added_loss:
            self.added_loss.x = x
            self.update_added_loss_term("added_loss", self.added_loss)

        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

    def train_inputs(self):
        """
        Returns a sample from the latent variable distribution.
        """
        return self.sample_latent_variable()

    def add_loss_term(self, added_loss: AddedLossTerm):
        """
        Register an added loss term.

        Parameters
        ----------
        :param added_loss: additional loss added to the log posterior during MAP estimation
        """
        self.added_loss = added_loss
        self.register_added_loss_term("added_loss")