import numpy as np
import torch
import warnings

from gpytorch.models.gplvm.latent_variable import MAPLatentVariable, PointLatentVariable, LatentVariable
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood, Likelihood, MultitaskGaussianLikelihood
from gpytorch.priors.prior import Prior
from gpytorch.priors import NormalPrior
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.mlls.added_loss_term import AddedLossTerm

from HyperbolicEmbeddings.gplvm.gp_models import ExactGP
from HyperbolicEmbeddings.gplvm.latent_variable import BackConstraintsLatentVariable, \
    TaxonomyBackConstraintsLatentVariable


class ExactGPLVM(ExactGP):
    """
    This class implements a Gaussian Process Latent Variable Model (GPLVM) class.
    The model is estimated exactly as opposed to variational models, which use inducing points for their estimation.

    Attributes
    ----------
    self.X: latent variables, an instance of gpytorch.models.gplvm.PointLatentVariable or
            gpytorch.models.gplvm.MAPLatentVariable
    self.train_targets: observations in the original space
    self.likelihood: likelihood of the GPs defining the generative mapping,
                     an instance of gpytorch.likelihoods.GaussianLikelihood
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: kernel of the GPs defining the generative mapping
    self.batch_shape: dimension of the Euclidean observations space R^D, used as batch dimension
    self.added_loss: additional loss added to the log posterior during MAP estimation

    Methods
    -------
    self.forward(self, x)
    self.sample_latent_variable(self)
    self._get_batch_idx(self)
    self.add_loss_term(added_loss)

    """

    def __init__(self, X: LatentVariable, train_targets: torch.Tensor, likelihood: Likelihood,
                 mean_module, covar_module):

        super().__init__(X(), train_targets, likelihood)

        self.X = X

        self.mean_module = mean_module
        self.covar_module = covar_module

        # Initialize an added loss term
        self.added_loss = None

    @property
    def train_inputs(self) -> tuple[torch.Tensor]:
        # train_inputs follow the shape as in gpytorch.models.ExactGP __init__()
        train_inputs = self.X()
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        return tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs) if train_inputs is not None else None

    @train_inputs.setter
    def train_inputs(self, _) -> None:
        # cannot set train_inputs directly as they a defined by the LatentVariable object
        pass

    def forward(self, x: torch.Tensor) -> MultitaskMultivariateNormal:
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
            self.update_added_loss_term("added_loss", self.added_loss)

        return dist
        # return MultitaskMultivariateNormal.from_batch_mvn(dist)

    def add_loss_term(self, added_loss: AddedLossTerm):
        """
        Register an added loss term.

        Parameters
        ----------
        :param added_loss: additional loss added to the log posterior during MAP estimation
        """
        self.added_loss = added_loss
        self.register_added_loss_term("added_loss")


class EuclideanExactGPLVM(ExactGPLVM):
    """
    This class implements an Gaussian Process Latent Variable Model (GPLVM) class.
    The model is estimated exactly as opposed to variational models, which use inducing points for their estimation.
    The latent variables are inferred as point estimates for latent X.

    Attributes
    ----------
    self.X: latent variables, an instance of gpytorch.models.gplvm.PointLatentVariable or
            gpytorch.models.gplvm.MAPLatentVariable
    self.train_targets: observations in the original space
    self.likelihood: likelihood of the GPs defining the generative mapping,
                     an instance of gpytorch.likelihoods.GaussianLikelihood
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: kernel of the GPs defining the generative mapping
    self.batch_shape: dimension of the Euclidean observations space R^D or torch.Size(), used as batch dimension
    self.added_loss: additional loss added to the log posterior during MAP estimation


    """
    def __init__(self, X: LatentVariable, data: torch.Tensor,
                 kernel_lengthscale_prior: Prior = None, kernel_outputscale_prior: Prior = None, batch_params=True):
        """
        Initialization.

        Parameters
        ----------
        :param data: observations  [n x dimension]

        Optional parameters
        -------------------
        :param kernel_lengthscale_prior: prior on the lengthscale parameter of the hyperbolic kernel
        :param kernel_outputscale_prior: prior on the scale parameter of the hyperbolic kernel
        """
        data_dim = data.shape[1]
        n = data.shape[0]
        if batch_params:
            self.batch_shape = torch.Size([data_dim])
        else:
            self.batch_shape = torch.Size()
        self.n = n

        # Gaussian likelihood
        likelihood = GaussianLikelihood(batch_shape=self.batch_shape)
        # likelihood = MultitaskGaussianLikelihood(data_dim, has_global_noise=False)

        # Kernel (acting on latent dimensions)
        mean_module = ZeroMean(batch_shape=self.batch_shape)
        covar_module = ScaleKernel(RBFKernel(batch_shape=self.batch_shape,
                                             lengthscale_prior=kernel_lengthscale_prior),
                                   outputscale_prior=kernel_outputscale_prior)

        super().__init__(X, data.T, likelihood, mean_module, covar_module)


class MapExactGPLVM(EuclideanExactGPLVM):
    """
    This class implements a Gaussian Process Latent Variable Model (GPLVM) class.
    The model is estimated exactly as opposed to variational models, which use inducing points for their estimation.
    The latent variables are inferred using MAP Inference.

    Attributes
    ----------
    self.X: latent variables, an instance of gpytorch.models.gplvm.PointLatentVariable or
            gpytorch.models.gplvm.MAPLatentVariable
    self.train_targets: observations in the original space
    self.likelihood: likelihood of the GPs defining the generative mapping,
                     an instance of gpytorch.likelihoods.GaussianLikelihood
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: kernel of the GPs defining the generative mapping
    self.batch_shape: dimension of the Euclidean observations space R^D or torch.Size(), used as batch dimension
    self.added_loss: additional loss added to the log posterior during MAP estimation


    """
    def __init__(self,  data: torch.Tensor, latent_dim: int, X_init: torch.Tensor = None, latent_prior: Prior = None,
                 **kwargs):
        """
        Initialization.

        Parameters
        ----------
        :param data: observations  [n x dimension]
        :param latent_dim: dimension of the latent space

        Optional parameters
        -------------------
        :param X_init: initial latent variables, if None, initialized randomly
        :param map_latent_variable: if True, use a MAP latent variable, otherwise use a point latent variable
        :param latent_prior: prior on the latent variable
        """
        data_dim = data.shape[1]
        n = data.shape[0]

        # Define prior for X
        if latent_prior is None:
            X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
        else:
            prior_x = latent_prior

        # Initialise X
        if X_init is None:
            print('No initialization given, the latent variables are initialized randomly.')
            X_init = torch.nn.Parameter(torch.randn(n, latent_dim))
        else:
            X_init = torch.nn.Parameter(X_init)

        # MAP latent variable
        X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, data, **kwargs)


class BackConstrainedExactGPLVM(EuclideanExactGPLVM):
    """
    This class implements a Back-constrained Bayesian Gaussian Process Latent Variable Model (GPLVM) class.
    The model is estimated exactly as opposed to variational models, which use inducing points for their estimation.
    The latent variables are inferred as a function of the data and of their given classes (relevant when the data are
    instances of a taxonomy).

    Attributes
    ----------
    self.X: latent variables, an instance of HyperbolicEmbeddings.gplvm.latent_variable.BackConstraintsLatentVariable or
            HyperbolicEmbeddings.gplvm.latent_variable.TaxonomyBackConstraintsLatentVariable
    self.train_targets: observations in the original space
    self.likelihood: likelihood of the GPs defining the generative mapping,
                     an instance of gpytorch.likelihoods.GaussianLikelihood
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: kernel of the GPs defining the generative mapping
    self.batch_shape: dimension of the Euclidean observations space R^D or torch.Size(), used as batch dimension
    self.added_loss: additional loss added to the log posterior during MAP estimation

    Methods
    -------
    self.forward(self, x)
    self.train_inputs(self)
    self._get_batch_idx(self)
    self.add_loss_term(added_loss)

    """
    def __init__(self, data: torch.Tensor, latent_dim: int, class_idx: torch.Tensor, X_init: torch.Tensor = None,
                 taxonomy_based_back_constraints: bool = True,
                 data_kernel: Kernel = None, classes_kernel: Kernel = None, weight_prior: Prior = None, **kwargs):
        """
        Initialization.

        Parameters
        ----------
        :param data: observations  [n x dimension]
        :param latent_dim: dimension of the latent space
        :param class_idx: indices of data classes

        Optional parameters
        -------------------
        :param X_init: initial latent variables, if None, initialized randomly
        :param taxonomy_based_back_constraints: if True, use the taxonomy-based back constraints
        :param data_kernel: kernel defining the relationship between the observations for the back constraints
        :param classes_kernel: kernel defining the relationship between the data classes for the back constraints
        :param weight_prior: prior for the weight parameter of the back constraints
        """
        data_dim = data.shape[1]
        n = data.shape[0]

        # Initialise the weights
        if X_init is None:
            print('No initialization given, the latent variables are initialized randomly.')
            weight_init = torch.nn.Parameter(0.01 * torch.randn(n, latent_dim))
        else:
            X_init = torch.nn.Parameter(X_init)
            # Compute least squares weights
            if taxonomy_based_back_constraints:
                kernel_backconstraints = (data_kernel(data) * classes_kernel(class_idx)).evaluate()
            else:
                kernel_backconstraints = data_kernel(data).evaluate()
            regularization = 1e-6 * torch.eye(n)
            weight_init = torch.matmul(torch.inverse(kernel_backconstraints + regularization), X_init)

        # Back constrained latent variables
        if taxonomy_based_back_constraints:
            X = TaxonomyBackConstraintsLatentVariable(n, latent_dim, data, class_idx, classes_kernel,
                                                      data_kernel, weight_prior, weight_init)
        else:
            X = BackConstraintsLatentVariable(n, latent_dim, data, data_kernel, weight_prior, weight_init)

        super().__init__(X, data, **kwargs)

