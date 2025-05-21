import numpy as np
import torch
import warnings

from gpytorch.models.gplvm.latent_variable import LatentVariable, MAPLatentVariable, PointLatentVariable
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.prior import Prior
from gpytorch.kernels import ScaleKernel, Kernel

from geoopt.manifolds import Lorentz
from geoopt import ManifoldParameter

import geometric_kernels.torch
from geometric_kernels.spaces import Hyperbolic
from geometric_kernels.kernels.feature_map import MaternFeatureMapKernel
from geometric_kernels.feature_maps.rejection_sampling import RejectionSamplingFeatureMapHyperbolic
from geometric_kernels.frontends.gpytorch import GPyTorchGeometricKernel

from HyperbolicEmbeddings.gplvm.gplvm_models import ExactGPLVM
from HyperbolicEmbeddings.kernels.kernels_hyperbolic import HyperbolicRiemannianGaussianKernel, LorentzGaussianKernel
from HyperbolicEmbeddings.hyperbolic_distributions.hyperbolic_wrapped_normal import LorentzWrappedNormal, \
    LorentzWrappedNormalPrior
from HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_latent_variable import HyperbolicBackConstraintsLatentVariable, \
    HyperbolicTaxonomyBackConstraintsLatentVariable
from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import log_map_mu0


class HyperbolicExactGPLVM(ExactGPLVM):
    """
    This class implements a Gaussian Process Hyperbolic Latent Variable Model (GPHLVM) class.
    The model is estimated exactly as opposed to variational models, which use inducing points for their estimation.
    The latent variables are inferred as point estimates for latent X.

    The Lorentz model of the hyperbolic manifold is used for all computations.

    Attributes
    ----------
    self.X: hyperbolic latent variables, an instance of gpytorch.models.gplvm.PointLatentVariable or
            gpytorch.models.gplvm.MAPLatentVariable
    self.train_targets: observations in the original space
    self.likelihood: likelihood of the GPs defining the generative mapping,
                     an instance of gpytorch.likelihoods.GaussianLikelihood
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: hyperbolic kernel of the GPs defining the generative mapping
    self.batch_shape: dimension of the Euclidean observations space R^D or torch.Size(), used as batch dimension
    self.added_loss: additional loss added to the log posterior during MAP estimation

    """
    def __init__(self, X: LatentVariable, data: torch.Tensor, latent_dim: int,
                 kernel_lengthscale_prior: Prior = None, kernel_outputscale_prior: Prior = None, batch_params=True, kernel_type="SlowHyperbolic"):
        """
        Initialization.

        Parameters
        ----------
        :param data: observations  [n x dimension]
        :param latent_dim: dimension of the latent space

        Optional parameters
        -------------------
        :param kernel_lengthscale_prior: prior on the lengthscale parameter of the hyperbolic kernel
        :param kernel_outputscale_prior: prior on the scale parameter of the hyperbolic kernel
        :param batch_params: batch dimension, either dimension of the Euclidean observations space R^D or torch.Size()
        :kernel_type: 
            "FastHyperbolic": use LorentzGaussianKernel. This is our new implementation, which we use to compute pullback metric and to train our models faster.
            "SlowHyperbolic": use HyperbolicRiemannianGaussianKernel. This is our old implementation, with which we trained the model of our ICML paper.
            "GeometricKernel": only with latent_dim=2, use the rejection sampling kernel from geometric_kernels
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

        # Model parameters
        # Kernel (acting on latent dimensions)
        mean_module = ZeroMean(batch_shape=self.batch_shape)
        if kernel_type == "FastHyperbolic":
            covar_module = ScaleKernel(LorentzGaussianKernel(dim=latent_dim, 
                                                            nb_points_integral=3000,
                                                            lengthscale_prior=kernel_lengthscale_prior, batch_shape=self.batch_shape), outputscale_prior=kernel_outputscale_prior)
            
        elif kernel_type == "SlowHyperbolic":
            covar_module = ScaleKernel(HyperbolicRiemannianGaussianKernel(latent_dim, nb_points_integral=3000,
                                                                        lengthscale_prior=kernel_lengthscale_prior,
                                                                        batch_shape=self.batch_shape),
                                    outputscale_prior=kernel_outputscale_prior)
        
        elif kernel_type == "GeometricKernel":
            if latent_dim==2:
                print('Using rejection sampling hyperbolic kernel')
                hyperboloid = Hyperbolic(dim=latent_dim)
                feature_map_rs = RejectionSamplingFeatureMapHyperbolic(hyperboloid, num_random_phases=3000, shifted_laplacian=False)
                kernel_rs = MaternFeatureMapKernel(hyperboloid, feature_map_rs, torch.Generator())  # torch.Generator('cuda' if cuda else None)
                base_kernel = GPyTorchGeometricKernel(kernel_rs)
                base_kernel.nu = torch.inf
                covar_module = ScaleKernel(base_kernel=base_kernel, outputscale_prior=kernel_outputscale_prior)
                # base_kernel.raw_nu_constraint = Positive()
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        super().__init__(X, data.T, likelihood, mean_module, covar_module)

        # Initialize lengthscale and outputscale to mean of priors
        if kernel_lengthscale_prior:
            self.covar_module.base_kernel.lengthscale = kernel_lengthscale_prior.mean
        if kernel_outputscale_prior:
            self.covar_module.outputscale = kernel_outputscale_prior.mean


class MapExactHyperbolicGPLVM(HyperbolicExactGPLVM):
    """
    This class implements a Hyperbolic Gaussian Process Latent Variable Model (GPHLVM) class.
    The model is estimated "exactly" as opposed to variational models, which use inducing points for their estimation.
    The latent variables are inferred using MAP Inference.

    Attributes
    ----------
    self.X: hyperbolic latent variables, an instance of gpytorch.models.gplvm.MAPLatentVariable
    self.train_targets: observations in the original space
    self.likelihood: likelihood of the GPs defining the generative mapping,
                     an instance of gpytorch.likelihoods.GaussianLikelihood
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: hyperbolic kernel of the GPs defining the generative mapping
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
            X_prior_mean = torch.zeros(n, latent_dim + 1)  # Mean on the Lorentz model -> N x Q+1
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

        # MAP latent variable
        X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        # Replace torch Parameters which are on the hyperbolic space by ManifoldParameters
        if not isinstance(X.X, ManifoldParameter):
            X.X = ManifoldParameter(X.X, manifold=Lorentz())  # Latent points

        super().__init__(X, data, latent_dim, **kwargs)


class BackConstrainedHyperbolicExactGPLVM(HyperbolicExactGPLVM):
    """
    This class implements a back-constrained Bayesian Gaussian Process Hyperbolic Latent Variable Model (GPHLVM) class.
    The model is estimated "exactly" as opposed to variational models, which use inducing points for their estimation.
    The latent variables belong to the hyperbolic manifold and is inferred as a function of the data and of their given
    classes (relevant when hyperbolic data are instances of a taxonomy).

    The Lorentz model of the hyperbolic manifold is used for all computations.

    Attributes
    ----------
    self.X: hyperbolic latent variables, instance of HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_latent_variable.
            HyperbolicBackConstraintsLatentVariable or HyperbolicEmbeddings.hyperbolic_gplvm.hyperbolic_latent_variable.
            HyperbolicTaxonomyBackConstraintsLatentVariable
    self.train_targets: observations in the original space
    self.likelihood: likelihood of the GPs defining the generative mapping,
                     an instance of gpytorch.likelihoods.GaussianLikelihood
    self.mean_module: mean of the GPs defining the generative mapping
    self.kernel_module: hyperbolic kernel of the GPs defining the generative mapping
    self.batch_shape: dimension of the Euclidean observations space R^D, used as batch dimension
    self.added_loss: additional loss added to the log posterior during MAP estimation

    Methods
    -------
    self.forward(self, x)
    self.train_inputs(self)
    self._get_batch_idx(self)
    self.add_loss_term(added_loss)

    """
    def __init__(self, data: torch.Tensor, latent_dim: int, hyperbolic_class_idx: torch.Tensor,
                 X_init: torch.Tensor = None, taxonomy_based_back_constraints: bool = True,
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
        :param data_kernel: kernel defining the relationship between the observations for the back constraints
        :param classes_kernel: kernel defining the relationship between the data classes for the back constraints
        :param weight_prior: prior for the weight parameter of the back constraints
        """
        data_dim = data.shape[1]
        n = data.shape[0]

        # Initialise X
        if X_init is None:
            print('No initialization given, the latent variables are initialized randomly.')
            weight_init = torch.nn.Parameter(0.01 * torch.randn(n, latent_dim))
        else:
            X_init_tangent = log_map_mu0(X_init)[:, 1:]
            # Compute least squares weights
            if taxonomy_based_back_constraints:
                kernel_backconstraints = (data_kernel(data) * classes_kernel(hyperbolic_class_idx)).evaluate()
            else:
                kernel_backconstraints = data_kernel(data).evaluate()
            regularization = 1e-6 * torch.eye(n)
            weight_init = torch.matmul(torch.inverse(kernel_backconstraints + regularization), X_init_tangent)

        # Back constrained latent variables
        if taxonomy_based_back_constraints:
            X = HyperbolicTaxonomyBackConstraintsLatentVariable(n, latent_dim, data,
                                                                hyperbolic_class_idx, classes_kernel, data_kernel,
                                                                weight_prior, weight_init)
        else:
            X = HyperbolicBackConstraintsLatentVariable(n, latent_dim, data, data_kernel, weight_prior,
                                                        weight_init)

        super().__init__(X, data, latent_dim, **kwargs)

