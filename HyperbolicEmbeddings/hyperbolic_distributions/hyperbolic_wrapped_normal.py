from typing import Any, Tuple, Optional
import torch
from torch import Tensor
from torch.nn import Module
import torch.distributions
from torch.distributions.kl import register_kl

from gpytorch.priors import Prior
from gpytorch.priors.utils import _bufferize_attributes
from HyperbolicEmbeddings.hyperbolic_manifold.math_ops_hyperbolic import logsinh
from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import lorentz_norm, \
    inverse_sample_projection_mu0, sample_projection_mu0


class LorentzWrappedNormal(torch.distributions.Distribution):
    """
    This class has been adapted from:
        https://github.com/joeybose/HyperbolicNF/blob/master/distributions/wrapped_normal.py
    This class creates a wrapped Gaussian distribution on the Lorentz model of the hyperbolic manifold H^n.
    The distribution is parametrized by a mean vector and a covariance matrix.
    Samples following this distribution satisfy y = exp_m(PT_m0->m (v) ), with v = [0, \tilde{v}],
     \tilde{v} ~ Normal(0, S), and m0 = [1, 0, 0, ...]

    Reference:
    [1] Nagano, Y., Yamaguchi, S., Fujita, Y. & Koyama, M. A Wrapped Normal Distribution on Hyperbolic Space for
        Gradient-Based Learning. ICML 2019.
    """
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc: Tensor, scale: Tensor, radius: Tensor = torch.tensor(1.0), *args: Any, **kwargs: Any) -> None:
        """
        Initialization.

        Parameters
        ----------
        :param loc: mean on the hyperbolic manifold             batch x n+1
        :param scale: covariance on the tangent space at m0     batch x n x n

        Optional parameters
        -------------------
        :param radius: radius of the hyperbolic manifold (default: 1)
        """
        self.dim = loc.shape[-1]
        self.radius = radius
        tangent_dim = self.dim - 1

        if scale.shape[-1] > 1 and scale.shape[-1] != tangent_dim:
            raise ValueError("Invalid scale dimension: neither isotropic nor elliptical.")

        if scale.shape[-1] == 1:  # repeat along last dim for (loc.shape[-1] - 1) times.
            s = [1] * len(scale.shape)
            s[-1] = tangent_dim
            scale = scale.repeat(s)  # Expand scalar scale to vector.

        # Loc has to be one dim bigger than scale or equal (in projected spaces).
        assert loc.shape[:-1] == scale.shape[:-1]
        assert tangent_dim == scale.shape[-1]

        self.loc = loc
        self.scale = scale
        self.device = self.loc.device
        smaller_shape = self.loc.shape[:-1] + torch.Size([tangent_dim])
        self.normal = torch.distributions.Normal(torch.zeros(smaller_shape, device=self.device), scale)
        super(LorentzWrappedNormal, self).__init__(*args, **kwargs)

    def logdet_variable_change(self, data: Tensor, radius: Tensor = torch.tensor(1.0)) -> Tensor:
        """
        Computes the change of variable term from the normal distribution to the wrapped normal distribution, i.e.,
        log(p(y)) = log(p(x)) - logdet(df/dx) with y = f(x)

        For the Lorentz model of the hyperbolic manifold, logdet(df/dx) = (n-1) log(sinh(r)) - log(r) with r = ||u||_H

        Parameters
        ----------
        :param data: data on the hyperbolic manifold        batch x n+1

        Optional parameters
        -------------------
        :param radius: radius of the hyperbolic manifold (default: 1)

        Returns
        -------
        :return: logdet(df/dx) for all data                 batch
        """
        r = lorentz_norm(data, dim=-1) / radius
        n = data.shape[-1] - 1
        logdet_partial = (n - 1) * (torch.log(radius) + logsinh(r) - torch.log(r))
        assert torch.isfinite(logdet_partial).all()
        return logdet_partial

    def rsample_with_parts(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        Samples from the distribution. Samples are first drawn from a Euclidean normal distribution in the tangent space
        of mu0, then parallel transported to the tangent space of mu, and projected on the manifold using the
        exponential map at mu.

        Parameters
        ----------
        :param sample_shape: desired shape of samples

        Returns
        -------
        :return z: samples of shape sample_shape x n+1
        :return helper_data
        """
        # v ~ N(0, \Sigma)
        v_tilde = self.normal.rsample(shape)
        assert torch.isfinite(v_tilde).all()
        # u = PT_{mu_0 -> mu}([0, v_tilde])
        # z = exp_{mu}(u)
        z, helper_data = sample_projection_mu0(v_tilde, at_point=self.loc, radius=self.radius)
        assert torch.isfinite(z).all()
        return z, helper_data

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """
        Samples from the wrapped distribution.

        Parameters
        ----------
        :param sample_shape: desired shape of samples

        Returns
        -------
        :return: samples of shape sample_shape x n+1
        """
        z, _ = self.rsample_with_parts(sample_shape)
        return z

    def log_prob_from_parts(self, data: Tuple[Tensor, ...]) -> Tensor:
        """
        Computes the log probability of the wrapped Gaussian distribution by exploiting the change of variable as
        log(p(y)) = log(p(x)) - logdet(df/dx) with y = f(x).

        Parameters
        ----------
        :param data: tuple containing (0) the data projected in the tangent space of the mean (with the logmap) and
                    (1) the same data parallel transported to the tangent space at mu0

        Returns
        -------
        :return: log probability
        """
        if data is None:
            raise ValueError("Additional data cannot be empty for WrappedNormal.")

        # log(z) = log p(v) - log det [(\partial / \partial v) proj_{\mu}(v)]
        v = data[1]
        assert torch.isfinite(v).all()
        n_logprob = self.normal.log_prob(v).sum(dim=-1)
        logdet = self.logdet_variable_change(data[0], self.radius)
        assert n_logprob.shape == logdet.shape
        log_prob = n_logprob - logdet
        assert torch.isfinite(log_prob).all()
        return log_prob

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes the log probability (should only be used for p_z, prefer log_prob_from_parts).

        Parameters
        ----------
        :param z: data          batch x n+1

        Returns
        -------
        :return log of the probability of z
        """
        assert torch.isfinite(z).all()
        data = inverse_sample_projection_mu0(z, at_point=self.loc, radius=self.radius)
        return self.log_prob_from_parts(data)

    def rsample_log_prob(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tensor]:
        """
        Samples from the wrapped distribution and computes the log probability of the samples.

        Parameters
        ----------
        :param sample_shape: desired shape of samples

        Returns
        -------
        :return z: samples of shape sample_shape x n+1
        :return: log probability of z
        """
        z, data = self.rsample_with_parts(shape)
        return z, self.log_prob_from_parts(data)

    def log_prob_per_dim(self, z: Tensor) -> Tensor:
        """
        Computes the log probability for each dimension of z.
        This function is the same as log_prob, but does not sum the log probability over the dimensions.

        Parameters
        ----------
        :param z: data          batch x n+1

        Returns
        -------
        :return log of the probability of z for each dimension of z
        """
        assert torch.isfinite(z).all()
        data = inverse_sample_projection_mu0(z, at_point=self.loc, radius=self.radius)
        v = data[1]
        assert torch.isfinite(v).all()
        n_logprob = self.normal.log_prob(v)
        logdet = self.logdet_variable_change(data[0], self.radius).unsqueeze(-1)
        assert n_logprob.shape[:-1] == logdet.shape[:-1]
        log_prob = n_logprob - logdet
        assert torch.isfinite(log_prob).all()
        return log_prob


class LorentzWrappedNormalPrior(Prior, LorentzWrappedNormal):
    """
    This class creates a Lorentz Wrapped Gaussian Prior
    Samples following this distribution satisfy y = exp_m(PT_m0->m (v) ), with v = [0, \tilde{v}],
     \tilde{v} ~ Normal(0, S), and m0 = [1, 0, 0, ...]

    pdf(x) = (2 * pi * sigma^2)^-0.5 * exp(-(x)^2 / (2 * sigma^2)) - (n-1) log(sinh(r)) - log(r) with r = ||x||_H

    where mu is the mean and sigma^2 is the variance.
    """

    def __init__(self, loc, scale, radius=torch.tensor(1.0), validate_args=False, transform=None):
        Module.__init__(self)
        LorentzWrappedNormal.__init__(self, loc=loc, scale=scale, radius=radius, validate_args=validate_args)
        _bufferize_attributes(self, ("loc", "scale"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return LorentzWrappedNormalPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))


@register_kl(LorentzWrappedNormal, LorentzWrappedNormal)
def sample_based_kl_lorentz_wrapped(p, q, nb_samples=10):
    """
    Computes the KL divergence between two wrapped Gaussian distribution on the Lorentz model of the hyperbolic manifold
    The KL divergence is computed by sampling.

    Parameters
    ----------
    :param p: Lorentz wrapped Gaussian distribution
    :param q: Lorentz wrapped Gaussian distribution

    Optional Parameters
    -------------------
    :param nb_samples: number of samples used to compute the KL divergence

    Returns
    -------
    :return: KL divergence between p and q
    """
    z = p.rsample((nb_samples, 1)).squeeze()
    logp = p.log_prob_per_dim(z)
    logq = q.log_prob_per_dim(z)
    kl = (logp - logq).mean(0)
    return kl   # should return nb_data x dim
