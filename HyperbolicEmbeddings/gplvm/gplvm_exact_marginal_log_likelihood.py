from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal


class GPLVMExactMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    The exact marginal log likelihood (MLL) for an exact GPLVM with a Gaussian likelihood.
    The difference with the ExactMarginalLogLikelihood function from GPytorch is the _add_other_terms function,
    in which the sum is computed over the entire prior instead of over batches.
    This is because for the GPLVM the batch size of the prior is the shape of the latent points, while the batch size of
    the likelihood is the dimension of the original points. Therefore, the batch size cannot be considered in the sum.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    """

    def __init__(self, likelihood, model):
        super(GPLVMExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for name, module, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure(module)).sum())

        return res

    def loglikelihood(self, function_dist, target, *params):
        r"""
        Computes the log-likelihood given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = output.log_prob(target).sum()

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)

    def forward(self, function_dist, target, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = output.log_prob(target).sum()
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)