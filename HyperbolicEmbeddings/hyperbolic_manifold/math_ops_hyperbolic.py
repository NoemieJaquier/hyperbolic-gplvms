from typing import Any, Optional, Tuple
import math
import torch
from torch.autograd.function import FunctionCtx

"""
This file was partially copied from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/math_ops.py . The functions AcoshSquared and acosh_squared were added.
"""

eps = 1e-8
max_norm = 85


class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None


def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)


class Acosh(torch.autograd.Function):
    """
    Numerically stable arccosh that never returns NaNs.
    Returns acosh(x) = arccosh(x) = log(x + sqrt(max(x^2 - 1, eps))).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        x = clamp(x, min=1 + eps)
        z = sqrt(x * x - 1.)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        z, = ctx.saved_tensors
        # z_ = clamp(z, min=eps)
        z_ = z
        return grad_output / z_


class AcoshSquared(torch.autograd.Function):
    """stable version of acosh(x)**2 that allows to compute derivatives at x=1 and returns 0 for x<1 instead of nan"""

    @staticmethod
    def forward(ctx: FunctionCtx, X: torch.Tensor) -> torch.Tensor:
        """
        acosh(x)**2

        Parameters
        -
        X: torch.shape([M])  M sclar inputs

        Returns
        -
        acosh2: torch.shape([M])  M scalar outputs
        """
        acosh = torch.acosh(X)
        acosh = torch.where(torch.isnan(acosh), 0, acosh)
        ctx.save_for_backward(X, acosh)
        return acosh**2

    @staticmethod
    def backward(ctx: FunctionCtx, diff_acosh2_L: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -
        diff_acosh2_L: torch.shape([M])   derivative of scalar Loss w.r.t. acosh2

        Returns
        -
        diff_x_L: torch.shape([M])   derivative of scalar Loss w.r.t. x
        """
        X, acosh = ctx.saved_tensors
        diff_x_acosh2 = 2*acosh/torch.sqrt(X**2 - 1)
        diff_x_acosh2 = torch.where(torch.isnan(diff_x_acosh2), 2., diff_x_acosh2)
        return diff_x_acosh2 * diff_acosh2_L
    

def acosh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable arccosh that never returns NaNs.
    :param x: The input tensor.
    :return: log(x + sqrt(max(x^2 - 1, eps))
    """
    return Acosh.apply(x)


def acosh_squared(X: torch.Tensor) -> torch.Tensor:
    return AcoshSquared.apply(X)


def cosh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.cosh(x)


def sinh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.sinh(x)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=1e-9)  # Smaller epsilon due to precision around x=0.
    return torch.sqrt(x)


def logsinh(x: torch.Tensor) -> torch.Tensor:
    x_exp = x.unsqueeze(dim=-1)
    signs = torch.cat((torch.ones_like(x_exp), -torch.ones_like(x_exp)), dim=-1)
    value = torch.cat((torch.zeros_like(x_exp), -2. * x_exp), dim=-1)
    return x + logsumexp_signs(value, dim=-1, signs=signs) - math.log(2)


def logsumexp_signs(value: torch.Tensor, dim: int = 0, keepdim: bool = False,
                    signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    if signs is None:
        signs = torch.ones_like(value)
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(clamp(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim), min=eps))


def expand_proj_dims(x: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(x.shape[:-1] + torch.Size([1])).to(x.device).to(x.dtype)
    return torch.cat((zeros, x), dim=-1)
