from typing import Any, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

from HyperbolicEmbeddings.hyperbolic_manifold.math_ops_hyperbolic import clamp, acosh, cosh, sinh, sqrt, \
    expand_proj_dims


def lorentz_distance_torch(x1, x2, diag=False):
    """
    This function computes the Riemannian distance between points on a hyperbolic manifold.

    Parameters
    ----------
    :param x1: points on the hyperbolic manifold                                N1 x dim or b1 x ... x bk x N1 x dim
    :param x2: points on the hyperbolic manifold                                N2 x dim or b1 x ... x bk x N2 x dim

    Optional parameters
    -------------------
    :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.

    Returns
    -------
    :return: matrix of manifold distance between the points in x1 and x2         N1 x N2 or b1 x ... x bk x N1 x N2
    """
    # Expand dimensions to compute all vector-vector distances
    x1 = x1.unsqueeze(-2)
    x2 = x2.unsqueeze(-3)

    # Repeat x and y data along -2 and -3 dimensions to have b1 x ... x ndata_x x ndata_y x dim arrays
    x1 = torch.cat(x2.shape[-2] * [x1], dim=-2)
    x2 = torch.cat(x1.shape[-3] * [x2], dim=-3)

    # Difference between x1 and x2
    diff_x = x1.view(-1, x1.shape[-1]) - x2.view(-1, x2.shape[-1])

    # Compute the hyperbolic distance
    mink_inner_prod = inner_minkowski_columns(diff_x.transpose(-1, -2), diff_x.transpose(-1, -2))
    mink_sqnorms = torch.maximum(torch.zeros_like(mink_inner_prod), mink_inner_prod)
    mink_norms = torch.sqrt(mink_sqnorms + 1e-8)  # +1e-8 for valid gradients
    distance = 2 * torch.arcsinh(.5 * mink_norms).view(x1.shape[:-1])
    # From manopt: the formula above is equivalent to d = max(0, real(acosh(-inner_minkowski_columns(x1, x2))))
    # but is numerically more accurate when distances are small.
    # When distances are large, it is better to use the acosh formula.

    if diag:
        distance = torch.diagonal(distance, 0, -2, -1)

    return distance


def inner_minkowski_columns(x, y):
    """
    This function computes the Minkowski inner product between points on the Lorentz model of the hyperbolic manifold.

    Parameters
    ----------
    :param x: points on the Lorentz model                                N1 x dim or b1 x ... x bk x N1 x dim
    :param y: points on the Lorentz model                                N2 x dim or b1 x ... x bk x N2 x dim

    Return
    ------
    :return: hyperbolic inner product between x and y

    """
    return -x[0]*y[0] + torch.sum(x[1:]*y[1:], dim=0)


def poincare_to_lorentz(x):
    """
    This functions maps data from the Poincare disk to the Lorentz model.

    Parameters
    ----------
    :param x: point on the Poincaré disk

    Returns
    -------
    :return: point on the Lorentz model
    """
    first_coord = 1 + torch.pow(torch.linalg.norm(x, dim=-1), 2)
    lorentz_point = torch.hstack((first_coord.unsqueeze(-1), 2*x)) / (1 - torch.pow(torch.linalg.norm(x, dim=-1), 2)).unsqueeze(-1)
    return lorentz_point


def lorentz_to_poincare(x, radius=1):
    """
    This functions maps data from the Lorentz model to the Poincare disk.

    Parameters
    ----------
    :param x: point on the Lorentz model            N+1 x dim

    Returns
    -------
    :return: point on the Poincaré disk             N x dim
    """
    poincare_x = radius * x[..., 1:] / (x[..., 0].unsqueeze(-1) + radius)
    return poincare_x


def lorentz_geodesic(x, y, nb_points=20):
    """
    This function computes the geodesic from x to y.

    Parameters
    ----------
    :param x: point on the Lorentz model
    :param y: point on the Lorentz model

    Return
    ------
    :return: equally-distributed points along the geodesic from x to y
    """
    if x.ndim == 1:
        x = x[None]
    if y.ndim == 1:
        y = y[None]

    u = log_map(y, x)
    t = torch.linspace(0., 1., nb_points)[:, None]

    geodesic_points = exp_map(u * t, x)

    return geodesic_points


# Functions below were copied from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py
def exp_map(u: Tensor, at_point: Tensor, radius: Tensor = torch.tensor(1.0)) -> Tensor:
    """
    This function maps a vector x lying on the tangent space of at_point into the manifold.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param u: vector in the tangent space
    :param at_point: basis point of the tangent space

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: x: point on the manifold
    """
    u_norm = lorentz_norm(u, keepdim=True) / radius
    u_normed = u / u_norm
    x = cosh(u_norm) * at_point + sinh(u_norm) * u_normed
    assert torch.isfinite(x).all()
    return x


def exp_map_mu0(x: Tensor, radius: Tensor = torch.tensor(1.0)) -> Tensor:
    """
    This function maps a vector x lying on the tangent space at the origin into the manifold.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param u: vector in the tangent space

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: x: point on the manifold
    """
    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
    x = x[..., 1:]
    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
    x_normed = F.normalize(x, p=2, dim=-1) * radius
    ret = torch.cat((cosh(x_norm) * radius, sinh(x_norm) * x_normed), dim=-1)
    assert torch.isfinite(ret).all()
    return ret


def log_map(x: Tensor, at_point: Tensor, radius: Tensor = torch.tensor(1.0)) -> Tensor:
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: point on the manifold
    :param at_point: basis point of the tangent space where x will be mapped

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: u: vector in the tangent space of x0
    """
    alpha = -lorentz_product(at_point, x, keepdim=True) / (radius**2)
    coef = acosh(alpha) / sqrt(alpha**2 - 1)
    u = coef * (x - alpha * at_point)
    return u


def log_map_mu0(x: Tensor, radius: Tensor = torch.tensor(1.0)) -> Tensor:
    """
    This functions maps a point lying on the manifold into the tangent space of the origin on the manifold.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: point on the manifold

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: u: vector in the tangent space of the origin
    """
    alpha = x[..., 0:1] / radius
    coef = acosh(alpha) / sqrt(alpha**2 - 1.)
    diff = torch.cat((x[..., 0:1] - alpha * radius, x[..., 1:]), dim=-1)
    return coef * diff


def parallel_transport_mu0(x: Tensor, dst: Tensor, radius: Tensor = torch.tensor(1.0)) -> Tensor:
    """
    This function parallel-transports a vector x from the tangent space at the origin onto the tangent space at dst.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: vector on the tangent space of the origin
    :param dst: point on the manifold

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: parallel-transported vector on the tangent space of dst
    """
    # PT_{mu0 -> dst}(x) = x + <dst, x>_L / (R^2 - <mu0, dst>_L) * (mu0+dst)
    denom = radius * (radius + dst[..., 0:1])  # lorentz_product(mu0, dst, keepdim=True) which is -dst[0]*radius
    lp = lorentz_product(dst, x, keepdim=True)
    coef = lp / denom
    right = torch.cat((dst[..., 0:1] + radius, dst[..., 1:]), dim=-1)  # mu0 + dst
    return x + coef * right


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor = torch.tensor(1.0)) -> Tensor:
    """
    This function parallel-transports a vector x from the tangent space at src onto the tangent space at the origin.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: vector on the tangent space of the origin
    :param src: point on the manifold

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: parallel-transported vector on the tangent space of the origin
    """
    # PT_{src -> mu0}(x) = x + <mu0, x>_L / (R^2 - <src, mu0>_L) * (src+mu0)
    denom = (radius + src[..., 0:1])  # lorentz_product(src, mu0, keepdim=True) which is -src[0]*radius
    lp = -x[..., 0:1]  # lorentz_product(mu0, x, keepdim=True) which is -x[0]*radius
    # coef = (lp * radius) / (radius * denom)
    coef = lp / denom
    right = torch.cat((src[..., 0:1] + radius, src[..., 1:]), dim=-1)  # mu0 + src
    return x + coef * right


def lorentz_norm(x: Tensor, **kwargs: Any) -> Tensor:
    """
    This function computes the norm of a vector in a tangent space of the hyperbolic manifold.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: tangent vector

    Return
    ------
    :return: norm of x
    """
    product = lorentz_product(x, x, **kwargs)
    ret = sqrt(product)
    return ret


def lorentz_product(x: Tensor, y: Tensor, keepdim: bool = False, dim: int = -1) -> Tensor:
    """
    This function computes the inner product between vectors in a tangent space of the hyperbolic manifold.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: tangent vector
    :param y: tangent vector

    Return
    ------
    :return: inner product between x and y
    """
    try:
        m = x * y
    except:
        m = torch.mm(x, y)
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret


def sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor = torch.tensor(1.0)) -> \
        Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """
    This function parallel-transports a vector in the tangent space of mu0 to the tangent space of at_point and project
    it on the manifold using the exponential map.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: vector on the tangent space of the origin
    :param at_point: point on the manifold

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: sample on the manifold as exp_{at_point}(PT_mu0->at_point(x))
    """
    max_clamp_norm = 40
    x_expanded = expand_proj_dims(x)
    pt = parallel_transport_mu0(x_expanded, dst=at_point, radius=radius)
    pt = clamp(pt, min=-max_clamp_norm, max=max_clamp_norm)
    x_proj = exp_map(pt, at_point=at_point, radius=radius)
    x_proj = clamp(x_proj, min=-max_clamp_norm, max=max_clamp_norm)
    return x_proj, (pt, x)


def inverse_sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor = torch.tensor(1.0)) -> \
        Tuple[Tensor, Tensor]:
    """
    This function projects a sample on the manifold to the tangent space of at_point and transports it to the tangent
    space of the origin.
    This function was adapted from: https://github.com/joeybose/HyperbolicNF/blob/master/utils/hyperbolics.py

    Parameters
    ----------
    :param x: point on the manifold
    :param at_point: point on the manifold

    Optional parameters
    -------------------
    :param radius: radius of the hyperbolic space

    Returns
    -------
    :return: sample on the tangent space of the origin PT_at_point->mu(log_atpoint(x))
    """
    max_clamp_norm = 40
    unmapped = log_map(x, at_point=at_point, radius=radius)
    unmapped = clamp(unmapped, min=-max_clamp_norm, max=max_clamp_norm)
    unpt = inverse_parallel_transport_mu0(unmapped, src=at_point, radius=radius)
    unpt = clamp(unpt, min=-max_clamp_norm, max=max_clamp_norm)
    return unmapped, unpt[..., 1:]
