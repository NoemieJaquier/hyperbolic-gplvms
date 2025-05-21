import torch
from typing import Literal
from HyperbolicEmbeddings.hyperbolic_manifold.math_ops_hyperbolic import acosh_squared


def origin(dim: int) -> torch.Tensor:
    """
    o = [1, 0, ..., 0]

    Parameters
    -
    dim: dimension of the lorentz manifold

    Returns
    -
    o: [dim+1] origin point on the lorentz manifold
    """
    return torch.cat([torch.ones(1), torch.zeros(dim)])


def metric_tensor(X: torch.Tensor) -> torch.Tensor:
    """
    G = diag(-1, 1, ..., 1)

    Parameters
    -
    X: [N, D]  N Lorentz points

    Returns
    G: [N, D, D] N Lorentz metric tensors
    """
    N, D = X.shape
    G = torch.diag(torch.cat([torch.tensor([-1.]), torch.ones(D-1)]))  # [D, D]
    return G.repeat(N, 1, 1)  # [N, D, D]


def matrix_volume(matrices: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    matrices: [M, D_x, D_x] tangent space matrices at M different points x

    Returns
    -
    volumes: [M]  square root of the determinant of each tangent space matrix
    """
    def volume(G: torch.Tensor) -> float:
        # Due to the extra dimension in the lorentz model one eigenvalue is zero.
        # This eigenvalue needs to be excluded to get useful volumes
        eigvals: torch.Tensor = torch.linalg.eigvals(G).real
        min_index = torch.argmin(eigvals)
        non_zero_eigvals = torch.cat([eigvals[:min_index], eigvals[min_index+1:]])
        return non_zero_eigvals.prod().sqrt().item()
    return torch.Tensor([volume(G) for G in matrices])


def inner_product(X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
    """
    <x,y>_L = -x_0y_0 + x_1y_1 + ... + x_Dy_D

    Parameters
    -
    X: [N, D]  N Lorentz or tangent space points
    Y: [M, D]  M Lorentz or tangent space points
    diag       True requires that N must equal M to return the diagonal of the inner product matrix

    Returns
    -
    inner_product: [N, M] if not diag else torch.shape([N])
    """
    G = metric_tensor(X)  # [N, D, D]
    if not diag:
        return X @ G[0] @ Y.T  # [N, M] = [N, D] x [D, D] x [D, M]
    assert X.shape[0] == Y.shape[0]
    return torch.bmm(torch.bmm(X.unsqueeze(1), G), Y.unsqueeze(2)).squeeze(1, 2)  # [N] = [N, 1, D] x [N, D, D] x [N, D, 1]


def norm(U: torch.Tensor) -> torch.Tensor:
    """
    u_norm = sqrt(<u, u>_L)

    Parameters
    -
    U: [N, D]  N tangent space vectors

    Returns
    -
    u_norm: [N]  norm of each tangent space vector
    """
    return torch.sqrt(inner_product(U, U, diag=True))


def distance(X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
    """
    d(x, y) = acosh(-<x, y>_L)
    Note: If you need to compute the squared distance use distance_squared instead which produces correct results for derivatives for X=Y.

    Parameters
    -
    X: [N, D]  N Lorentz points
    Y: [M, D]  M Lorentz points

    Returns
    -
    distance: [N, M] if not diag else [N]
    """
    distance = torch.acosh(-inner_product(X, Y, diag))
    return torch.where(torch.isnan(distance), 0, distance)


def distance_squared(X: torch.Tensor, Y: torch.Tensor, diag=False) -> torch.Tensor:
    """
    d(x, y)^2 = acosh(-<x, y>_L)^2

    Parameters
    -
    X: [N, D]  N Lorentz points
    Y: [M, D]  M Lorentz points

    Returns
    -
    distance_squared: [N, M] if not diag else [N]
    """
    return acosh_squared(-inner_product(X, Y, diag))


def exp(X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, D]  points on the Lorentz model
    U: [N, D]  points on the tangent space at X

    Returns
    -
    exp_x_u: [N, D] where exp_x_u[i] = Exp_{x_i}(u_i)
    """
    u_norm = norm(U).unsqueeze(1)  # [N, 1]
    Y = torch.cosh(u_norm)*X + torch.sinh(u_norm)*U/u_norm
    return torch.where(torch.isnan(Y), X, Y)


def log(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [M, D]  points on the Lorentz model
    Y: [M, D]  points on the Lorentz model

    Returns
    -
    log_x_y: [M, D_x] where log_x_y[i] = Log_{x_i}(y_i)
    """
    _, u, rho, s = common_ops(X, Y, operations='Gurs')
    return (rho / s).unsqueeze(-1) * (Y + u.unsqueeze(-1) * X)


def parallel_transport(X: torch.Tensor, Y: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, D]  N Lorentz points
    Y: [N, D]  N Lorentz points
    U: [N, D]  N tangent space vectors at X

    Returns
    -
    V: [N, D] where V[i] = P_{x_i->y_i}(u_i)
    """
    yu = inner_product(Y, U, diag=True).unsqueeze(1)  # [N, 1]
    xy = inner_product(X, Y, diag=True).unsqueeze(1)  # [N, 1]
    return U + yu / (1-xy) * (X + Y)  # [N, D]


def tangent_space_projection_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    P_x = G + x@x.T

    Parameters
    -
    X: [N, D]   N Lorentz points

    Returns
    -
    P: [N, D, D] N projection matrices where P[i] = P_{x_i}
    """
    G = metric_tensor(X)  # [N, 4, 4]
    XX_T = torch.bmm(X.unsqueeze(2), X.unsqueeze(1))  # [N, 4, 4] = [N, 4, 1] x [N, 1, 4]
    return G + XX_T


def proj(X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    u_proj = P_x @ u

    Parameters
    -
    X: [N, D]   N Lorentz points
    U: [N, D]   N points not necessarily on the lorentz model

    Returns
    -
    U_proj: [N, D]  points U projected onto the tangent space of X 
    """
    P_x = tangent_space_projection_matrix(X)  # [N, D, D]
    return torch.bmm(P_x, U.unsqueeze(2)).squeeze(2)  # [N, D]


def tangent_space_basis_vector_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    V_x = [P_{o->x}(e_2), P_{o->x}(e_3), ..., P_{o->x}(e_D)]

    Parameters
    -
    X: [N, D]  N Lorentz points

    Returns
    -
    V: [N, D, D-1]  N tangent space basis vector matrices where V[i] = V_{x_i}
    """
    N, D = X.shape
    V_o = torch.concatenate([torch.zeros(1, D-1), torch.eye(D-1)], dim=0).repeat(N, 1, 1)  # [N, D, D-1]
    o = origin(D-1).repeat(N, 1)  # [N, D]
    V_x = torch.zeros_like(V_o)  # [N, D, D-1]
    for i in range(D-1):
        V_x[:, :, i] = parallel_transport(o, X, V_o[:, :, i])
    return V_x


def are_valid_points(X: torch.Tensor) -> bool:
    """
    x_0 > 0 and <x, x>_L = -1

    Parameters
    -
    X: [N, D] N potential Lorentz points

    Returns
    -
    valid   True if all X are valid Lorentz points
    """
    return all(X[:, 0] > 0.) and torch.allclose(inner_product(X, X, diag=True), -torch.ones(1))


def are_valid_tangent_space_vectors(X: torch.Tensor, U: torch.Tensor) -> bool:
    """
    <x, u>_L = 0

    Parameters
    -
    X: [N, D]   valid Lorentz points
    U: [N, D]   potential tangent space vectors at X

    Returns
    -
    valid   True if all U are valid tangent space vectors at X
    """
    return torch.allclose(inner_product(X, U, diag=True), torch.zeros(1))


def random_points(N: int, dim: int) -> torch.Tensor:
    """
    Returns
    -
    X: [N, dim+1]  N random Lorentz points
    """
    poincare_points = []
    while len(poincare_points) < N:
        candidate = 2*torch.rand(dim)-1  # [dim]
        if torch.norm(candidate) < 1:
            poincare_points.append(candidate)
    random_poincare = torch.stack(poincare_points)  # [N, dim]
    return from_poincare(random_poincare)  # [N, dim+1]


def random_tangent_space_vectors(X: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    X: [N, D]  N Lorentz points

    Returns
    -
    V: [N, D]  N random tangent space vectors at X where V[i] in tangent space of X[i]
    """
    N, D = X.shape
    V_o = torch.cat([torch.zeros(N, 1), torch.randn(N, D-1)], dim=1)  # [N, D]
    o = origin(D-1).repeat(N, 1)  # [N, D]
    return parallel_transport(o, X, V_o)  # [N, D]


def from_poincare(X_poincare: torch.Tensor) -> torch.Tensor:
    """
    map points from the Poincare model to the Lorentz model

    Parameters
    -
    X_poincare: [N, D] N Poincare points

    Returns
    -
    X_lorentz: [N, D+1] N Lorentz points
    """
    X_normsquared = torch.sum(X_poincare**2, dim=1, keepdim=True)  # [N, 1]
    return torch.cat([1 + X_normsquared, 2*X_poincare], dim=1) / (1 - X_normsquared)  # [N, D+1]


def common_ops(X: torch.Tensor, Y: torch.Tensor, operations: Literal['Gurs']) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    convenience method to perform common operations such as
    (G)metric_tensor, (u)inner_product, (r)distance, (s)sqrt(u**2-1)

    Parameters
    -
    X: [N, 4]  N points on the Lorentz model
    Y: [N, 4]  N points on the Lorentz model
    operations the operations to perform. Note: I only ever needed Gurs. More operations might be added in the future.

    Returns
    -
    G: [M, 4, 4] metric tensor
    u: [M] diagonal inner product
    r: [M] diagonal distance
    s: [M] sqrt(u**2-1)
    """
    N, D = X.shape
    G = metric_tensor(X)  # [N, 4, 4]
    u = inner_product(X, Y, diag=True)  # [N]
    acosh_u = torch.acosh(-u)  # [N]
    sqrt_u = torch.sqrt(u**2-1)  # [N]
    rho = torch.where(torch.isnan(acosh_u), 0, acosh_u)  # same as distance(X, Y, diag=True)
    s = torch.where(torch.isnan(sqrt_u), 0, sqrt_u)
    return G, u, rho, s


def geodesic(x, y, nb_points=20):
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

    u = log(y, x)
    t = torch.linspace(0., 1., nb_points)[:, None]

    geodesic_points = exp(u * t, x)

    return geodesic_points
