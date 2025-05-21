import torch


def metric_tensor(X: torch.Tensor) -> torch.Tensor:
    """
    G = (2 / 1-x_norm**2)**2 * I

    Parameters
    -
    X: [N, D]  N Poincare points

    Returns
    G: [N, D, D] N Lorentz metric tensors
    """
    N, D = X.shape
    X_norm = torch.norm(X, dim=1)  # [N]
    poincare_factor = 4 / (1 - X_norm**2)**2  # [N]
    I = torch.eye(D).repeat(N, 1, 1)  # [N, D, D]
    return poincare_factor[:, None, None] * I  # [N, D, D]


def outer_product(x1: torch.Tensor, x2: torch.Tensor,  diag=False) -> torch.Tensor:
    """
    Computes the Poincare outer product between points in the hyperbolic manifold (Poincaré ball representation) as
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
    outer_product = 0.5 * torch.log((1. - torch.norm(x1, dim=-1)**2) / torch.norm(diff_x, dim=-1)**2)

    return outer_product


def are_valid_points(X: torch.Tensor) -> torch.Tensor:
    """
    norm(x) < 1

    Parameters
    -
    X: [N, D] N potential Poincare points

    Returns
    -
    valid   True if all X are valid Poincare points
    """
    X_norm = torch.norm(X, dim=1)  # [N]
    return all(X_norm < 1)


def random_points(N: int, dim: int) -> torch.Tensor:
    """
    Returns
    -
    X: [N, dim]  N random Poincare points
    """
    poincare_points = []
    while len(poincare_points) < N:
        candidate = 2*torch.rand(dim)-1  # [dim]
        if torch.norm(candidate) < 1:
            poincare_points.append(candidate)
    return torch.stack(poincare_points)  # [N, dim]


def from_lorentz(X_lorentz: torch.Tensor) -> torch.Tensor:
    """
    map points from the Lorentz model to the Poincare model

    Parameters
    -
    X_lorentz: [N, D]   N Lorentz points

    Returns
    -
    X_poincare: [N, D-1] N Poincare points
    """
    return X_lorentz[:, 1:] / (1 + X_lorentz[:, 0, None])


