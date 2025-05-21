import torch
from HyperbolicEmbeddings.hyperbolic_manifold.manifold import Manifold


def curve_energy(X_new: torch.Tensor, G: torch.Tensor, manifold: Manifold, reduce="sum") -> torch.Tensor:
    """
    Let x_1, ..., x_M be a curve then this function computes the curve energy
    E = \sum_{i=1}^{M-1} v_i@G_{x_i}@v_i for v_i=Log_{x_i}(x_{i+1})

    Parameters
    -
    X: [M, D_x]        M manifold points interpreted as samples along the curve
    G: [M-1, D_x, D_x] metric tensor at each curve point but the last
    reduce: "sum" | "mean" | None

    Returns
    -
    E: [1]  scalar curve energy
    """
    V = manifold.log(X_new[:-1], X_new[1:])  # [M-1, D_x, 1]
    if reduce == "sum":
        return torch.bmm(torch.bmm(V.unsqueeze(1), G), V.unsqueeze(2)).sum()
    elif reduce == "mean":
        return torch.bmm(torch.bmm(V.unsqueeze(1), G), V.unsqueeze(2)).mean()
    return torch.bmm(torch.bmm(V.unsqueeze(1), G), V.unsqueeze(2)).squeeze()


def curve_length(X_new: torch.Tensor, G: torch.Tensor, manifold: Manifold, reduce="sum") -> torch.Tensor:
    """
    Let x_1, ..., x_M be a curve then this function computes the curve length
    L = \sum_{i=1}^{M-1} sqrt(v_i@G_{x_i}@v_i for v_i=Log_{x_i}(x_{i+1}))

    Parameters
    -
    X: [M, D_x]        M manifold points interpreted as samples along the curve
    G: [M-1, D_x, D_x] metric tensor at each curve point but the last
    reduce: "sum" | "mean" | None

    Returns
    -
    L: [1]  scalar curve length
    """
    V = manifold.log(X_new[:-1], X_new[1:])  # [M-1, D_x, 1]
    if reduce == "sum":
        return torch.sqrt(torch.bmm(torch.bmm(V.unsqueeze(1), G), V.unsqueeze(2))).sum()
    elif reduce == "mean":
        return torch.sqrt(torch.bmm(torch.bmm(V.unsqueeze(1), G), V.unsqueeze(2))).mean()
    return torch.sqrt(torch.bmm(torch.bmm(V.unsqueeze(1), G), V.unsqueeze(2))).squeeze()


def spline_energy(X: torch.Tensor, manifold: Manifold) -> torch.Tensor:
    """
    Let x_1, ..., x_M be a curve then this function computes the spline energy
    E = \sum_{i=2}^{M-1} d(x_i, xm_i)^2 for xm_i=geodesic_midpoint(x_{i-1}, x_{i+1})

    Parameters
    -
    X: [M, D_x]        M manifold points interpreted as samples along the curve

    Returns
    -
    E: [1]  scalar spline energy
    """
    direction_vectors = manifold.log(X[:-2], X[2:])  # [M-2, D_x]
    midpoints = manifold.exp(X[:-2], 0.5*direction_vectors)  # [M-2, D_x]
    distances = manifold.distance_squared(X[1:-1], midpoints, diag=True)  # [M-2]
    return distances.sum()