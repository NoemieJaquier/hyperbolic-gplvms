import torch
from geoopt import ManifoldParameter
from geoopt.optim.radam import RiemannianAdam
from typing import Callable
from HyperbolicEmbeddings.pullback_metric.curves import spline_energy
from HyperbolicEmbeddings.hyperbolic_manifold.manifold import Manifold
from HyperbolicEmbeddings.pullback_metric.curves import curve_energy


def get_geodesic(start: torch.Tensor, end: torch.Tensor, N: int, manifold: Manifold) -> torch.Tensor:
    """
    Parameters
    -
    start: [D] point on the manifold
    end:   [D] point on the manifold
    N          number of points that should be in the geodesic

    Returns
    -
    geodesic: [N, D] N evenly spaced points along the geodesic from start to end
    """
    t = torch.linspace(0, 1, N).unsqueeze(1)  # [N, 1]
    start, end = start.unsqueeze(0), end.unsqueeze(0)  # [1, D]
    direction = manifold.log(start, end)  # [1, D]
    return manifold.exp(start.repeat(N, 1), t*direction)  # [N, D]


def optimize_geodesic_on_learned_manifold(X_new_init: torch.Tensor, metric_function: Callable[[torch.Tensor], torch.Tensor], manifold: Manifold, n_steps=300, learning_rate=1e-3, spline_energy_weight=1.) -> torch.Tensor:
    """
    computes lorentz geodesics by directly optimizing the curve points.

    Parameters
    -
    X_new_init: torch.Tensor([M, 4])  initial curve points on the lorentz manifold
    metric_function: [M,4] -> [M,4,4] returns the metric tensor at each input point x
                        X |-> G(X)

    Returns
    -
    X_new_init: torch.Tensor([M, 4])  optimized curve points on the lorentz manifold
    """
    X_new = ManifoldParameter(X_new_init, manifold.geoopt)
    optim = RiemannianAdam([X_new], lr=learning_rate)
    curve_energies, weighted_spline_energies = [], []

    for step in range(n_steps):
        optim.zero_grad()
        G_pullback = metric_function(X_new)  # [M-1, D_x, D_x], [M-1, D_x, D_x, D_x]
        E_curve = curve_energy(X_new, G_pullback[:-1], manifold)  # [1]
        E_spline = spline_energy_weight * spline_energy(X_new, manifold)  # [1]
        loss = E_curve + E_spline
        loss.backward()
        diff_x_E = X_new.grad

        diff_x_E[0] = 0  # keep first point fixed
        diff_x_E[-1] = 0  # keep last point fixed
        X_new.grad = diff_x_E  # [M, D_x]
        optim.step()

        print(f"Iteration {step}: {E_curve} {E_spline}")
        curve_energies.append(E_curve.item())
        weighted_spline_energies.append(E_spline.item())

    # import matplotlib.pyplot as plt
    # _, ax = plt.subplots()
    # ax.set_title("Curve Energy during training")
    # ax.set_xlabel("iteration")
    # ax.plot(weighted_spline_energies, label=f"weighted spline energy")
    # ax.plot(curve_energies, label="curve energy")
    # ax.legend()
    return X_new.detach()