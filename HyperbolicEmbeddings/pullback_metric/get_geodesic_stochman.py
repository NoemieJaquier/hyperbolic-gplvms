import torch
import pickle
import os
from pathlib import Path
from HyperbolicEmbeddings.plot_utils.plots_pullback import sample_2D_grid
from stochman.curves import CubicSpline
from stochman.geodesic import geodesic_minimizing_energy
from stochman.discretized_manifold import DiscretizedManifold
from stochman.manifold import Manifold


class DiscreteGPLVMManifoldWrapper(DiscretizedManifold):
    def __init__(self, grid, metric_function):
        super().__init__()
        self.grid = grid
        self.metric_function = metric_function

    def fit(self, batch_size: int = 4):
        return super().fit(self, self.grid, batch_size=batch_size)

    def metric(self, c: torch.Tensor, return_deriv=False):
        return self.metric_function(c)
    
    def save_discretized_manifold(self, filepath: Path):
        obj_to_save = {
            "G": self.G,
            "grid": self.grid,
            "grid_size": self.grid_size,
            "__metric__": self.__metric__,
        }

        with open(filepath, "wb") as fp:
            pickle.dump(obj_to_save, fp)

class GPLVMManifoldWrapper(Manifold):
    def __init__(self, metric_function):
        super().__init__()
        self.metric_function = metric_function

    def metric(self, c: torch.Tensor, return_deriv=False):
        return self.metric_function(c)


def get_geodesic_on_discrete_learned_manifold(x_new_init: torch.Tensor, x: torch.Tensor, metric_function, n_eval=20,
                                              grid_size=32, num_nodes_to_optimize=20, discretized_path=None) -> torch.Tensor:
    _, limits = sample_2D_grid(x, grid_size)
    discrete_manifold = DiscreteGPLVMManifoldWrapper([torch.linspace(*limits, grid_size),
                                                      torch.linspace(*limits, grid_size), ], metric_function)
    discrete_manifold.fit()
    if discretized_path:
        if os.path.isfile(discretized_path):
            discrete_manifold = discrete_manifold.from_path(discrete_manifold, discretized_path)
        else:
            discrete_manifold.save_discretized_manifold(discretized_path)

    geodesic, _ = discrete_manifold.connecting_geodesic(x_new_init[0], x_new_init[-1],
                                                        CubicSpline(x_new_init[0], x_new_init[-1],
                                                                    num_nodes=num_nodes_to_optimize))
    time = torch.linspace(0, 1, n_eval)
    return geodesic(time).detach()


def get_geodesic_on_learned_manifold(x_new_init: torch.Tensor, metric_function, n_eval=20, num_nodes_to_optimize=20) \
        -> torch.Tensor:
    manifold = GPLVMManifoldWrapper(metric_function)
    # start, end = x_new_init[0].unsqueeze(0), x_new_init[-1].unsqueeze(0)
    geodesic = CubicSpline(x_new_init[0], x_new_init[-1], num_nodes=num_nodes_to_optimize)
    # geodesic = CubicSpline(start, end)
    geodesic_minimizing_energy(geodesic, manifold, max_iter=2000)
    time = torch.linspace(0, 1, n_eval)
    return geodesic(time).detach()
