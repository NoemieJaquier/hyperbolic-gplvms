from typing import Callable, Optional, Tuple
from itertools import product
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import numpy as np


def get_3D_unit_sphere_points(n_points=500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -
    x: [n_points, n_points]
    y: [n_points, n_points]
    z: [n_points, n_points]
    """
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def sample_2D_grid(X: torch.Tensor, grid_size: int = 128, padding_percent=0.05) -> Tuple[torch.Tensor, Tuple[float, float]]:
    """
    computes the smallest square grid that contains all the latent points in X plus some padding.

    Parameters
    -
    X: torch.shape([N, 2])  N latent points

    Returns
    -
    X_grid: torch.shape([grid_size*grid_size, 2]) regularly sampled latent points in a square grid
    [
     x_1,             x_2,           , ... , x_grid_size, \\
     x_(grid_size+1), x_(grid_size+2), ... , x_(2*grid_size),
     ...
    ]
    limits: [min, max]  the minimum and maximum value of any dimension of X plus padding
    """
    min_in_latent = X.min().item()
    max_in_latent = X.max().item()
    padding = (max_in_latent - min_in_latent) * padding_percent
    limits = (min_in_latent - padding, max_in_latent + padding)
    return torch.Tensor([[x, y] for x, y in product(torch.linspace(*limits, grid_size),
                                                    reversed(torch.linspace(*limits, grid_size)))]), limits


def from_tensor_to_image(values: torch.Tensor, limits: Tuple[float], grid_size: int) -> np.ndarray:
    """
    Grabs a tensor of size (grid_size*grid_size) and transforms
    it into an image of size (grid_size, grid_size) according to
    the limits provided.

    Parameters
    ----------

    - values (torch.Tensor of shape (grid_size ** 2,)): the tensor
      to convert into an image.

    - limits (Tuple[float]): the limits of the gridsize (min, max).

    - grid_size (int): the size of the grid/image.

    Returns
    -------

    - image (np.ndarray of shape (grid_size, grid_size)): the image
      version of the values provided, assuming they are in the same
      order as the grid below.

    """
    grid = torch.Tensor([[x, y] for x, y in product(torch.linspace(*limits, grid_size),
                                                    reversed(torch.linspace(*limits, grid_size)))])
    z_to_values_map = {(z1.item(), z2.item()): values[i].item() for i, (z1, z2) in enumerate(grid)}
    image_of_values = np.zeros((grid_size, grid_size))
    for j, x in enumerate(torch.linspace(*limits, grid_size)):
        for i, y in enumerate(reversed(torch.linspace(*limits, grid_size))):
            image_of_values[i, j] = z_to_values_map[x.item(), y.item()]
    return image_of_values


def plot_3D_unit_sphere(ax: Axes3D) -> None:
    x, y, z = get_3D_unit_sphere_points()
    ax.plot_surface(x, y, z, color=(0.7, 0.7, 0.7, 0.1))


def plot_2D_unit_circle(ax: Axes) -> None:
    circle = patches.Circle((0, 0), 1, edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    padding = 0.1
    ax.set_xlim(-1 - padding, 1 + padding)
    ax.set_ylim(-1 - padding, 1 + padding)
    ax.set_aspect('equal')


def plot_2D_metric_volume_on_grid(ax: Axes, X: torch.Tensor, get_metric_volume: Callable[[torch.Tensor], torch.Tensor], grid_size=128, restrict_to_unit_circle=False) -> None:
    """
    plots the metric volume at each point of a 2D square grid that contains the given input points.

    Parameters
    -
    X: [M, 2]   two dimensional points that should be contained in the grid
    get_metric_volume: (X_grid: [grid_size**2, 2]) -> [grid_size**2] computes the metric volume for the given inputs
    """
    if restrict_to_unit_circle:
        radius = 0.95
        X_grid_2D, limits = sample_2D_grid(torch.tensor([[-radius, -radius], [radius, radius]]), grid_size, padding_percent=0)
        mask_unit_circle = torch.norm(X_grid_2D, dim=1) < radius
        img_tensor = torch.zeros(grid_size**2)
        img_tensor[mask_unit_circle] = get_metric_volume(X_grid_2D[mask_unit_circle])
    else:
        X_grid_2D, limits = sample_2D_grid(X, grid_size, padding_percent=0.05)
        img_tensor = get_metric_volume(X_grid_2D)

    img_tensor = torch.clamp(img_tensor, min=0, max=100)

    img = from_tensor_to_image(img_tensor, limits, grid_size=grid_size)
    volume_plot = ax.imshow(img, extent=[*limits, *limits], interpolation="bicubic")
    plt.colorbar(volume_plot, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')


def get_bounding_box_points(X_2D: torch.Tensor, grid_size=120, padding_percent=0.05) -> torch.Tensor:
    """
    Parameters
    X_2D: [N, 2]  2D points

    Returns
    -
    X_grid_2d: [grid_size**2, 2]   2D grid points around the given point at which the metric got evaluated
    metric_volumes: [grid_size**2] metric volume evaluated at each grid point
    """
    (x_min, y_min), _ = X_2D.min(dim=0)
    (x_max, y_max), _ = X_2D.max(dim=0)
    padding_x = (x_max - x_min) * padding_percent / 2
    padding_y = (y_max - y_min) * padding_percent / 2
    x_values = torch.linspace(x_min - padding_x, x_max + padding_x, steps=grid_size)
    y_values = torch.linspace(y_min - padding_y, y_max + padding_y, steps=grid_size)
    xx, yy = torch.meshgrid(x_values, y_values, indexing='xy')
    return torch.stack([xx, yy], dim=-1).reshape(grid_size**2, 2)


def get_pullback_metric_volume_around_point(point: torch.Tensor, get_metric_volume: Callable[[torch.Tensor], torch.Tensor], grid_size: Optional[int] = None, distance_between_points: Optional[float] = None, radius: Optional[float] = None, restrict_to_unit_circle=False) -> torch.Tensor:
    """
    Returns
    -
    X_grid_2d: [grid_size**2, 2]   2D grid points around the given point at which the metric got evaluated
    metric_volumes: [grid_size**2] metric volume evaluated at each grid point
    """
    if radius is None:
        radius = grid_size * distance_between_points / 2
    elif grid_size is None:
        grid_size = int(2 * radius / distance_between_points)

    x_values = torch.linspace(point[0] - radius, point[0] + radius, steps=grid_size)
    y_values = torch.linspace(point[1] - radius, point[1] + radius, steps=grid_size)
    xx, yy = torch.meshgrid(x_values, y_values, indexing='xy')
    X_grid_2D = torch.stack([xx, yy], dim=-1).reshape(grid_size**2, 2)  # [grid_size**2, 2]

    if restrict_to_unit_circle:
        evaluation_radius_threshold = 0.99
        mask_unit_circle = torch.norm(X_grid_2D, dim=1) < evaluation_radius_threshold
        metric_volumes = torch.zeros(grid_size**2)
        metric_volumes[mask_unit_circle] = get_metric_volume(X_grid_2D[mask_unit_circle])  # [grid_size**2]
        metric_volumes[~mask_unit_circle] = torch.nan
    else:
        metric_volumes = get_metric_volume(X_grid_2D)  # [grid_size**2]
    return X_grid_2D, metric_volumes


def plot_2D_metric_volume_around_point(ax: Axes, X_grid_2D: torch.Tensor, metric_volumes: torch.Tensor, color_threshold=None, hide_frame=False, fontsize=14, show_cbar=True) -> None:
    """
    plots the metric volume at each point of a 2D square grid that contains the given input points.

    Define two of those three parameters to define the grid

    Parameters
    -
    X_grid_2D: [grid_size**2, 2]       2D grid points
    metric_volumes: [grid_size**2]     metric volume to plot or nan at indices that should be transparent
    radius       distance from the center point to the grid border
    """
    grid_size = int(np.sqrt(X_grid_2D.shape[0]))
    img_tensor = metric_volumes.reshape(grid_size, grid_size)  # [grid_size, grid_size]

    img_tensor = torch.flip(img_tensor, dims=[0])
    nan_mask = torch.isnan(img_tensor)
    alpha_channel = torch.ones(grid_size, grid_size)  # [grid_size, grid_size]
    alpha_channel[nan_mask] = 0  # [grid_size, grid_size]
    img_tensor[nan_mask] = img_tensor[~nan_mask].max()
    if color_threshold is None:
        color_threshold = img_tensor.max().item()
    img_tensor_clipped = torch.clamp(img_tensor, max=color_threshold)

    viridis_colormap = cm.get_cmap('viridis')
    rgba_image = viridis_colormap(img_tensor_clipped / img_tensor_clipped.max())
    rgba_image[:, :, -1] = alpha_channel

    start, end = X_grid_2D[0], X_grid_2D[-1]
    ax.imshow(rgba_image, extent=[start[0], end[0], start[1], end[1]], interpolation="bicubic")

    norm = mcolors.Normalize(vmin=0, vmax=min(img_tensor.max(), color_threshold))
    sm = plt.cm.ScalarMappable(cmap=viridis_colormap, norm=norm)
    sm.set_array([])  # Only required to avoid warnings when calling plt.colorbar

    if show_cbar:
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        tick_labels = [f'{int(tick)}' for tick in cbar.get_ticks()]
        if img_tensor.max().item() > color_threshold:
            tick_labels[-1] = f'>{int(color_threshold)}'
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    radius = 1
    ax.set_xticks([-radius, 0, radius])
    ax.set_yticks([-radius, 0, radius])

    # ax.set_xlabel('x_0')
    # ax.set_ylabel('x_1')

    if hide_frame:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    return cbar.ax if show_cbar else None



