import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as pltc
import matplotlib.patches as mpatches



def plot_distance_matrix(
    distance_matrix,
    max_distance=None,
    min_distance=0.,
    save_path=None,
    x_colors=None,
    show=True,
    box_colors=None
):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    plot = plt.imshow(distance_matrix, vmin=min_distance, vmax=max_distance, cmap='bone')
    if max_distance:
        ticks = list(range(0, int(max_distance+1)))
        # ticks = list(range(0, int(max_distance+1))) + [max_distance]
        cbar = plt.colorbar(plot, ticks=ticks, fraction=0.046, pad=0.04)
    else:
        cbar = plt.colorbar(plot, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    plt.axis('off')

    # Build blocks to highlight
    final_blocks = None
    if box_colors is not None:
        blocks = []
        for i, color in enumerate(box_colors):
            if color == "red":
                if beginning is None:
                    beginning = i
                end = i
                blocks.append((beginning, end))
            else:
                beginning, end = None, None

        final_blocks = []
        for current_beginning in [x for x, _ in blocks]:
            current_end = -1
            for beg, end in blocks:
                if beg == current_beginning:
                    if end > current_end:
                        current_end = end
            final_blocks.append((current_beginning, current_end))

        final_blocks = list(set(final_blocks))

    if x_colors:
        if box_colors is None:
            box_colors = ["black"] * len(x_colors)

        N = distance_matrix.shape[0]
        # Add border
        rect = mpatches.Rectangle((-0.5, -0.5), N, N, fill=False, edgecolor="black")
        ax.add_patch(rect)

        # Add classes colors on the sides
        width = 0.05*N
        for n in range(N):
            # The colors of each block itself.
            rect = mpatches.Rectangle(
                (-0.5 + n, -0.5 - width),
                1.0,
                width,
                facecolor=x_colors[n],
                edgecolor="black",
            )
            ax.add_patch(rect)
            rect = mpatches.Rectangle(
                (-0.5 - width, -0.5 + n),
                width,
                1.0,
                facecolor=x_colors[n],
                edgecolor="black",
            )
            ax.add_patch(rect)
            if box_colors[n] == "red":
                # Half-wide full box of red. (?)
                rect = mpatches.Rectangle(
                    (-0.5 + n, -0.5 - width),
                    1.0,
                    width / 2,
                    facecolor="red",
                    edgecolor="red",
                )
                ax.add_patch(rect)
                rect = mpatches.Rectangle(
                    (-0.5 - width, -0.5 + n),
                    width / 2,
                    1.0,
                    facecolor="red",
                    edgecolor="red",
                )
                ax.add_patch(rect)

        # Draw blocks
        if final_blocks is not None:
            for beginning, end in final_blocks:
                # Plot a square beginning and ending in block
                box_width = end - beginning + 1  # I think
                rect = mpatches.Rectangle(
                    (-0.5 + beginning, -0.5 - width),
                    box_width,
                    N + width,
                    # facecolor="red",
                    fill=False,
                    edgecolor="red",
                    linewidth=2.
                )
                ax.add_patch(rect)
                rect = mpatches.Rectangle(
                    (-0.5 - width, -0.5 + beginning),
                    N + width,
                    box_width,
                    # facecolor="red",
                    fill=False,
                    edgecolor="red",
                    linewidth=2.
                )
                ax.add_patch(rect)
            # plt.plot([-0.5 + n, 0.5 + n], [-1.5, -1.5], c=x_colors[n], linewidth=0.2*N)
            # plt.plot([-1.5, -1.5], [-0.5 + n, 0.5 + n], c=x_colors[n], linewidth=0.2*N)

        plt.ylim([N+0.5, -0.5-width])
        plt.xlim([-0.5-width, N+0.5])

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()


def plot_distance_matrix_added_data(
        distance_matrix,
        pose_names,
        max_distance=None,
        save_path=None,
        x_colors=None,
        show=True
):
    # Adds "Added" to class names to differentiate them
    # when painting.

    box_colors = [
        "red" if "Added" in pose_name else "black" for pose_name in pose_names
    ]

    plot_distance_matrix(
        distance_matrix,
        max_distance=max_distance,
        x_colors=x_colors,
        box_colors=box_colors,
        show=show,
        save_path=save_path
    )