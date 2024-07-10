from matplotlib import pyplot as plt
import matplotlib.colors as pltc
from mayavi import mlab


def plot_euclidean_gplvm_2d(x_latent, x_colors, x_legend=None, save_path=None, show=True, geodesics=None,
                            fig=None, marker=None, alpha=None):
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        close = True
    else:
        ax = fig.gca()
        close = False

    # Plot points
    if marker == "*":
        s = 400
    else:
        s = 140
    for n in range(x_latent.shape[0]):

        plt.scatter(
            x_latent[n, 0],
            x_latent[n, 1],
            c=x_colors[n],
            edgecolors="black",
            s=s,
            zorder=2,
            linewidths=2.0,
            marker=marker,
            alpha=alpha,
        )
        if x_legend:
            plt.text(
                x_latent[n, 0], x_latent[n, 1], x_legend[n], fontdict={"fontsize": 20}
            )
    xmin, xmax, ymin, ymax = plt.axis()

    diff_axes = (xmax - xmin) - (ymax - ymin)
    if diff_axes < 0:
        xmax += -diff_axes / 2.0
        xmin -= -diff_axes / 2.0
    elif diff_axes > 0:
        ymax += diff_axes / 2.0
        ymin -= diff_axes / 2.0

    # Plot geodesics
    if geodesics:
        for g in range(len(geodesics)):
            geodesic = geodesics[g]
            plt.plot(geodesic[:, 0], geodesic[:, 1], color="black")

    plt.axis('equal')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.axis("off")

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    if close:
        plt.close()


def plot_euclidean_gplvm_3d(x_latent, x_colors, x_legend=None, geodesics=None, save_path=None, show=True,
    fig=None, marker=None, opacity=1.0):
    if fig is None:
        fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))

    # Plot points
    if marker is None:
        marker = "sphere"
    if marker != "sphere":
        line_width = 5.0
        scale_factor = 0.2
    else:
        line_width = 2.0
        scale_factor = 0.15
    for n in range(x_latent.shape[0]):
        mlab.points3d(
            x_latent[n, 0],
            x_latent[n, 1],
            x_latent[n, 2],
            color=pltc.to_rgb(x_colors[n]),
            scale_factor=scale_factor,
            mode=marker,
            opacity=opacity,
            line_width=line_width,
        )
        if marker == "2dcross":
            mlab.points3d(
                x_latent[n, 0],
                x_latent[n, 1],
                x_latent[n, 2],
                color=pltc.to_rgb("black"),
                scale_factor=scale_factor + 0.02,
                mode="2dthick_cross",
                opacity=opacity,
                line_width=line_width,
            )
        if x_legend:
            mlab.text3d(
                x_latent[n, 0], x_latent[n, 1], x_latent[n, 2], x_legend[n], scale=0.03
            )

    # Plot geodesics
    if geodesics:
        for g in range(len(geodesics)):
            geodesic = geodesics[g]
            mlab.plot3d(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2], color=pltc.to_rgb("black"), line_width=2.5,
                        tube_radius=None)

    if save_path is not None:
        mlab.savefig(save_path)

    if show:
        mlab.show()

    return fig

