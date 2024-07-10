import numpy as np
import torch
import os
from os import path
import math

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as pltc
from mayavi import mlab

from HyperbolicEmbeddings.hyperbolic_manifold.lorentz_functions_torch import poincare_to_lorentz


def plot_hyperbolic_gplvm_2d(x_poincare, x_colors, x_legend=None, save_path=None, show=True, geodesics=None,
                             fig=None, marker=None, alpha=None):
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        close = True
    else:
        close = False
        ax = fig.gca()

    # Plot Poincaré disk
    circle = plt.Circle(np.array([0, 0]), radius=1., color='black', fill=False, linewidth=2., zorder=3)
    ax.add_patch(circle)

    # Plot points
    if marker == "*":
        s = 400
    else:
        s = 140
    for n in range(x_poincare.shape[0]):
        plt.scatter(x_poincare[n, 0], x_poincare[n, 1], c=x_colors[n], edgecolors='black', s=s, linewidths=2.,
                    zorder=2, alpha=alpha, marker=marker)
        if x_legend:
            plt.text(x_poincare[n, 0], x_poincare[n, 1], x_legend[n], fontdict={"fontsize": 10})

    # Plot geodesics
    if geodesics:
        for g in range(len(geodesics)):
            geodesic = geodesics[g]
            plt.plot(geodesic[:, 0], geodesic[:, 1], color='black')

    plt.axis('off')

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()
    
    if close:
        plt.close()
    

def plot_hyperbolic_gplvm_3d(x_poincare, x_colors, x_legend=None, save_path=None, show=True, geodesics=None,
                             fig=None, marker=None, opacity=1.0):

    if fig is None:
        # Mayavi plot of the Poincare ball
        fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
        num_pts = 200
        u = np.linspace(0, 2 * np.pi, num_pts)
        v = np.linspace(0, np.pi, num_pts)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        mlab.clf()
        mlab.mesh(x, y, z, color=(0.7, 0.7, 0.7), opacity=0.3)

    # Plot points
    if marker is None:
        marker = 'sphere'
    if marker != 'sphere':
        line_width = 5.
        scale_factor = 0.12
    else:
        line_width = 2.
        scale_factor = 0.07
    for n in range(x_poincare.shape[0]):
        mlab.points3d(x_poincare[n, 0], x_poincare[n, 1], x_poincare[n, 2],
                      color=pltc.to_rgb(x_colors[n]), scale_factor=scale_factor, mode=marker, opacity=opacity,
                      line_width=line_width)
        if marker == '2dcross':
            mlab.points3d(x_poincare[n, 0], x_poincare[n, 1], x_poincare[n, 2],
                          color=pltc.to_rgb('black'), scale_factor=scale_factor+0.02, mode='2dthick_cross', opacity=opacity, line_width=line_width)
        if x_legend:
            mlab.text3d(x_poincare[n, 0], x_poincare[n, 1], x_poincare[n, 2], x_legend[n],
                        scale=0.03)

    # Plot geodesics
    if geodesics:
        for g in range(len(geodesics)):
            geodesic = geodesics[g]
            mlab.plot3d(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2], color=pltc.to_rgb('black'),
                            line_width=2.5, tube_radius=None)

    if save_path is not None:
        mlab.savefig(str(save_path), size=(400, 400))

    if show:
        mlab.show()
    
    return fig

