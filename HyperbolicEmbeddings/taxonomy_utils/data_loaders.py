from pathlib import Path
import numpy as np
import torch

from HyperbolicEmbeddings.taxonomy_utils.grasp_data_utils import load_grasps_data
from HyperbolicEmbeddings.utils.normalization import centering

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def load_taxonomy_data(dataset):
    # Load data
    data_folder_path = ROOT_DIR / 'data'

    if dataset == 'grasps':
        training_data, adjacency_matrix, graph_distances, nodes_names, indices_in_graph, \
        nodes_legend_for_plot, color_function, joint_names = \
            load_grasps_data(data_folder_path)

        max_manifold_distance = 5.0

    else:
        raise NotImplementedError

    # Center
    training_data, data_mean = centering(training_data)

    # Rescale distance
    if max_manifold_distance:
        max_graph_distance = np.max(graph_distances.detach().numpy())
        graph_distances = graph_distances / max_graph_distance * max_manifold_distance

    return training_data, data_mean, adjacency_matrix, graph_distances, nodes_names, indices_in_graph, \
           nodes_legend_for_plot, color_function, max_manifold_distance, joint_names
