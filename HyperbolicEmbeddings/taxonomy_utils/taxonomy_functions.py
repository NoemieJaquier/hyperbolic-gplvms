import numpy as np


def reorder_taxonomy_data(data, node_names, ordered_node_names):
    """
    Reorder data according to a desired order of associated node names.

    Parameters
    ----------
    data: unordered data
    node_names: current names
    ordered_node_names: ordered names

    Returns
    -------
    reordered data

    """
    reordered_data = []

    for name in ordered_node_names:
        for n in range(len(data)):
            if node_names[n] == name:
                reordered_data.append(data[n])

    return reordered_data


def reorder_distance_matrix(distance_matrix, node_names, ordered_node_names):
    """
    Reorder a distance matrix according to a desired order of node names associated to the corresponding data.

    Parameters
    ----------
    distance_matrix: unordered matrix
    node_names: current names
    ordered_node_names: ordered names

    Returns
    -------
    reordered distance matrix

    """
    indices = np.array(range(0, distance_matrix.shape[0]))
    reordered_indices = reorder_taxonomy_data(indices, node_names, ordered_node_names)

    reordered_distance_matrix = np.zeros_like(distance_matrix)

    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            reordered_distance_matrix[i, j] = distance_matrix[reordered_indices[i], reordered_indices[j]]
            reordered_distance_matrix[j, i] = distance_matrix[reordered_indices[j], reordered_indices[i]]

    return reordered_distance_matrix

