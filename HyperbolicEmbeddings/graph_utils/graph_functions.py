import os
import numpy as np
import pandas
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_edge_list(path, symmetrize=False):
    """
    This function load the graph edges, graph node indexes and correspoding labels from graph file.

    Note: Taken from Poincare embeddings code

    Parameters
    ----------
    path: File path
    symmetrize: Flag to symmetrize graph info

    Returns
    -------
    idx: Node indexes
    objects:
    weights: Edge weights of the graph
    """
    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights


def get_indices_nodes_graph(file_path_graph: str, nodes_label):
    """
    This function gets the indices values of nodes in the graph.
    These indices corresponds to the indices of the nodes in the adjacency matrix.

    Parameters
    ----------
    file_path_graph: File path to the graph info
    nodes_label: List of N nodes labels, must match the names associated to graph nodes

    Returns
    -------
    adjacency: adjacency matrix associated to the shape poses given as inputs
    graph_indices: N indices associated to the N nodes given as inputs, corresponding to the adjacency matrix

    """
    # Load edges, node indexes and labels from graph file
    idx, objects, weights = load_edge_list(file_path_graph, symmetrize=True)

    # Compute adjacency matrix
    adjacency = adjacency_from_edges(edges_list=idx, nb_edges=len(objects), weights=weights, symmetrize=True)

    # Get indices
    graph_indices = []
    for i in range(len(nodes_label)):
        graph_indices.append(objects.index(nodes_label[i]))

    return adjacency, np.array(graph_indices)


def adjacency_from_edges(edges_list, nb_edges, weights=None, symmetrize=False):
    """
    This function computes the adjacency matrix of a graph from its edges.

    Parameters
    ----------
    :param edges_list: list of the edges of the considered graph
    :param nb_edges: number of edges of the considered graph

    Optional parameters
    -------------------
    :param weights: weight of the edges [nb_edges]
    :param symmetrize: if True, the adjacency matrix is computed symmetrically

    Return
    ------
    :return adjacency: adjacency matrix [nb_edges x nb_edges]
    """
    adjacency = np.zeros((nb_edges, nb_edges))
    if weights is None:
        weights = np.ones(nb_edges)

    idx = 0
    for edge in edges_list:
        adjacency[edge[0], edge[1]] = weights[idx]
        if symmetrize:
            adjacency[edge[1], edge[0]] = weights[idx]
        idx += 1

    return adjacency


def graph_distance_matrix_from_adjacency(adjacency):
    """
    This function computes the distance matrix of a graph given its adjacency matrix.

    Parameters
    ----------
    :param adjacency: adjacency matrix of the considered graph [nb_edges x nb_edges]

    Return
    ------
    :return distance matrix of the considered graph [nb_edges x nb_edges]
    """
    distances = np.empty_like(adjacency)
    distances[:] = np.NaN
    # Diagonal has 0 distances
    distances[np.diag_indices(distances.shape[0])] = 0.

    adjacency_pow = np.eye(adjacency.shape[0])
    degree = 0
    while np.isnan(np.sum(distances)):
        adjacency_pow = np.dot(adjacency_pow, adjacency)
        degree += 1
        distances[np.isnan(distances) & (adjacency_pow > 0)] = degree

    return distances


def distance_matrix_from_tree(idx, tree_leave_nodes_labels, tree_distances):
    """
    This function does a simple distance mapping given the array of (previously-computed) distances 'tree_distances' and
    the labels of the tree leave nodes. It does so by using the corresponding index stored in 'idx'.

    Parameters
    ----------
    idx: List of pairs of indexes extracted from the hand grasp taxonomy file
    tree_leave_nodes_labels: Labels of the tree leave nodes
    tree_distances: Vector of distances among all the pairs of leave nodes of the tree, stored in the hand taxonomy file

    Returns
    -------
    distance_matrix: Matrix of distances for all the pairs of types of grasps of the hand grasp taxonomy
    """
    distance_matrix = np.zeros((len(tree_leave_nodes_labels), len(tree_leave_nodes_labels)))

    for i in range(len(idx)):
        distance_matrix[idx[i][0], idx[i][1]] = tree_distances[i]

    return distance_matrix


def nodes_distance_mapping(file_path_graph: str, nodes_labels):
    """
    Maps the graph distance matrix for all the pairs of given node labels.
    This function merely maps the graph distances already store in the file given as "file_path_graph" argument.

    Parameters
    ----------
    file_path_graph: File path to the graph info
    nodes_labels: List of N nodes labels corresponding to N datapoints

    Optional parameters
    -------------------

    Returns
    -------
    hand_grasps_graph_distances: NxN Matrix of graph distances associated to the N hand grasps given as inputs
    """
    # Load edges, node indexes and labels from graph file
    idx, objects, weights = load_edge_list(file_path_graph, symmetrize=True)

    # Initialize variables to map graph distances to shape poses corresponding to graph nodes
    N = len(nodes_labels)
    nodes_graph_distances = torch.zeros(N, N)

    # Compute adjacency matrix and corresponding graph distances
    adjacency = adjacency_from_edges(edges_list=idx, nb_edges=len(objects), weights=weights, symmetrize=True)
    distances = graph_distance_matrix_from_adjacency(adjacency)

    # Compute graph distance matrix for shape poses based on the labels assigned above
    for i in range(N):
        idx1 = objects.index(nodes_labels[i])
        for j in range(i + 1, N):
            idx2 = objects.index(nodes_labels[j])
            # Filling both distance entries as the distance matrix is symmetric
            nodes_graph_distances[i, j] = distances[idx1, idx2]
            nodes_graph_distances[j, i] = distances[idx1, idx2]

    return nodes_graph_distances


if __name__ == '__main__':
    dset = os.path.join(CURRENT_DIR, '../../examples/embeddings_support_poses_taxonomy/support_poses_closure.csv')
    idx, objects, weights = load_edge_list(dset, symmetrize=True)

    adjacency = adjacency_from_edges(edges_list=idx, nb_edges=len(objects), weights=weights, symmetrize=True)
    distances = graph_distance_matrix_from_adjacency(adjacency)

    print(adjacency)
    print(distances)

