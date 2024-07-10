from __future__ import division

from pathlib import Path

import numpy as np
import pandas
import torch
from xml.etree import ElementTree
import os
from operator import itemgetter

from HyperbolicEmbeddings.graph_utils.graph_functions import get_indices_nodes_graph, nodes_distance_mapping

# Hand grasp types as defined in [1]
# [1] F. Stival, et al. "A quantitative taxonomy of human hand grasps". 2019
HAND_GRASPS_NAMES = ["Lateral", "ExtensionType", "Quadpod", "ParallelExtension", "IndexFingerExtension", "Stick",
                     "WritingTripod", "PrismaticFourFingers", "PowerDisk", "LargeDiameter", "MediumWrap",
                     "SmallDiameter", "FixedHook", "Tripod", "PowerSphere", "PrecisionSphere", "ThreeFingersSphere",
                     "PrismaticPinch", "TipPinch", "Ring"]
ordered_hand_grasps_names = ["Lateral", "LateralAdded",
                             "ExtensionType", "ExtensionTypeAdded",
                             "Quadpod", "QuadpodAdded",
                             "ParallelExtension", "ParallelExtensionAdded",
                             "IndexFingerExtension", "IndexFingerExtensionAdded",
                             "Stick", "StickAdded",
                             "WritingTripod", "WritingTripodAdded",
                             "PrismaticFourFingers", "PrismaticFourFingersAdded",
                             "PowerDisk", "PowerDiskAdded",
                             "LargeDiameter", "LargeDiameterAdded",
                             "MediumWrap", "MediumWrapAdded",
                             "SmallDiameter", "SmallDiameterAdded",
                             "FixedHook", "FixedHookAdded",
                             "Tripod", "TripodAdded",
                             "PowerSphere", "PowerSphereAdded",
                             "PrecisionSphere", "PrecisionSphereAdded",
                             "ThreeFingersSphere", "ThreeFingersSphereAdded",
                             "PrismaticPinch", "PrismaticPinchAdded",
                             "TipPinch", "TipPinchAdded",
                             "Ring", "RingAdded"]
# Prismatic Pinch == Palmar Pinch
# ThreeFingersSphere missing in the data


def load_grasps_data(data_folder_path):
    # Load data
    data_folder_path = data_folder_path / "hand_grasps"
    dataset_path = data_folder_path / "grasp_dataset.npz"
    data = np.load(dataset_path)
    hand_joint_data = data['joint_data']
    grasp_names = data['grasp_names']
    joint_names = data['joint_names']
    grasp_names = list(grasp_names)

    # Mapping grasps labels to short labels associated to the tree leaves (for each grasp datapoint)
    hand_grasp_labels = hand_grasp_name_to_grasp_labels(grasp_names)

    # Adjacency matrix
    adjacency_file_path = data_folder_path / 'hand_grasps_closure.csv'
    adjacency_matrix, grasps_indices_in_tree = get_indices_nodes_graph(adjacency_file_path, hand_grasp_labels)

    # Distances
    # graph_file_path = data_folder_path / 'hand_grasps_taxonomy.csv'
    grasps_graph_distances = nodes_distance_mapping(adjacency_file_path, hand_grasp_labels)

    # Legend
    grasps_legend_for_plot = hand_grasp_name_to_grasp_labels(grasp_names)

    # Set training data
    training_data = torch.from_numpy(hand_joint_data)
    grasps_indices_in_tree = torch.from_numpy(grasps_indices_in_tree)

    return training_data, adjacency_matrix, grasps_graph_distances, grasp_names, grasps_indices_in_tree, \
           grasps_legend_for_plot, simple_color_function_grasp, joint_names


def hand_grasp_name_to_grasp_labels(hand_grasp_names):
    """
    This function maps the hand grasps names presented in [1] to names of nodes of the grasping taxonomy.
    """
    grasp_labels = [''] * len(hand_grasp_names)

    # For loop to map pose labels to support pose nodes
    for grasp_id, grasp_name in enumerate(hand_grasp_names):
        if grasp_name == 'Lateral':
            grasp_labels[grasp_id] = 'La'
        elif grasp_name == 'ExtensionType':
            grasp_labels[grasp_id] = 'ET'
        elif grasp_name == 'Quadpod':
            grasp_labels[grasp_id] = 'Qu'
        elif grasp_name == 'ParallelExtension':
            grasp_labels[grasp_id] = 'PE'
        elif grasp_name == 'IndexFingerExtension':
            grasp_labels[grasp_id] = 'IE'
        elif grasp_name == 'Stick':
            grasp_labels[grasp_id] = 'St'
        elif grasp_name == 'WritingTripod':
            grasp_labels[grasp_id] = 'WT'
        elif grasp_name == 'PrismaticFourFingers':
            grasp_labels[grasp_id] = 'PF'
        elif grasp_name == 'PowerDisk':
            grasp_labels[grasp_id] = 'PD'
        elif grasp_name == 'LargeDiameter':
            grasp_labels[grasp_id] = 'LD'
        elif grasp_name == 'MediumWrap':
            grasp_labels[grasp_id] = 'MW'
        elif grasp_name == 'SmallDiameter':
            grasp_labels[grasp_id] = 'SD'
        elif grasp_name == 'FixedHook':
            grasp_labels[grasp_id] = 'FH'
        elif grasp_name == 'Tripod':
            grasp_labels[grasp_id] = 'Tr'
        elif grasp_name == 'PowerSphere':
            grasp_labels[grasp_id] = 'PS'
        elif grasp_name == 'PrecisionSphere':
            grasp_labels[grasp_id] = 'RS'
        elif grasp_name == 'ThreeFingersSphere':
            grasp_labels[grasp_id] = 'TS'
        elif grasp_name == 'PrismaticPinch':  # = Palmar pinch
            grasp_labels[grasp_id] = 'PP'
        elif grasp_name == 'TipPinch':
            grasp_labels[grasp_id] = 'TP'
        elif grasp_name == 'Ring':
            grasp_labels[grasp_id] = 'Ri'

    return grasp_labels


def simple_color_function_grasp(grasp_name):
    if grasp_name == 'Lateral':
        color = "darkgreen"  # (0.0, 0.39215686274509803, 0.0)
    elif grasp_name == 'ExtensionType':
        color = (0.0, 0.1, 0.0)  # "darkgreen"
    elif grasp_name == 'Quadpod':
        color = (0.0, 0.7, 0.0)  # "darkgreen"
    elif grasp_name == 'ParallelExtension':
        color = "aquamarine"
    elif grasp_name == 'IndexFingerExtension':
        color = "gray"
    elif grasp_name == 'Stick':
        color = (0.8, 0.5, 0.0)  # "darkorange"
    elif grasp_name == 'WritingTripod':
        color = (1.0, 0.4, 0.0)  # "darkorange"  # (1.0, 0.5490196078431373, 0.0)
    elif grasp_name == 'PrismaticFourFingers':
        color = (1.0, 0.7, 0.0)  # "orange"  # (1.0, 0.6470588235294118, 0.0)
    elif grasp_name == 'PowerDisk':
        color = (0.6, 0.3, 0.0)  # "darkorange"
    elif grasp_name == 'LargeDiameter':
        color = "royalblue"
    elif grasp_name == 'MediumWrap':
        color = (0.5, 0.7, 1.0)  # "royalblue"
    elif grasp_name == 'SmallDiameter':
        color = "mediumblue"
    elif grasp_name == 'FixedHook':
        color = (0.0, 0.0, 0.3)
    elif grasp_name == 'Tripod':
        color = (0.8, 0.8, 0.0)  # "gold"
    elif grasp_name == 'PowerSphere':
        color = "gold"  # (1.0, 0.8431372549019608, 0.0)
    elif grasp_name == 'PrecisionSphere':
        color = (1.0, 0.9, 0.7)  # "gold"
    elif grasp_name == 'ThreeFingersSphere':
        color = (0.65, 0.0, 0.1)  # "crimson"
    elif grasp_name == 'PrismaticPinch':  # = Palmar pinch
        color = (1.0, 0.6, 0.6)  # "crimson"
    elif grasp_name == 'TipPinch':
        color = "crimson"
    elif grasp_name == 'Ring':
        color = (0.6, 0.0, 0.)  # "crimson"

    return color






