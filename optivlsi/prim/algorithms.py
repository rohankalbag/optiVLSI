"""
Prim Minimum Spanning Tree algorithm implementations.

Provides three variants: pure Python, NetworkX-based, and Numba-accelerated.
"""

import time
import numba
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from numba.typed import List


def create_nx_graph(size, prob, wt_min, wt_max):
    """Create a random directed acyclic graph.

    Args:
        size: Number of nodes.
        prob: Probability of edge existence.
        wt_min: Minimum edge weight.
        wt_max: Maximum edge weight.

    Returns:
        NetworkX DiGraph.
    """
    generated = False
    while not generated:
        graph = nx.gnp_random_graph(size, prob, directed=True)
        circuit = nx.DiGraph([(u, v,
                                {'weight': random.randint(wt_min, wt_max)})
                              for (u, v) in graph.edges() if u < v])
        generated = nx.is_directed_acyclic_graph(circuit)
    return circuit


def graph_to_numpy(graph):
    """Convert a NetworkX graph to numpy arrays.

    Args:
        graph: NetworkX graph.

    Returns:
        Tuple of (nodes, edges) as numpy arrays.
    """
    nodes = np.array(graph.nodes, dtype=np.int64)
    nodes.sort()
    edges = graph.edges
    edge_list = []
    for e in edges:
        edge_list.append((e[0], e[1], graph[e[0]][e[1]]["weight"]))
    return (nodes, np.array(edge_list, dtype=np.int64))


def numpy_to_graph(nodes, edges):
    """Convert numpy arrays to a NetworkX graph (undirected for Prim).

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        NetworkX Graph (undirected).
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])
    return G


def prim_pythonic(nodes, edges):
    """Find MST using pure Python Prim.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    mst = []
    if len(edges) == 0:
        return (nodes, np.array(mst))
    visited = []
    curr_node = nodes[0]
    visited.append(curr_node)
    cond1 = np.array([False]*len(edges[:, 0]))
    cond2 = np.array([False]*len(edges[:, 0]))

    cond1 = cond1 | (edges[:, 0] == curr_node)
    cond2 = cond2 | (edges[:, 1] == curr_node)

    while (any((~cond1) | (~cond2))):
        connected_edges = edges[(cond1 & (~cond2)) | ((~cond1) & (cond2)), :]
        connected_edges_wts = connected_edges[:, 2]
        min_wt_index = np.argmin(connected_edges_wts)
        visited.append(connected_edges[min_wt_index, 1])
        curr_node = connected_edges[min_wt_index, 1]
        mst.append(connected_edges[min_wt_index, :])

        cond1 = cond1 | (edges[:, 0] == curr_node)
        cond2 = cond2 | (edges[:, 1] == curr_node)
        curr_node = connected_edges[min_wt_index, 0]
        cond1 = cond1 | (edges[:, 0] == curr_node)
        cond2 = cond2 | (edges[:, 1] == curr_node)

    return (nodes, np.array(mst))


def prim_networkx(nodes, edges):
    """Find MST using NetworkX Prim implementation.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    graph = numpy_to_graph(nodes, edges)
    mst = nx.minimum_spanning_tree(graph, algorithm='prim')
    n, e = graph_to_numpy(mst)
    return (n, e)


@numba.njit
def prim_numba_accelerated(nodes, edges, mst):
    """Find MST using Numba-accelerated Prim.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        mst: Numba typed list for output MST edges.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    mst = []
    visited = []
    curr_node = nodes[0]
    visited.append(curr_node)
    cond1 = np.array([False]*len(edges[:, 0]))
    cond2 = np.array([False]*len(edges[:, 0]))

    cond1 = cond1 | (edges[:, 0] == curr_node)
    cond2 = cond2 | (edges[:, 1] == curr_node)

    for i in range(len(nodes)):

        if (~(np.any((~cond1) | (~cond2)))):
            break

        connected_edges = edges[(cond1 & (~cond2)) | ((~cond1) & (cond2)), :]
        connected_edges_wts = connected_edges[:, 2]
        min_wt_index = np.argmin(connected_edges_wts)
        visited.append(connected_edges[min_wt_index, 1])
        curr_node = connected_edges[min_wt_index, 1]
        mst.append(connected_edges[min_wt_index, :])

        cond1 = cond1 | (edges[:, 0] == curr_node)
        cond2 = cond2 | (edges[:, 1] == curr_node)
        curr_node = connected_edges[min_wt_index, 0]
        cond1 = cond1 | (edges[:, 0] == curr_node)
        cond2 = cond2 | (edges[:, 1] == curr_node)

    return (nodes, mst)