"""
Kruskal Minimum Spanning Tree algorithm implementations.

Provides three variants: pure Python (DSU), NetworkX-based, and Numba-accelerated.
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
    """Create a random directed acyclic graph (used as undirected for MST).

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
        circuit = nx.DiGraph([(u, v, {'weight': random.randint(wt_min, wt_max)}) for
                              (u, v) in graph.edges() if u < v])
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
    """Convert numpy arrays to a NetworkX graph (undirected for Kruskal).

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


def kruskal_pythonic(nodes, edges):
    """Find MST using pure Python Kruskal with Disjoint Set Union.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    parent = np.array(range(len(nodes)))
    rank = [0]*len(nodes)

    def find(i):
        if not (parent[i] == i):
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        x, y = find(i), find(j)
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[x] = y
            rank[y] += 1

    mst = []
    wts = edges[:, 2]
    sorted_indices = np.argsort(wts)
    sorted_edges = edges[sorted_indices]

    for e in sorted_edges:
        src, end, wt = e
        if find(src) != find(end):
            union(src, end)
            mst.append(e)

    return (nodes, np.array(mst))


def kruskal_networkx(nodes, edges):
    """Find MST using NetworkX Kruskal implementation.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    graph = numpy_to_graph(nodes, edges)
    mst = nx.minimum_spanning_tree(graph)
    n, e = graph_to_numpy(mst)
    return (n, e)


@numba.njit
def find(i, parent):
    """Find with path compression for DSU (Numba-compatible)."""
    if not (parent[i] == i):
        parent[i] = find(parent[i], parent)
    return parent[i]


@numba.njit
def union(i, j, rank, parent):
    """Union by rank for DSU (Numba-compatible)."""
    x, y = find(i, parent), find(j, parent)
    if rank[x] < rank[y]:
        parent[x] = y
    elif rank[x] > rank[y]:
        parent[y] = x
    else:
        parent[x] = y
        rank[y] += 1


@numba.njit
def kruskal_numba_accelerated(nodes, edges, mst):
    """Find MST using Numba-accelerated Kruskal with DSU.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        mst: Numba typed list for output MST edges.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    parent = np.copy(nodes)
    rank = np.zeros_like(nodes)

    wts = edges[:, 2]
    sorted_indices = np.argsort(wts)
    sorted_edges = edges[sorted_indices]

    for e in sorted_edges:
        src, end, wt = e
        if find(src, parent) != find(end, parent):
            union(src, end, rank, parent)
            mst.append((e[0], e[1], e[2]))

    return (nodes, mst)