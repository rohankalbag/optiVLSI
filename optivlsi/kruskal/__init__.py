"""
Kruskal Minimum Spanning Tree algorithm implementations.

Provides three variants:
- kruskal(): Pure Python implementation
- kruskal_networkx(): NetworkX-based implementation
- kruskal_numba(): Numba-accelerated implementation
"""

import numpy as np

from .algorithms import (
    kruskal_pythonic,
    kruskal_networkx as _kruskal_networkx,
    kruskal_numba_accelerated,
    graph_to_numpy,
    numpy_to_graph,
    create_nx_graph as _create_nx_graph,
)


def kruskal(nodes, edges):
    """Find MST using pure Python Kruskal with DSU.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    return kruskal_pythonic(nodes, edges)


def kruskal_networkx(nodes, edges):
    """Find MST using NetworkX Kruskal implementation.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    return _kruskal_networkx(nodes, edges)


def kruskal_numba(nodes, edges):
    """Find MST using Numba-accelerated Kruskal with DSU.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    from numba.typed import List

    mst = List()
    mst.append((1, 1, 1))
    mst.pop()
    result_nodes, result_edges = kruskal_numba_accelerated(nodes, edges, mst)
    # Convert numba typed list to numpy array for consistent return type
    edge_list = [(e[0], e[1], e[2]) for e in result_edges]
    return (result_nodes, np.array(edge_list, dtype=np.int64))


__all__ = [
    "kruskal",
    "kruskal_networkx",
    "kruskal_numba",
    "graph_to_numpy",
    "numpy_to_graph",
]