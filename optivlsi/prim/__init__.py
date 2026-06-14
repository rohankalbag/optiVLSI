"""
Prim Minimum Spanning Tree algorithm implementations.

Provides three variants:
- prim(): Pure Python implementation
- prim_networkx(): NetworkX-based implementation
- prim_numba(): Numba-accelerated implementation
"""

from .algorithms import (
    prim_pythonic,
    prim_networkx as _prim_networkx,
    prim_numba_accelerated,
    graph_to_numpy,
    numpy_to_graph,
    create_nx_graph as _create_nx_graph,
)


def prim(nodes, edges):
    """Find MST using pure Python Prim.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    return prim_pythonic(nodes, edges)


def prim_networkx(nodes, edges):
    """Find MST using NetworkX Prim implementation.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        Tuple of (nodes, mst_edges) for the minimum spanning tree.
    """
    return _prim_networkx(nodes, edges)


def prim_numba(nodes, edges):
    """Find MST using Numba-accelerated Prim.

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
    return prim_numba_accelerated(nodes, edges, mst)


__all__ = [
    "prim",
    "prim_networkx",
    "prim_numba",
    "graph_to_numpy",
    "numpy_to_graph",
]