"""
Bellman-Ford shortest path algorithm implementations.

Provides three variants:
- bellman_ford(): Pure Python implementation
- bellman_ford_networkx(): NetworkX-based implementation
- bellman_ford_numba(): Numba-accelerated implementation
"""

from .algorithms import (
    bellman_ford_pythonic,
    bellman_ford_nx,
    bellman_ford_numba_accelerated,
    graph_to_numpy,
    numpy_to_graph,
    create_nx_graph as _create_nx_graph,
)


def bellman_ford(nodes, edges, src, end):
    """Find shortest path using pure Python Bellman-Ford.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    return bellman_ford_pythonic(nodes, edges, src, end)


def bellman_ford_networkx(nodes, edges, src, end):
    """Find shortest path using NetworkX Bellman-Ford.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    return bellman_ford_nx(nodes, edges, src, end)


def bellman_ford_numba(nodes, edges, src, end):
    """Find shortest path using Numba-accelerated Bellman-Ford.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    from numba.typed import List

    path = List()
    path.append(1)
    path.pop()
    return bellman_ford_numba_accelerated(nodes, edges, src, end, path)


__all__ = [
    "bellman_ford",
    "bellman_ford_networkx",
    "bellman_ford_numba",
    "graph_to_numpy",
    "numpy_to_graph",
]