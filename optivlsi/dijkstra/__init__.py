"""
Dijkstra shortest path algorithm implementations.

Provides three variants:
- dijkstra(): Pure Python implementation
- dijkstra_networkx(): NetworkX-based implementation
- dijkstra_numba(): Numba-accelerated implementation
"""

from .algorithms import (
    dijkstra_pythonic,
    dijkstra_nx,
    dijkstra_numba_accelerated,
    graph_to_numpy,
    numpy_to_graph,
    create_nx_graph as _create_nx_graph,
)


def dijkstra(nodes, edges, src, end):
    """Find shortest path using pure Python Dijkstra.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    return dijkstra_pythonic(nodes, edges, src, end)


def dijkstra_networkx(nodes, edges, src, end):
    """Find shortest path using NetworkX Dijkstra.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    return dijkstra_nx(nodes, edges, src, end)


def dijkstra_numba(nodes, edges, src, end):
    """Find shortest path using Numba-accelerated Dijkstra.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    from numba.typed import List

    priority_queue = List()
    priority_queue.append((1, 1, 1))
    priority_queue.pop()

    path = List()
    path.append(1)
    path.pop()

    result = dijkstra_numba_accelerated(
        nodes, edges, src, end, priority_queue, path
    )
    # Numba version returns path in reverse, flip it back for consistency
    if result[0] > 0:
        return (1, list(result[1][::-1]))
    return (int(result[0]), None)


__all__ = [
    "dijkstra",
    "dijkstra_networkx",
    "dijkstra_numba",
    "graph_to_numpy",
    "numpy_to_graph",
]