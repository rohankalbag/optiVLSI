"""Tests for Bellman-Ford shortest path algorithm."""

import numpy as np
import pytest
from optivlsi.bellman_ford import (
    bellman_ford,
    bellman_ford_networkx,
    bellman_ford_numba,
)


def test_shortest_path_on_known_graph(small_bellman_ford_graph):
    """Correct shortest path on known 5-node DAG."""
    nodes, edges = small_bellman_ford_graph
    status, path = bellman_ford(nodes, edges, 0, 4)
    assert status == 1
    # Two possible paths: [0, 2, 4] (weight 7) or [0, 1, 3, 4] (weight 12)
    assert path == [0, 2, 4], f"Expected [0, 2, 4], got {path}"


def test_correct_distance(small_bellman_ford_graph):
    """Distance value must be correct. Path [0,2,4] has total weight 7."""
    nodes, edges = small_bellman_ford_graph
    status, path = bellman_ford(nodes, edges, 0, 4)
    assert status == 1
    # Compute path weight
    weight = 0
    for i in range(len(path) - 1):
        mask = (edges[:, 0] == path[i]) & (edges[:, 1] == path[i + 1])
        weight += edges[mask][0, 2]
    assert weight == 7


def test_no_path():
    """No path case returns (-1, None)."""
    nodes = np.array([0, 1, 2], dtype=np.int64)
    edges = np.array([[0, 1, 1]], dtype=np.int64)  # No edge to node 2
    status, path = bellman_ford(nodes, edges, 0, 2)
    assert status == -1
    assert path is None


def test_implementations_match(small_bellman_ford_graph):
    """Pythonic vs NetworkX vs Numba produce identical results."""
    nodes, edges = small_bellman_ford_graph
    py_result = bellman_ford(nodes, edges, 0, 4)
    nx_result = bellman_ford_networkx(nodes, edges, 0, 4)
    nb_result = bellman_ford_numba(nodes, edges, 0, 4)

    assert py_result == nx_result
    assert py_result == nb_result


def test_single_node():
    """Single-node graph: src == end should find trivial path."""
    nodes = np.array([0], dtype=np.int64)
    edges = np.empty((0, 3), dtype=np.int64)
    status, path = bellman_ford(nodes, edges, 0, 0)
    assert status == 1
    assert path == [0]


def test_disconnected_graph():
    """Disconnected graph where no path exists."""
    nodes = np.array([0, 1, 2, 3], dtype=np.int64)
    # Two disconnected components: {0, 1} and {2, 3}
    edges = np.array([
        [0, 1, 2],
        [2, 3, 3],
    ], dtype=np.int64)
    status, path = bellman_ford(nodes, edges, 0, 3)
    assert status == -1
    assert path is None