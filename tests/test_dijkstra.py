"""Tests for Dijkstra shortest path algorithm."""

import numpy as np
import pytest
from optivlsi.dijkstra import (
    dijkstra,
    dijkstra_networkx,
    dijkstra_numba,
)


def test_shortest_path_on_known_graph(small_dijkstra_graph):
    """Correct shortest path on known 6-node directed graph."""
    nodes, edges = small_dijkstra_graph
    status, path = dijkstra(nodes, edges, 0, 5)
    assert status == 1
    # Shortest path: 0 -> 2 -> 4 -> 5 (total: 9)
    # Or: 0 -> 2 -> 1 -> 3 -> 5 (total: 10)
    # The shortest is 0 -> 2 -> 4 -> 5
    assert path == [0, 2, 4, 5], f"Expected [0, 2, 4, 5], got {path}"


def test_no_path():
    """No path case returns (-1, None)."""
    nodes = np.array([0, 1, 2], dtype=np.int64)
    edges = np.array([[0, 1, 1]], dtype=np.int64)
    status, path = dijkstra(nodes, edges, 0, 2)
    assert status == -1
    assert path is None


def test_implementations_match(small_dijkstra_graph):
    """Pythonic vs NetworkX vs Numba produce identical results."""
    nodes, edges = small_dijkstra_graph
    py_result = dijkstra(nodes, edges, 0, 5)
    nx_result = dijkstra_networkx(nodes, edges, 0, 5)
    nb_result = dijkstra_numba(nodes, edges, 0, 5)

    assert py_result == nx_result
    assert py_result == nb_result


def test_single_node():
    """Single-node graph: src == end should find trivial path."""
    nodes = np.array([0], dtype=np.int64)
    edges = np.empty((0, 3), dtype=np.int64)
    status, path = dijkstra(nodes, edges, 0, 0)
    assert status == 1
    assert path == [0]


def test_disconnected_graph():
    """Disconnected graph where no path exists."""
    nodes = np.array([0, 1, 2, 3], dtype=np.int64)
    edges = np.array([
        [0, 1, 2],
        [2, 3, 3],
    ], dtype=np.int64)
    status, path = dijkstra(nodes, edges, 0, 3)
    assert status == -1
    assert path is None