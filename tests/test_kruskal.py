"""Tests for Kruskal Minimum Spanning Tree algorithm."""

import numpy as np
import pytest
from optivlsi.kruskal import (
    kruskal,
    kruskal_networkx,
    kruskal_numba,
)


def compute_mst_weight(nodes, mst_edges):
    """Helper to compute total MST weight."""
    if isinstance(mst_edges, list):
        total = sum(e[2] for e in mst_edges)
    else:
        total = int(np.sum(mst_edges[:, 2]))
    return total


def test_mst_total_weight(small_kruskal_graph):
    """Correct MST total weight on small 4-node graph."""
    nodes, edges = small_kruskal_graph
    mst_nodes, mst_edges = kruskal(nodes, edges)
    weight = compute_mst_weight(mst_nodes, mst_edges)
    assert weight == 6, f"Expected MST weight 6, got {weight}"


def test_mst_edge_count(small_kruskal_graph):
    """Correct number of edges in MST (n-1)."""
    nodes, edges = small_kruskal_graph
    mst_nodes, mst_edges = kruskal(nodes, edges)
    expected_edges = len(nodes) - 1
    if isinstance(mst_edges, list):
        assert len(mst_edges) == expected_edges
    else:
        assert mst_edges.shape[0] == expected_edges


def test_implementations_match(small_kruskal_graph):
    """Pythonic vs NetworkX vs Numba produce identical MST weights."""
    nodes, edges = small_kruskal_graph
    py_nodes, py_edges = kruskal(nodes, edges)
    nx_nodes, nx_edges = kruskal_networkx(nodes, edges)
    nb_nodes, nb_edges = kruskal_numba(nodes, edges)

    py_weight = compute_mst_weight(py_nodes, py_edges)
    nx_weight = compute_mst_weight(nx_nodes, nx_edges)
    nb_weight = compute_mst_weight(nb_nodes, nb_edges)

    assert py_weight == nx_weight == nb_weight


def test_disconnected_graph():
    """Disconnected graph should still produce a valid MST for each component."""
    nodes = np.array([0, 1, 2], dtype=np.int64)
    edges = np.array([
        [0, 1, 5],
    ], dtype=np.int64)
    mst_nodes, mst_edges = kruskal(nodes, edges)
    # Component {0,1} gets one edge, node 2 is isolated
    if isinstance(mst_edges, list):
        assert len(mst_edges) == 1
    else:
        assert mst_edges.shape[0] == 1


def test_single_node():
    """Single-node graph has empty MST."""
    nodes = np.array([0], dtype=np.int64)
    edges = np.empty((0, 3), dtype=np.int64)
    mst_nodes, mst_edges = kruskal(nodes, edges)
    if isinstance(mst_edges, list):
        assert len(mst_edges) == 0
    else:
        assert mst_edges.shape[0] == 0