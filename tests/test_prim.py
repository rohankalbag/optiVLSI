"""Tests for Prim Minimum Spanning Tree algorithm."""

import numpy as np
import pytest
from optivlsi.prim import (
    prim,
    prim_networkx,
    prim_numba,
)


def compute_mst_weight(nodes, mst_edges):
    """Helper to compute total MST weight."""
    if isinstance(mst_edges, list):
        total = sum(e[2] for e in mst_edges)
    else:
        total = int(np.sum(mst_edges[:, 2]))
    return total


def test_mst_total_weight(small_prim_graph):
    """Correct MST total weight on small 4-node graph."""
    nodes, edges = small_prim_graph
    mst_nodes, mst_edges = prim(nodes, edges)
    weight = compute_mst_weight(mst_nodes, mst_edges)
    assert weight == 6, f"Expected MST weight 6, got {weight}"


def test_mst_edge_count(small_prim_graph):
    """Correct number of edges in MST (n-1)."""
    nodes, edges = small_prim_graph
    mst_nodes, mst_edges = prim(nodes, edges)
    expected_edges = len(nodes) - 1
    if isinstance(mst_edges, list):
        assert len(mst_edges) == expected_edges
    else:
        assert mst_edges.shape[0] == expected_edges


def test_implementations_match(small_prim_graph):
    """Pythonic vs NetworkX vs Numba produce identical MST weights."""
    nodes, edges = small_prim_graph
    py_nodes, py_edges = prim(nodes, edges)
    nx_nodes, nx_edges = prim_networkx(nodes, edges)
    nb_nodes, nb_edges = prim_numba(nodes, edges)

    py_weight = compute_mst_weight(py_nodes, py_edges)
    nx_weight = compute_mst_weight(nx_nodes, nx_edges)
    nb_weight = compute_mst_weight(nb_nodes, nb_edges)

    assert py_weight == nx_weight == nb_weight


def test_single_node():
    """Single-node graph has empty MST."""
    nodes = np.array([0], dtype=np.int64)
    edges = np.empty((0, 3), dtype=np.int64)
    mst_nodes, mst_edges = prim(nodes, edges)
    if isinstance(mst_edges, list):
        assert len(mst_edges) == 0
    else:
        assert mst_edges.shape[0] == 0