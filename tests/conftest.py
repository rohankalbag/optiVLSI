"""Shared pytest fixtures for optiVLSI tests."""

import numpy as np
import pytest
import tempfile
import os


@pytest.fixture
def small_bellman_ford_graph():
    """Fixture: returns (nodes, edges) for a known 5-node DAG.

    Graph: 0 -> 1 (5), 0 -> 2 (3), 1 -> 3 (6), 1 -> 2 (2), 2 -> 4 (4), 3 -> 4 (1)
    Shortest path from 0 to 4: 0 -> 2 -> 4 (total: 7)
    """
    nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    edges = np.array([
        [0, 1, 5],
        [0, 2, 3],
        [1, 3, 6],
        [1, 2, 2],
        [2, 4, 4],
        [3, 4, 1],
    ], dtype=np.int64)
    return nodes, edges


@pytest.fixture
def small_dijkstra_graph():
    """Fixture: returns (nodes, edges) for a known 6-node directed graph.

    Graph: 0 -> 1 (4), 0 -> 2 (2), 1 -> 3 (5), 2 -> 1 (1), 2 -> 4 (3), 3 -> 5 (2), 4 -> 5 (4)
    Shortest path from 0 to 5: 0 -> 2 -> 1 -> 3 -> 5 (total: 10) or 0 -> 2 -> 4 -> 5 (total: 9)
    """
    nodes = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    edges = np.array([
        [0, 1, 4],
        [0, 2, 2],
        [1, 3, 5],
        [2, 1, 1],
        [2, 4, 3],
        [3, 5, 2],
        [4, 5, 4],
    ], dtype=np.int64)
    return nodes, edges


@pytest.fixture
def small_kruskal_graph():
    """Fixture: returns (nodes, edges) for a known 4-node undirected graph.

    Edges: 0-1 (4), 0-2 (3), 1-2 (1), 1-3 (2), 2-3 (5)
    MST should include: 1-2 (1), 1-3 (2), 0-2 (3) total weight: 6
    """
    nodes = np.array([0, 1, 2, 3], dtype=np.int64)
    edges = np.array([
        [0, 1, 4],
        [0, 2, 3],
        [1, 2, 1],
        [1, 3, 2],
        [2, 3, 5],
    ], dtype=np.int64)
    return nodes, edges


@pytest.fixture
def small_prim_graph():
    """Fixture: returns (nodes, edges) for a known 4-node undirected graph.

    Edges: 0-1 (4), 0-2 (3), 1-2 (1), 1-3 (2), 2-3 (5)
    MST should include: 1-2 (1), 1-3 (2), 0-2 (3) total weight: 6
    """
    nodes = np.array([0, 1, 2, 3], dtype=np.int64)
    edges = np.array([
        [0, 1, 4],
        [0, 2, 3],
        [1, 2, 1],
        [1, 3, 2],
        [2, 3, 5],
    ], dtype=np.int64)
    return nodes, edges


@pytest.fixture
def simple_maze():
    """Fixture: returns (maze, start, end) for a small 5x5 grid.

    0 = open, 1 = blocked.
    A simple open grid with walls around the perimeter except start and end.
    """
    maze = np.zeros((5, 5), dtype=np.int64)
    # Block some cells to create a non-trivial path
    maze[1, 1] = 1
    maze[1, 2] = 1
    maze[2, 2] = 1
    sx, sy = 0, 0
    ex, ey = 4, 4
    return maze, sx, sy, ex, ey


@pytest.fixture
def small_circuit_benchmark_file():
    """Fixture: creates a temporary fulladder circuit file for testing."""
    content = """inp a b cin
outp sum cout
xor a b x1
and a b c1
and x1 cin c2
xor x1 cin sum
or c1 c2 cout
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)