"""Benchmark tests for optiVLSI algorithms using pytest-benchmark.

These benchmarks measure small-scale performance for regression detection.
They are informational and do not fail CI on performance regression.
"""

import numpy as np
import pytest

# Graph algorithm benchmarks
from optivlsi.bellman_ford import bellman_ford, bellman_ford_networkx, bellman_ford_numba
from optivlsi.dijkstra import dijkstra, dijkstra_networkx, dijkstra_numba
from optivlsi.kruskal import kruskal, kruskal_networkx, kruskal_numba
from optivlsi.prim import prim, prim_networkx, prim_numba
from optivlsi.lee import lee, lee_networkx, lee_numba


# ---- Fixtures ----

@pytest.fixture(scope="module")
def bf_graph():
    """Small Bellman-Ford benchmark graph."""
    nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    edges = np.array([
        [0, 1, 5], [0, 2, 3], [1, 3, 6],
        [1, 2, 2], [2, 4, 4], [3, 4, 1],
    ], dtype=np.int64)
    return nodes, edges


@pytest.fixture(scope="module")
def dijkstra_graph():
    """Small Dijkstra benchmark graph."""
    nodes = np.array(list(range(10)), dtype=np.int64)
    edges = np.array([
        [0, 1, 4], [0, 2, 2], [0, 3, 7],
        [1, 4, 5], [2, 4, 3], [2, 5, 6],
        [3, 6, 1], [4, 7, 2], [5, 7, 4],
        [5, 8, 3], [6, 8, 5], [7, 9, 1],
        [8, 9, 2],
    ], dtype=np.int64)
    return nodes, edges


@pytest.fixture(scope="module")
def mst_graph():
    """Small MST benchmark graph."""
    nodes = np.array([0, 1, 2, 3], dtype=np.int64)
    edges = np.array([
        [0, 1, 4], [0, 2, 3], [1, 2, 1],
        [1, 3, 2], [2, 3, 5],
    ], dtype=np.int64)
    return nodes, edges


@pytest.fixture(scope="module")
def maze():
    """Small maze for Lee benchmark."""
    maze = np.zeros((20, 20), dtype=np.int64)
    maze[5:15, 5:15] = 1  # Block central region
    # Create a narrow corridor
    maze[10, 5:10] = 0
    return maze, 0, 0, 19, 19


# ---- Bellman-Ford Benchmarks ----

def test_benchmark_bellman_ford(benchmark, bf_graph):
    nodes, edges = bf_graph
    result = benchmark(bellman_ford, nodes, edges, 0, 4)
    assert result[0] == 1


def test_benchmark_bellman_ford_networkx(benchmark, bf_graph):
    nodes, edges = bf_graph
    result = benchmark(bellman_ford_networkx, nodes, edges, 0, 4)
    assert result[0] == 1


def test_benchmark_bellman_ford_numba(benchmark, bf_graph):
    nodes, edges = bf_graph
    result = benchmark(bellman_ford_numba, nodes, edges, 0, 4)
    assert result[0] == 1


# ---- Dijkstra Benchmarks ----

def test_benchmark_dijkstra(benchmark, dijkstra_graph):
    nodes, edges = dijkstra_graph
    result = benchmark(dijkstra, nodes, edges, 0, 9)
    assert result[0] == 1


def test_benchmark_dijkstra_networkx(benchmark, dijkstra_graph):
    nodes, edges = dijkstra_graph
    result = benchmark(dijkstra_networkx, nodes, edges, 0, 9)
    assert result[0] == 1


def test_benchmark_dijkstra_numba(benchmark, dijkstra_graph):
    nodes, edges = dijkstra_graph
    result = benchmark(dijkstra_numba, nodes, edges, 0, 9)
    assert result[0] == 1


# ---- Kruskal Benchmarks ----

def test_benchmark_kruskal(benchmark, mst_graph):
    nodes, edges = mst_graph
    result = benchmark(kruskal, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_kruskal_networkx(benchmark, mst_graph):
    nodes, edges = mst_graph
    result = benchmark(kruskal_networkx, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_kruskal_numba(benchmark, mst_graph):
    nodes, edges = mst_graph
    result = benchmark(kruskal_numba, nodes, edges)
    assert len(result[1]) > 0


# ---- Prim Benchmarks ----

def test_benchmark_prim(benchmark, mst_graph):
    nodes, edges = mst_graph
    result = benchmark(prim, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_prim_networkx(benchmark, mst_graph):
    nodes, edges = mst_graph
    result = benchmark(prim_networkx, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_prim_numba(benchmark, mst_graph):
    nodes, edges = mst_graph
    result = benchmark(prim_numba, nodes, edges)
    assert len(result[1]) > 0


# ---- Lee Algorithm Benchmarks ----

def test_benchmark_lee(benchmark, maze):
    m, sx, sy, ex, ey = maze
    distance = benchmark(lee, m, sx, sy, ex, ey)
    assert distance > 0


def test_benchmark_lee_networkx(benchmark, maze):
    m, sx, sy, ex, ey = maze
    distance = benchmark(lee_networkx, m, sx, sy, ex, ey)
    assert distance > 0


def test_benchmark_lee_numba(benchmark, maze):
    m, sx, sy, ex, ey = maze
    result = benchmark(lee_numba, m, sx, sy, ex, ey)
    assert result[0] > 0