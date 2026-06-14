#!/usr/bin/env python3
"""Example: Dijkstra shortest path on a directed graph."""

import numpy as np
from optivlsi.dijkstra import dijkstra, dijkstra_networkx

# Small directed graph: 6 nodes, weighted edges
nodes = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
edges = np.array([
    [0, 1, 4],
    [0, 2, 2],
    [1, 3, 5],
    [2, 3, 1],
    [2, 4, 7],
    [3, 5, 3],
    [4, 5, 1],
], dtype=np.int64)

# Dijkstra from node 0 to node 5
status, path = dijkstra(nodes, edges, 0, 5)
print(f"Dijkstra path from 0 to 5: {path}")

# Also try NetworkX variant
status_nx, path_nx = dijkstra_networkx(nodes, edges, 0, 5)
print(f"NetworkX Dijkstra path:  {path_nx}")
print(f"Paths match: {path == path_nx}")