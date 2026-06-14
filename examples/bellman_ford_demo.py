"""Example: Bellman-Ford shortest path algorithm."""

import numpy as np
from optivlsi.bellman_ford import bellman_ford, bellman_ford_networkx, bellman_ford_numba

nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
edges = np.array([
    [0, 1, 5],
    [0, 2, 3],
    [1, 3, 6],
    [1, 2, 2],
    [2, 4, 4],
    [3, 4, 1],
], dtype=np.int64)

status, path = bellman_ford(nodes, edges, 0, 4)
print(f"Pythonic result: status={status}, path={path}")

status, path = bellman_ford_networkx(nodes, edges, 0, 4)
print(f"NetworkX result: status={status}, path={path}")

status, path = bellman_ford_numba(nodes, edges, 0, 4)
print(f"Numba result: status={status}, path={path}")