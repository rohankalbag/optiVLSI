# Algorithm Implementations

optiVLSI provides multiple implementations of each VLSI CAD algorithm, comparing pure Python, NetworkX, and Numba-accelerated variants.

## Graph Algorithms

### Bellman-Ford Shortest Path
- **`bellman_ford()`** — Pure Python implementation with O(V·E) time complexity
- **`bellman_ford_networkx()`** — NetworkX-based implementation
- **`bellman_ford_numba()`** — Numba-accelerated JIT-compiled implementation

### Dijkstra's Shortest Path
- **`dijkstra()`** — Pure Python implementation using heapq
- **`dijkstra_networkx()`** — NetworkX-based implementation
- **`dijkstra_numba()`** — Numba-accelerated JIT-compiled implementation

### Kruskal's Minimum Spanning Tree
- **`kruskal()`** — Pure Python implementation with Disjoint Set Union (DSU)
- **`kruskal_networkx()`** — NetworkX-based implementation
- **`kruskal_numba()`** — Numba-accelerated JIT-compiled implementation

### Prim's Minimum Spanning Tree
- **`prim()`** — Pure Python implementation using NumPy array operations
- **`prim_networkx()`** — NetworkX-based implementation
- **`prim_numba()`** — Numba-accelerated JIT-compiled implementation

## Maze Routing

### Lee Algorithm
- **`lee()`** — Pure Python BFS-based maze routing
- **`lee_networkx()`** — NetworkX BFS-based implementation
- **`lee_numba()`** — Numba-accelerated JIT-compiled BFS

## Digital Circuit Simulation

### Compiled-Code Simulator
- **`simulate()`** — Gate-level compiled-code simulation with custom gate classes
- **`simulate_numba()`** — Numba-accelerated variant using integer-encoded gate types

### Event-Driven Simulator
- **`simulate()`** — Event-driven simulation with fanout-based event propagation
- **`simulate_numba()`** — Numba-accelerated event-driven simulation

## Binary Decision Diagrams

### ROBDD
- **`build()`** — Build a Reduced Ordered Binary Decision Diagram from a logical expression
- **`evaluate_expression()`** — Evaluate a logical expression for given variable assignments