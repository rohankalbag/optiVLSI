# API Reference

## optivlsi.bellman_ford

```python
bellman_ford(nodes, edges, src, end) -> tuple
bellman_ford_networkx(nodes, edges, src, end) -> tuple
bellman_ford_numba(nodes, edges, src, end) -> tuple
graph_to_numpy(graph) -> tuple
numpy_to_graph(nodes, edges) -> nx.DiGraph
```

## optivlsi.dijkstra

```python
dijkstra(nodes, edges, src, end) -> tuple
dijkstra_networkx(nodes, edges, src, end) -> tuple
dijkstra_numba(nodes, edges, src, end) -> tuple
graph_to_numpy(graph) -> tuple
numpy_to_graph(nodes, edges) -> nx.DiGraph
```

## optivlsi.kruskal

```python
kruskal(nodes, edges) -> tuple
kruskal_networkx(nodes, edges) -> tuple
kruskal_numba(nodes, edges) -> tuple
graph_to_numpy(graph) -> tuple
numpy_to_graph(nodes, edges) -> nx.Graph
```

## optivlsi.prim

```python
prim(nodes, edges) -> tuple
prim_networkx(nodes, edges) -> tuple
prim_numba(nodes, edges) -> tuple
graph_to_numpy(graph) -> tuple
numpy_to_graph(nodes, edges) -> nx.Graph
```

## optivlsi.lee

```python
lee(maze, sx, sy, ex, ey) -> int
lee_networkx(maze, sx, sy, ex, ey) -> int
lee_numba(maze, sx, sy, ex, ey) -> tuple
```

## optivlsi.simulation.compiled_code

```python
simulate(testvector, circuit) -> list
generate_circuit(filename) -> tuple
get_input_output_nodes(filename) -> tuple
```

## optivlsi.simulation.compiled_code_numba

```python
simulate(testvector, circuit) -> list
generate_circuit(filename) -> tuple
```

## optivlsi.simulation.event_driven

```python
simulate(nets, inputs, outputs, gates, nets_fanout, testvector) -> tuple
simulate_numba(nets, inputs, outputs, gates, nets_fanout, testvector) -> tuple
read_file(filename) -> tuple
get_input_output_nodes(filename) -> tuple
```

## optivlsi.bdd

```python
build(expression, variable_order) -> list
evaluate_expression(x, exp) -> int
int_to_binary(integer) -> np.array