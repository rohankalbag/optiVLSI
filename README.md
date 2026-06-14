# optiVLSI

optiVLSI is a Python package that implements a collection of classic graph and circuit algorithms, including:

- Bellman‑Ford
- Dijkstra
- Prim
- Kruskal
- Lee (maze solver)
- BDD (binary decision diagrams)
- Simulation engines (compiled‑code and event‑driven)

The project has been refactored into a proper Python package with a modern `pyproject.toml`, type hints, comprehensive tests, documentation, and CI/CD pipelines.

## Quick Start

```bash
# Install the package
pip install optivlsi

# Run a quick demo
python -m optivlsi.lee.algorithms.lee_algorithm
```

## Documentation

- [API Reference](https://github.com/rohankalbag/optiVLSI/blob/main/docs/api.md)
- [Algorithms Overview](https://github.com/rohankalbag/optiVLSI/blob/main/docs/algorithms.md)
- [Benchmarks](https://github.com/rohankalbag/optiVLSI/blob/main/docs/benchmarks.md)

## Contributing

See the [CONTRIBUTING.md](https://github.com/rohankalbag/optiVLSI/blob/main/CONTRIBUTING.md) file for guidelines.

## Implemented Algorithms

### Graph Algorithms
| Algorithm | Package | Variants |
|-----------|---------|----------|
| Bellman-Ford Shortest Path | `optivlsi.bellman_ford` | Pythonic, NetworkX, Numba |
| Dijkstra Shortest Path | `optivlsi.dijkstra` | Pythonic, NetworkX, Numba |
| Kruskal Minimum Spanning Tree | `optivlsi.kruskal` | Pythonic (DSU), NetworkX, Numba |
| Prim Minimum Spanning Tree | `optivlsi.prim` | Pythonic, NetworkX, Numba |

### Routing
| Algorithm | Package | Variants |
|-----------|---------|----------|
| Lee Maze Routing | `optivlsi.lee` | Pythonic BFS, NetworkX, Numba |

### Digital Circuit Simulation
| Algorithm | Package | Variants |
|-----------|---------|----------|
| Compiled-Code Simulator | `optivlsi.simulation.compiled_code` | Gate classes, Numba |
| Event-Driven Simulator | `optivlsi.simulation.event_driven` | Event propagation, Numba |

### Binary Decision Diagrams
| Algorithm | Package | Variants |
|-----------|---------|----------|
| ROBDD | `optivlsi.bdd` | Python, Numba |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=optivlsi --cov-report=term

# Run benchmarks
pytest tests/test_benchmarks.py --benchmark-only
```

## Documentation

Full documentation is available in the [docs/](https://github.com/rohankalbag/optiVLSI/tree/main/docs) directory:

- [API Reference](https://github.com/rohankalbag/optiVLSI/blob/main/docs/api.md)
- [Algorithm Details](https://github.com/rohankalbag/optiVLSI/blob/main/docs/algorithms.md)

Detailed research paper: [OptiVLSI.pdf](https://github.com/rohankalbag/optiVLSI/blob/main/OptiVLSI.pdf)

## Benchmarking

Each algorithm module includes an `automate.py` file for automan-based benchmarking across various problem sizes. The package also provides pytest-benchmark integration for performance regression detection.

## Optimization Tools Used

- **Numba**: All algorithms have Numba-accelerated variants with JIT compilation
- **Automan**: Automated simulation and benchmarking infrastructure
- **NetworkX**: Reference implementations using standard graph library

## Project Structure

```
optivlsi/                   # Main package
├── bellman_ford/           # Bellman-Ford algorithm
├── dijkstra/               # Dijkstra's algorithm
├── kruskal/                # Kruskal's MST
├── prim/                   # Prim's MST
├── lee/                    # Lee maze routing
├── simulation/
│   ├── compiled_code/      # Compiled-code simulator
│   ├── compiled_code_numba/# Numba-accelerated variant
│   └── event_driven/       # Event-driven simulator
├── bdd/                    # ROBDD
└── utils/                  # Shared utilities
bellman-ford/               # Original standalone modules
dijkstra/                   # (preserved for reproducibility)
kruskal/                    #
prim/                       #
lee-algorithm/              #
compiled-code-simulator/    #
event-driven-sim/           #
ROBDD/                      #
```

## Original Research

This open-source codebase started as a course project for **AE6102 - Parallel Scientific Computing and Visualization** at IIT Bombay. The original standalone research modules are preserved in their respective directories.

## Collaborators

- Rohan Rajesh Kalbag
- Neeraj Prabhu

## License

MIT