# Benchmarks

## Overview

Each algorithm module includes benchmark infrastructure to measure and compare performance across three implementation variants:

1. **Pure Python** — Reference implementation using native Python data structures
2. **NetworkX** — Implementation using the NetworkX graph library
3. **Numba-Accelerated** — JIT-compiled implementation using Numba

## Running Benchmarks

### Automan Benchmarks (Original)

Each module directory contains an `automate.py` file using the automan library. Run from the original module directory:

```bash
cd bellman-ford && python3 automate.py
```

This generates timing and speedup plots in `manuscript/figures/`.

### pytest-benchmark Integration

Run benchmark tests integrated with the test suite:

```bash
pytest tests/test_benchmarks.py --benchmark-only
```

This measures small-scale performance and reports timing statistics.

## Results

Benchmark results from the original research are available in each module's `manuscript/figures/` directory:

| Module | Timing Plot | Speedup Plot |
|--------|-------------|--------------|
| Bellman-Ford | `bellman-ford/manuscript/figures/Bellman-Ford/timing.pdf` | `bellman-ford/manuscript/figures/Bellman-Ford/speedup.pdf` |
| Dijkstra | `dijkstra/manuscript/figures/Dijkstra/timing.pdf` | `dijkstra/manuscript/figures/Dijkstra/speedup.pdf` |
| Kruskal | `kruskal/manuscript/figures/Kruskal/timing.pdf` | `kruskal/manuscript/figures/Kruskal/speedup.pdf` |
| Prim | `prim/manuscript/figures/Prim/timing.pdf` | `prim/manuscript/figures/Prim/speedup.pdf` |
| Lee | `lee-algorithm/manuscript/figures/Lee/timing.pdf` | `lee-algorithm/manuscript/figures/Lee/speedup.pdf` |
| Compiled-Code Sim | `compiled-code-simulator/manuscript/figures/timing.pdf` | — |
| Event-Driven Sim | `event-driven-sim/manuscript/figures/timing.pdf` | — |
| ROBDD | `ROBDD/Timing.pdf` | `ROBDD/Speedup.pdf` |

## Key Findings

- Numba acceleration typically achieves 10–100x speedup over pure Python implementations
- NetworkX implementations provide a robust reference baseline
- Speedup increases with problem size due to reduced overhead amortization