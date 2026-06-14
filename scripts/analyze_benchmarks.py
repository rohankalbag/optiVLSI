#!/usr/bin/env python3
"""Analyze benchmark results and compare with research paper methodology.

Usage:
    pytest tests/test_benchmarks.py --benchmark-json=benchmark_data.json
    python scripts/analyze_benchmarks.py benchmark_data.json
"""

import json
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_benchmarks.py <benchmark_json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    # Group by algorithm and size
    size_data = defaultdict(lambda: defaultdict(dict))

    for b in data['benchmarks']:
        name = b['name']
        mean_us = b['stats']['mean'] * 1e6
        # Parse name like "test_benchmark_bellman_ford_pythonic[50]"
        base = name.split('[')[0] if '[' in name else name
        parts = base.replace('test_benchmark_', '').split('_')
        variant = parts[-1]
        algo = '_'.join(parts[:-1])
        size_str = name.split('[')[1].rstrip(']') if '[' in name else '0'
        size_data[algo][size_str][variant] = mean_us

    total_path = 0
    print("=" * 80)
    print("BENCHMARK RESULTS vs RESEARCH PAPER (OptiVLSI.pdf)")
    print("=" * 80)
    print()
    print("Paper methodology used:")
    print("  Bellman-Ford: p=0.5, w=[5,15], sizes 10-200")
    print("  Dijkstra:     p=0.3, w=[5,15], sizes 10-175")
    print("  Kruskal/Prim: p=0.3, w=[5,15], sizes 10-500")
    print("  Lee:          maze sizes 2-500")
    print()

    for algo in sorted(size_data.keys()):
        print(f"{'─' * 70}")
        display_name = algo.replace('_', ' ').title()
        print(f"  {display_name}")
        print(f"{'─' * 70}")

        for size in sorted(size_data[algo].keys(), key=lambda x: int(x)):
            v = size_data[algo][size]
            py = v.get('pythonic')
            nx = v.get('networkx')
            nb = v.get('numba')

            times = []
            if py is not None:
                times.append(('Pythonic', py))
            if nx is not None:
                times.append(('NetworkX', nx))
            if nb is not None:
                times.append(('Numba', nb))

            if not times:
                continue

            # Sort by time (fastest first)
            times.sort(key=lambda x: x[1])
            fastest_time = times[0][1]

            print(f"\n    Size {size:>3}:")
            for t_name, t_val in times:
                ratio = t_val / fastest_time if fastest_time > 0 else 1
                marker = "★" if ratio == 1.0 else " "
                print(f"      {marker} {t_name:<10s}: {t_val:>10.2f} μs  ({ratio:.2f}x vs fastest)")

            # Paper expects: Numba >> Pythonic (10-100x speedup at larger sizes)
            if py is not None and nb is not None and nb > 0:
                speedup = py / nb
                status = "✓" if speedup > 1.5 else "⚠"
                size_i = int(size)
                if size_i >= 100:
                    expected = "Paper expects 10-100x speedup at large sizes"
                    match = "MATCHES" if speedup > 5 else "BELOW expected"
                elif size_i >= 50:
                    expected = "Paper expects 2-10x speedup at medium sizes"
                    match = "MATCHES" if speedup > 1.5 else "BELOW expected"
                else:
                    expected = "Small size, overhead dominates"
                    match = "acceptable"
                print(f"      {status} Py/Nb speedup: {speedup:.1f}x  ({match} — {expected})")

            if nx is not None and nb is not None and nb > 0:
                speedup_nx = nx / nb
                size_i = int(size)
                if size_i >= 100:
                    status2 = "✓" if speedup_nx > 5 else "⚠"
                elif size_i >= 50:
                    status2 = "✓" if speedup_nx > 1.5 else "⚠"
                else:
                    status2 = "—"
                print(f"      {status2} NX/Nb speedup: {speedup_nx:.1f}x")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    total_algo = len(size_data)
    total_sizes = sum(len(sizes) for sizes in size_data.values())
    print(f"  Algorithms tested: {total_algo}")
    print(f"  Size configurations: {total_sizes}")
    print(f"  Total benchmarks: {len(data['benchmarks'])}")
    print()
    print("  Note: pytest-benchmark measures micro-benchmarks on CI-grade")
    print("  hardware. The paper used automan on different hardware with")
    print("  warm caches. Absolute times differ, but the relative")
    print("  performance ordering (Numba > Pythonic ~ NetworkX) should")
    print("  be consistent at larger problem sizes.")
    print()


if __name__ == "__main__":
    main()