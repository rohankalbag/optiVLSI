#!/usr/bin/env python3
"""
Generate a static HTML benchmark report from pytest-benchmark JSON output.

Groups by algorithm and size, showing all 3 variants side-by-side with
speedup ratios and visual bars.

Usage:
    pytest tests/test_benchmarks.py --benchmark-json=benchmark_data.json
    python scripts/generate_benchmark_report.py benchmark_data.json docs/benchmark_report.html
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict


def load_benchmark_data(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def parse_test_name(name: str):
    """Parse a test name like 'test_benchmark_bellman_ford_pythonic[50]'.

    Returns (algorithm_name, variant, size) or (None, None, None) on failure.
    """
    # Strip the pytest parametrize suffix like [50], [100], [175]
    m = re.match(r'^test_benchmark_(.+?)_(pythonic|networkx|numba)\[(\d+)\]$', name)
    if m:
        algo = m.group(1).replace('_', ' ').title()
        variant = m.group(2)
        size = int(m.group(3))
        return algo, variant, size
    return None, None, None


def variant_display(variant: str) -> str:
    return {"pythonic": "Pythonic", "networkx": "NetworkX", "numba": "Numba"}.get(variant, variant)


def generate_html(data: dict) -> str:
    benchmarks = data.get("benchmarks", [])
    machine_info = data.get("machine_info", {})
    commit_info = data.get("commit_info", {})

    # Organize: algorithms[algo_name][size][variant] = stats
    algorithms = defaultdict(lambda: defaultdict(dict))

    for b in benchmarks:
        algo, variant, size = parse_test_name(b["name"])
        if algo is None:
            continue
        # Also store the size param from the test for reference
        algorithms[algo][size][variant] = b["stats"]

    total_benchmarks = len(benchmarks)
    total_algos = len(algorithms)
    all_means = [b['stats']['mean'] * 1e6 for b in benchmarks]
    fastest_overall = min(all_means) if all_means else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>optiVLSI Benchmark Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
        h1 {{ color: #58a6ff; margin-bottom: 0.5rem; }}
        h2 {{ color: #f0f6fc; margin: 2rem 0 0.5rem; }}
        .algo-section {{ margin-bottom: 2.5rem; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
        .algo-header {{ background: #1c2128; padding: 1rem 1.5rem; border-bottom: 1px solid #30363d; }}
        .algo-header h2 {{ margin: 0; }}
        .size-table {{ border-bottom: 1px solid #21262d; }}
        .size-table:last-child {{ border-bottom: none; }}
        .size-title {{ padding: 0.75rem 1.5rem; background: #161b22; color: #8b949e; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .meta {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }}
        .meta span {{ margin-right: 1.5rem; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 0.65rem 1rem; text-align: left; border-bottom: 1px solid #21262d; }}
        th {{ background: #0d1117; color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; }}
        td {{ font-size: 0.85rem; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: #1c2128; }}
        .bar-container {{ display: flex; align-items: center; gap: 0.5rem; }}
        .bar {{ height: 18px; border-radius: 3px; min-width: 4px; transition: width 0.3s; }}
        .bar.green {{ background: #3fb950; }}
        .bar.yellow {{ background: #d29922; }}
        .bar.red {{ background: #f85149; }}
        .value {{ min-width: 72px; text-align: right; font-variant-numeric: tabular-nums; }}
        .footer {{ margin-top: 2rem; color: #8b949e; font-size: 0.8rem; text-align: center; border-top: 1px solid #30363d; padding-top: 1rem; }}
        .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
        .card {{ background: #161b22; border-radius: 6px; padding: 1.25rem; border: 1px solid #30363d; }}
        .card h3 {{ color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .card .number {{ color: #f0f6fc; font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }}
    </style>
</head>
<body>
    <h1>📊 optiVLSI Benchmark Report</h1>
    <div class="meta">
        <span>🖥️ {machine_info.get('platform', 'N/A')}</span>
        <span>🐍 Python {machine_info.get('python_version', 'N/A')}</span>
        <span>⏱️ {total_benchmarks} benchmarks</span>
    </div>

    <div class="summary-cards">
        <div class="card">
            <h3>Total Benchmarks</h3>
            <div class="number">{total_benchmarks}</div>
        </div>
        <div class="card">
            <h3>Algorithms Tested</h3>
            <div class="number">{total_algos}</div>
        </div>
        <div class="card">
            <h3>Fastest Mean</h3>
            <div class="number">{fastest_overall:.1f}μs</div>
        </div>
    </div>
"""

    for algo_name in sorted(algorithms.keys()):
        sizes = algorithms[algo_name]
        html += f'<div class="algo-section">\n'
        html += f'  <div class="algo-header"><h2>{algo_name}</h2></div>\n'

        for size in sorted(sizes.keys()):
            variants = sizes[size]
            if not variants:
                continue

            # Sort variants by mean time (fastest first)
            times = [(v, s["mean"] * 1e6) for v, s in variants.items()]
            times_sorted = sorted(times, key=lambda x: x[1])
            fastest_time = times_sorted[0][1] if times_sorted else 1
            slowest_time = times_sorted[-1][1] if times_sorted else 1

            html += f'  <div class="size-table">\n'
            html += f'    <div class="size-title">size = {size}</div>\n'
            html += """    <table>
        <thead>
            <tr>
                <th>Variant</th>
                <th>Mean (μs)</th>
                <th>Min (μs)</th>
                <th>Max (μs)</th>
                <th>Median (μs)</th>
                <th>Rounds</th>
                <th>vs Fastest</th>
            </tr>
        </thead>
        <tbody>\n"""

            for variant_name, mean_time in times_sorted:
                stats_obj = variants[variant_name]
                mean_us = stats_obj["mean"] * 1e6
                min_us = stats_obj["min"] * 1e6
                max_us = stats_obj["max"] * 1e6
                median_us = stats_obj["median"] * 1e6
                rounds = stats_obj.get("rounds", "—")
                ratio = mean_time / fastest_time if fastest_time > 0 else 1

                # Color: fastest=green, middle=yellow, slowest=red
                if ratio == 1.0:
                    bar_class = "green"
                elif variant_name == times_sorted[-1][0]:
                    bar_class = "red"
                else:
                    bar_class = "yellow"

                bar_width = max((mean_time / slowest_time) * 100, 5) if slowest_time > 0 else 5

                html += f"""            <tr>
                <td><strong>{variant_display(variant_name)}</strong></td>
                <td class="value">{mean_us:.2f}</td>
                <td class="value">{min_us:.2f}</td>
                <td class="value">{max_us:.2f}</td>
                <td class="value">{median_us:.2f}</td>
                <td class="value">{rounds}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar {bar_class}" style="width: {bar_width:.1f}%"></div>
                        <span>× {ratio:.1f}</span>
                    </div>
                </td>
            </tr>\n"""

            html += "        </tbody>\n    </table>\n"
            html += '  </div>\n'

        html += '</div>\n'

    # Speedup summary section
    html += """
    <h2 style="margin-top: 3rem;">⚡ Numba Speedup Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Algorithm</th>
                <th>Size</th>
                <th>Pythonic → Numba</th>
                <th>NetworkX → Numba</th>
            </tr>
        </thead>
        <tbody>
"""

    for algo_name in sorted(algorithms.keys()):
        sizes = algorithms[algo_name]
        for size in sorted(sizes.keys()):
            variants = sizes[size]
            py = variants.get("pythonic")
            nx = variants.get("networkx")
            nb = variants.get("numba")

            py_speedup = (py["mean"] / nb["mean"]) if py and nb else None
            nx_speedup = (nx["mean"] / nb["mean"]) if nx and nb else None

            html += f"""            <tr>
                <td>{algo_name}</td>
                <td>{size}</td>
                <td>{f'{py_speedup:.1f}x' if py_speedup else '—'}</td>
                <td>{f'{nx_speedup:.1f}x' if nx_speedup else '—'}</td>
            </tr>\n"""

    html += """        </tbody>
    </table>
"""

    html += f"""
    <div class="footer">
        Generated by optiVLSI benchmark suite | {commit_info.get('time', '')}
    </div>
</body>
</html>"""
    return html


def main():
    if len(sys.argv) < 3:
        print("Usage: generate_benchmark_report.py <benchmark_json> <output_html>")
        sys.exit(1)

    json_path = sys.argv[1]
    html_path = sys.argv[2]

    data = load_benchmark_data(json_path)
    html = generate_html(data)

    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    Path(html_path).write_text(html)
    print(f"Benchmark report written to {html_path}")


if __name__ == "__main__":
    main()