#!/usr/bin/env python3
"""
Generate a static HTML benchmark report from pytest-benchmark JSON output.

Usage:
    pytest tests/test_benchmarks.py --benchmark-json=benchmark_data.json
    python scripts/generate_benchmark_report.py benchmark_data.json docs/benchmark_report.html
"""

import json
import sys
from pathlib import Path


def load_benchmark_data(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def generate_html(data: dict) -> str:
    benchmarks = data.get("benchmarks", [])
    machine_info = data.get("machine_info", {})
    commit_info = data.get("commit_info", {})

    # Group by algorithm (extract algorithm name from test name)
    algorithms = {}
    for b in benchmarks:
        name = b["name"]
        # Parse name like "test_benchmark_bellman_ford" -> "Bellman-Ford"
        parts = name.replace("test_benchmark_", "").split("_")
        algo_parts = []
        variant = ""
        for part in parts:
            if part in ("pythonic", "networkx", "numba", "nx"):
                variant = part
            else:
                algo_parts.append(part)
        algo_name = " ".join(algo_parts).title()
        if not algo_name:
            algo_name = name

        # Determine variant display name
        variant_display = {
            "": "Pythonic",
            "pythonic": "Pythonic",
            "nx": "NetworkX",
            "networkx": "NetworkX",
            "numba": "Numba",
        }.get(variant, variant or "Pythonic")

        if algo_name not in algorithms:
            algorithms[algo_name] = {}
        algorithms[algo_name][variant_display] = b["stats"]

    # HTML template
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
        h2 {{ color: #f0f6fc; margin: 2rem 0 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
        .meta {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }}
        .meta span {{ margin-right: 1.5rem; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; background: #161b22; border-radius: 6px; overflow: hidden; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ background: #1c2128; color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
        td {{ font-size: 0.875rem; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: #1c2128; }}
        .fastest {{ color: #3fb950; font-weight: 600; }}
        .slowest {{ color: #f85149; }}
        .bar-container {{ display: flex; align-items: center; gap: 0.5rem; }}
        .bar {{ height: 20px; border-radius: 4px; min-width: 4px; transition: width 0.3s; }}
        .bar.fastest {{ background: #3fb950; }}
        .bar.mid {{ background: #d29922; }}
        .bar.slowest {{ background: #f85149; }}
        .value {{ min-width: 80px; text-align: right; }}
        .footer {{ margin-top: 2rem; color: #8b949e; font-size: 0.8rem; text-align: center; border-top: 1px solid #30363d; padding-top: 1rem; }}
        .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
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
        <span>⏱️ {len(benchmarks)} benchmarks</span>
    </div>

    <div class="summary-cards">
        <div class="card">
            <h3>Total Benchmarks</h3>
            <div class="number">{len(benchmarks)}</div>
        </div>
        <div class="card">
            <h3>Algorithms Tested</h3>
            <div class="number">{len(algorithms)}</div>
        </div>
        <div class="card">
            <h3>Fastest Mean</h3>
            <div class="number">{min(b['stats']['mean'] * 1e6 for b in benchmarks):.1f}μs</div>
        </div>
    </div>
"""

    for algo_name in sorted(algorithms.keys()):
        variants = algorithms[algo_name]
        if not variants:
            continue

        # Find the fastest variant for bar scaling
        times = [(v, s["mean"] * 1e6) for v, s in variants.items()]
        times_sorted = sorted(times, key=lambda x: x[1])
        max_time = max(t[1] for t in times) if times else 1
        fastest_name = times_sorted[0][0]

        html += f"<h2>{algo_name}</h2>\n"
        html += """<table>
            <thead>
                <tr>
                    <th>Variant</th>
                    <th>Mean (μs)</th>
                    <th>Min (μs)</th>
                    <th>Max (μs)</th>
                    <th>Median (μs)</th>
                    <th>Rounds</th>
                    <th>Comparison</th>
                </tr>
            </thead>
            <tbody>\n"""

        for variant_name, mean_time in sorted(times, key=lambda x: x[1]):
            stats_obj = variants[variant_name]
            mean_us = stats_obj["mean"] * 1e6
            min_us = stats_obj["min"] * 1e6
            max_us = stats_obj["max"] * 1e6
            median_us = stats_obj["median"] * 1e6
            rounds = stats_obj.get("rounds", "—")

            bar_class = "fastest" if variant_name == fastest_name else ("slowest" if variant_name == times_sorted[-1][0] else "mid")
            bar_width = max((mean_time / max_time) * 100, 5) if max_time > 0 else 5

            html += f"""<tr>
                <td><strong>{variant_name}</strong></td>
                <td class="value">{mean_us:.2f}</td>
                <td class="value">{min_us:.2f}</td>
                <td class="value">{max_us:.2f}</td>
                <td class="value">{median_us:.2f}</td>
                <td class="value">{rounds}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar {bar_class}" style="width: {bar_width:.1f}%"></div>
                        <span>× {mean_time / times_sorted[0][1]:.1f}</span>
                    </div>
                </td>
            </tr>\n"""

        html += "</tbody></table>\n"

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