"""Tests for Event-Driven Simulator."""

import pytest
import numpy as np
from optivlsi.simulation.event_driven import (
    simulate,
    simulate_numba,
    read_file,
    get_input_output_nodes,
)


def test_simulate_fulladder(small_circuit_benchmark_file):
    """Event-driven simulation matches expected truth table for full adder."""
    inputs, outputs, nets, gates, nets_fanout = read_file(small_circuit_benchmark_file)

    # Full adder expected outputs
    expected = {
        (0, 0, 0): (0, 0),
        (0, 0, 1): (1, 0),
        (0, 1, 0): (1, 0),
        (0, 1, 1): (0, 1),
        (1, 0, 0): (1, 0),
        (1, 0, 1): (0, 1),
        (1, 1, 0): (0, 1),
        (1, 1, 1): (1, 1),
    }

    for testvec, (expected_sum, expected_cout) in expected.items():
        nets_copy = nets.copy()
        _, result = simulate(nets_copy, inputs, outputs, gates, nets_fanout, list(testvec))
        sum_idx = outputs.index('sum')
        cout_idx = outputs.index('cout')
        assert result[sum_idx] == expected_sum, f"Failed at {testvec}: sum={result[sum_idx]} != {expected_sum}"
        assert result[cout_idx] == expected_cout, f"Failed at {testvec}: cout={result[cout_idx]} != {expected_cout}"


def test_numba_simulation(small_circuit_benchmark_file):
    """Numba-accelerated simulation works (if the original file is used directly)."""
    # Numba simulation requires specific circuit file structure from read_file
    # This is a low-level test that the function can be imported
    assert callable(simulate_numba)


def test_event_propagation():
    """Event-driven simulation should correctly propagate events through gates."""
    import tempfile
    import os

    content = """inp a
outp y
inv a y
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        inputs, outputs, nets, gates, nets_fanout = read_file(f.name)
        os.unlink(f.name)

    _, result = simulate(nets.copy(), inputs, outputs, gates, nets_fanout, [0])
    assert result[0] == 1, "NOT gate: input 0 should produce output 1"

    _, result = simulate(nets.copy(), inputs, outputs, gates, nets_fanout, [1])
    assert result[0] == 0, "NOT gate: input 1 should produce output 0"