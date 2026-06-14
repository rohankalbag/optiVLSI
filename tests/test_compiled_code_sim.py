"""Tests for Compiled-Code Simulator."""

import pytest
from itertools import product
from optivlsi.simulation.compiled_code import (
    simulate,
    generate_circuit,
    andgate,
    orgate,
    notgate,
    nandgate,
    norgate,
    xorgate,
)


def test_simulate_fulladder(small_circuit_benchmark_file):
    """Simulate a known circuit (fulladder) and verify output truth table."""
    circuit = generate_circuit(small_circuit_benchmark_file)
    inputs = circuit[2]
    outputs = circuit[3]

    assert inputs == ['a', 'b', 'cin']
    assert 'sum' in outputs
    assert 'cout' in outputs

    # Full adder truth table:
    # a b cin | sum cout
    # 0 0 0   | 0   0
    # 0 0 1   | 1   0
    # 0 1 0   | 1   0
    # 0 1 1   | 0   1
    # 1 0 0   | 1   0
    # 1 0 1   | 0   1
    # 1 1 0   | 0   1
    # 1 1 1   | 1   1

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
        result = simulate(list(testvec), circuit)
        sum_idx = outputs.index('sum')
        cout_idx = outputs.index('cout')
        assert result[sum_idx] == expected_sum, f"Failed at {testvec}: sum={result[sum_idx]} != {expected_sum}"
        assert result[cout_idx] == expected_cout, f"Failed at {testvec}: cout={result[cout_idx]} != {expected_cout}"


def test_simulate_and_gate(small_circuit_benchmark_file):
    """Single-gate circuits should work correctly."""
    import tempfile
    import os

    content = "inp a b\noutp y\nand a b y\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        circuit = generate_circuit(f.name)
        os.unlink(f.name)

    assert simulate([0, 0], circuit) == [0]
    assert simulate([0, 1], circuit) == [0]
    assert simulate([1, 0], circuit) == [0]
    assert simulate([1, 1], circuit) == [1]


def test_gate_classes():
    """Gate classes should evaluate correctly."""
    g = andgate('a', 'b', 'y')
    assert g.__repr__() == "AND Gate"
    nets = {'a': 1, 'b': 1, 'y': 'u'}
    assert g.ready(nets) is True
    g.evaluate(nets)
    assert nets['y'] == 1

    nets = {'a': 1, 'b': 0, 'y': 'u'}
    g2 = andgate('a', 'b', 'y')
    g2.evaluate(nets)
    assert nets['y'] == 0

    nets = {'a': 1, 'y': 'u'}
    g3 = notgate('a', 'y')
    g3.evaluate(nets)
    assert nets['y'] == 0