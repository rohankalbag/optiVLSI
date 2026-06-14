"""
Compiled-code digital circuit simulator.

Provides:
- simulate(): Simulate a circuit for a given test vector
- generate_circuit(): Parse a circuit netlist file
- Gate classes: AND, OR, NOT, NAND, NOR, XOR
"""

from .simulator import (
    simulate as _simulate_cc,
    generate_circuit,
    get_input_output_nodes,
    andgate,
    orgate,
    notgate,
    nandgate,
    norgate,
    xorgate,
)


def simulate(testvector, circuit):
    """Simulate a compiled-code circuit for a given test vector.

    Args:
        testvector: List of input values (0 or 1).
        circuit: Circuit tuple from generate_circuit().

    Returns:
        List of output values.
    """
    return _simulate_cc(testvector, circuit)


__all__ = [
    "simulate",
    "generate_circuit",
    "get_input_output_nodes",
    "andgate",
    "orgate",
    "notgate",
    "nandgate",
    "norgate",
    "xorgate",
]