"""
Numba-accelerated compiled-code digital circuit simulator.

Provides:
- simulate(): Numba-accelerated simulation of a circuit for a given test vector
- generate_circuit(): Parse a circuit netlist file into numba-compatible format
"""

from .simulator import (
    simulate as _simulate_ccn,
    generate_circuit,
    get_input_output_nodes,
)


def simulate(testvector, circuit):
    """Simulate a circuit using Numba-accelerated compiled-code simulation.

    Args:
        testvector: List of input values (0 or 1).
        circuit: Circuit tuple from generate_circuit().

    Returns:
        List of output values.
    """
    return _simulate_ccn(testvector, circuit)


__all__ = [
    "simulate",
    "generate_circuit",
    "get_input_output_nodes",
]