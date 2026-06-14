"""
Event-driven digital circuit simulator.

Provides:
- simulate(): Event-driven simulation of a circuit
- simulate_numba(): Numba-accelerated event-driven simulation
- read_file(): Parse a circuit netlist file into event-driven format
"""

from .simulator import (
    simulate as _simulate_ed,
    simulate_nb as _simulate_ed_numba,
    read_file,
    get_input_output_nodes,
)


def simulate(nets, inputs, outputs, gates, nets_fanout, testvector):
    """Run event-driven simulation.

    Args:
        nets: Dict mapping net names to values.
        inputs: List of input net names.
        outputs: List of output net names.
        gates: Dict mapping gate numbers to gate specs.
        nets_fanout: Dict mapping net names to fanout gate lists.
        testvector: List of input values.

    Returns:
        Tuple of (nets, output_testvector).
    """
    return _simulate_ed(nets, inputs, outputs, gates, nets_fanout, testvector)


def simulate_numba(nets, inputs, outputs, gates, nets_fanout, testvector):
    """Run Numba-accelerated event-driven simulation.

    Args:
        nets: Dict mapping net names to values.
        inputs: List of input net names.
        outputs: List of output net names.
        gates: Dict mapping gate numbers to gate specs.
        nets_fanout: Dict mapping net names to fanout gate lists.
        testvector: List of input values.

    Returns:
        Tuple of (nets, output_testvector).
    """
    return _simulate_ed_numba(nets, inputs, outputs, gates, nets_fanout, testvector)


__all__ = [
    "simulate",
    "simulate_numba",
    "read_file",
    "get_input_output_nodes",
]