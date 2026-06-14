"""
Event-driven digital circuit simulator.

Implements event-driven simulation where gates only evaluate when input events occur.
Provides both pure Python and Numba-accelerated variants.
"""

from itertools import product
import pandas as pd
import numba
import numpy as np
import warnings
import time


def get_input_output_nodes(filename):
    """Return the input and output nodes for a circuit netlist file.

    Args:
        filename: Path to the circuit netlist file.

    Returns:
        Tuple of (inputs, outputs) as lists of net names.
    """
    inputs = []
    outputs = []

    with open(filename, 'r') as t:
        m = [x.split() for x in t.readlines()]
        for i in m:
            if (len(i) == 0):
                pass
            elif (i[0] == 'inp'):
                inputs = i[1:]
            elif (i[0] == 'outp'):
                outputs = i[1:]
            else:
                pass
    return (inputs, outputs)


def read_file(filename):
    """Parse a circuit netlist file into event-driven simulator format.

    Args:
        filename: Path to the circuit netlist file.

    Returns:
        Tuple of (inputs, outputs, nets, gates, nets_fanout).
    """
    inputs = []
    outputs = []

    nets = {}
    gates = {}
    nets_fanout = {}

    with open(filename, 'r') as t:
        gate_number = 0
        m = [x.split() for x in t.readlines()]
        for i in m:
            if (len(i) == 0):
                pass
            elif (i[0] == 'inp'):
                inputs = i[1:]
            elif (i[0] == 'outp'):
                outputs = i[1:]
            elif (i[0] == 'and'):
                for j in range(1, 4):
                    if (i[j] not in nets):
                        nets[i[j]] = 2
                if (i[1] not in nets_fanout):
                    nets_fanout[i[1]] = [gate_number]
                else:
                    nets_fanout[i[1]].append(gate_number)
                if (i[2] not in nets_fanout):
                    nets_fanout[i[2]] = [gate_number]
                else:
                    nets_fanout[i[2]].append(gate_number)
                gates[gate_number] = [1, i[1], i[2], i[3]]
                gate_number += 1
            elif (i[0] == 'or'):
                for j in range(1, 4):
                    if (i[j] not in nets):
                        nets[i[j]] = 2
                if (i[1] not in nets_fanout):
                    nets_fanout[i[1]] = [gate_number]
                else:
                    nets_fanout[i[1]].append(gate_number)
                if (i[2] not in nets_fanout):
                    nets_fanout[i[2]] = [gate_number]
                else:
                    nets_fanout[i[2]].append(gate_number)
                gates[gate_number] = [2, i[1], i[2], i[3]]
                gate_number += 1
            elif (i[0] == 'inv'):
                for j in range(1, 3):
                    if (i[j] not in nets):
                        nets[i[j]] = 2
                if (i[1] not in nets_fanout):
                    nets_fanout[i[1]] = [gate_number]
                else:
                    nets_fanout[i[1]].append(gate_number)
                gates[gate_number] = [0, i[1], i[1], i[2]]
                gate_number += 1
            elif (i[0] == 'nand'):
                for j in range(1, 4):
                    if (i[j] not in nets):
                        nets[i[j]] = 2
                if (i[1] not in nets_fanout):
                    nets_fanout[i[1]] = [gate_number]
                else:
                    nets_fanout[i[1]].append(gate_number)
                if (i[2] not in nets_fanout):
                    nets_fanout[i[2]] = [gate_number]
                else:
                    nets_fanout[i[2]].append(gate_number)
                gates[gate_number] = [5, i[1], i[2], i[3]]
                gate_number += 1
            elif (i[0] == 'nor'):
                for j in range(1, 4):
                    if (i[j] not in nets):
                        nets[i[j]] = 2
                if (i[1] not in nets_fanout):
                    nets_fanout[i[1]] = [gate_number]
                else:
                    nets_fanout[i[1]].append(gate_number)
                if (i[2] not in nets_fanout):
                    nets_fanout[i[2]] = [gate_number]
                else:
                    nets_fanout[i[2]].append(gate_number)
                gates[gate_number] = [6, i[1], i[2], i[3]]
                gate_number += 1
            elif (i[0] == 'xor'):
                for j in range(1, 4):
                    if (i[j] not in nets):
                        nets[i[j]] = 2
                if (i[1] not in nets_fanout):
                    nets_fanout[i[1]] = [gate_number]
                else:
                    nets_fanout[i[1]].append(gate_number)
                if (i[2] not in nets_fanout):
                    nets_fanout[i[2]] = [gate_number]
                else:
                    nets_fanout[i[2]].append(gate_number)
                gates[gate_number] = [7, i[1], i[2], i[3]]
                gate_number += 1
            else:
                pass

    return inputs, outputs, nets, gates, nets_fanout


def find_out(g, in1, in2):
    """Evaluate gate output for a given gate type and inputs.

    Gate types: 0=NOT, 1=AND, 2=OR, 5=NAND, 6=NOR, 7=XOR.
    in2 is ignored for NOT gate (type 0).

    Args:
        g: Gate type.
        in1: First input value.
        in2: Second input value.

    Returns:
        Output value (0 or 1).
    """
    g = int(g)
    in1 = int(in1)
    in2 = int(in2)
    if g == 0:  # NOT
        return 0 if in1 == 1 else 1
    elif g == 1:  # AND
        return 1 if (in1 and in2) else 0
    elif g == 2:  # OR
        return 1 if (in1 or in2) else 0
    elif g == 5:  # NAND
        return 0 if (in1 and in2) else 1
    elif g == 6:  # NOR
        return 0 if (in1 or in2) else 1
    elif g == 7:  # XOR
        return 1 if (in1 != in2) else 0
    return 0


def simulate(nets_old, inputs, outputs, gates, nets_fanout, testvector):
    """Run event-driven simulation on a circuit.

    Args:
        nets_old: Dict mapping net names to values.
        inputs: List of input net names.
        outputs: List of output net names.
        gates: Dict mapping gate numbers to gate specs.
        nets_fanout: Dict mapping net names to fanout gate lists.
        testvector: List of input values.

    Returns:
        Tuple of (nets, output_testvector).
    """
    nets = nets_old
    gate_queue = []
    for i in range(len(testvector)):
        if nets[inputs[i]] != testvector[i]:
            nets[inputs[i]] = testvector[i]
            for k in nets_fanout[inputs[i]]:
                if k not in gate_queue:
                    gate_queue.append(k)

    while len(gate_queue) > 0:
        idx = int(gate_queue[0])
        g = gates[idx]
        temp = find_out(g[0], nets[g[1]], nets[g[2]])
        if nets[g[3]] != temp:
            nets[g[3]] = temp
            if g[3] in nets_fanout:
                for k in nets_fanout[g[3]]:
                    if k not in gate_queue:
                        gate_queue.append(k)
        gate_queue = gate_queue[1:]

    output_testvector = []
    for i in outputs:
        output_testvector.append(nets[i])

    return nets, output_testvector


@numba.njit
def find_out_nb(g, in1, in2):
    """Numba-compatible gate evaluation.

    Args:
        g: Gate type character code.
        in1: First input value.
        in2: Second input value.

    Returns:
        Output value (0 or 1).
    """
    out = 0
    g = ord(g) - 48
    in1 = int(in1)
    in2 = int(in2)
    if (g == 0):
        if (in1 == 1):
            out = 0
        else:
            out = 1
    elif (g == 1):
        if (in1 and in2):
            out = 1
        else:
            out = 0
    elif (g == 2):
        if (in1 or in2):
            out = 1
        else:
            out = 0
    return out


@numba.njit
def simulate_nb(nets_old, inputs, outputs, gates, nets_fanout, testvector):
    """Run Numba-accelerated event-driven simulation.

    Args:
        nets_old: Dict mapping net names to values.
        inputs: List of input net names.
        outputs: List of output net names.
        gates: Dict mapping gate numbers to gate specs.
        nets_fanout: Dict mapping net names to fanout gate lists.
        testvector: List of input values.

    Returns:
        Tuple of (nets, output_testvector).
    """
    nets = nets_old

    gate_queue = -np.ones(shape=1)
    for i in range(len(testvector)):
        if (nets[inputs[i]] != testvector[i]):
            nets[inputs[i]] = testvector[i]
            for k in nets_fanout[inputs[i]]:
                if k not in gate_queue:
                    gate_queue = np.append(gate_queue, k)

    gate_queue = gate_queue[1:]
    while (len(gate_queue) > 0):
        temp = find_out_nb(str(gates[int(gate_queue[0])][0]),
                           nets[str(gates[int(gate_queue[0])][1])],
                           nets[str(gates[int(gate_queue[0])][2])])
        if nets[str(gates[int(gate_queue[0])][3])] != temp:
            nets[str(gates[int(gate_queue[0])][3])] = temp
            temp2 = ord(str(gates[int(gate_queue[0])][3]))
            if temp2 in list(map(int, [ord(i) for i in nets_fanout.keys()])):
                for k in nets_fanout[str(gates[int(gate_queue[0])][3])]:
                    if k not in gate_queue:
                        gate_queue = np.append(gate_queue, k)

        gate_queue = gate_queue[1:]

    output_testvector = []
    for i in outputs:
        output_testvector.append(nets[i])

    return nets, output_testvector