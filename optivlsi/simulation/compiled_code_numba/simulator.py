"""
Numba-accelerated compiled-code digital circuit simulator.

Implements gate evaluation using integer-coded gate types for Numba compatibility.
Gate types: 0=AND, 1=OR, 2=NAND, 3=NOR, 4=XOR, 5=NOT
"""

from itertools import product
import pandas as pd
import numba
import numpy as np
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


def generate_circuit(filename):
    """Parse a circuit netlist file into Numba-compatible format.

    Args:
        filename: Path to the circuit netlist file.

    Returns:
        Tuple of (net_hashmap, nets, gates, gatetype, inputs, outputs, associated_nets).
    """
    net_number = 0
    net_hashmap = {}
    nets = []
    gates = []
    gatetype = []
    inputs = []
    outputs = []
    associated_nets = []

    with open(filename, 'r') as t:
        m = [x.split() for x in t.readlines()]
        for i in m:
            if (len(i) == 0):
                pass
            elif (i[0] == 'inp'):
                inputs = i[1:]
            elif (i[0] == 'outp'):
                outputs = i[1:]
            elif (i[0] == 'and'):
                corresponding_nets = []
                for j in range(1, 4):
                    if (i[j] not in net_hashmap.keys()):
                        net_hashmap[i[j]] = net_number
                        corresponding_nets.append(net_number)
                        nets.append(-1)
                        net_number += 1
                    else:
                        corresponding_nets.append(net_hashmap[i[j]])
                gates.append(-1)
                associated_nets.append(corresponding_nets)
                gatetype.append(0)
            elif (i[0] == 'or'):
                corresponding_nets = []
                for j in range(1, 4):
                    if (i[j] not in net_hashmap.keys()):
                        net_hashmap[i[j]] = net_number
                        corresponding_nets.append(net_number)
                        nets.append(-1)
                        net_number += 1
                    else:
                        corresponding_nets.append(net_hashmap[i[j]])
                gates.append(-1)
                associated_nets.append(corresponding_nets)
                gatetype.append(1)
            elif (i[0] == 'inv'):
                corresponding_nets = []
                for j in range(1, 3):
                    if (i[j] not in net_hashmap.keys()):
                        net_hashmap[i[j]] = net_number
                        corresponding_nets.append(net_number)
                        nets.append(-1)
                        net_number += 1
                    else:
                        corresponding_nets.append(net_hashmap[i[j]])
                gates.append(-1)
                corresponding_nets.append(69)  # dummy for uniformity
                associated_nets.append(corresponding_nets)
                gatetype.append(5)
            elif (i[0] == 'nand'):
                corresponding_nets = []
                for j in range(1, 4):
                    if (i[j] not in net_hashmap.keys()):
                        net_hashmap[i[j]] = net_number
                        corresponding_nets.append(net_number)
                        nets.append(-1)
                        net_number += 1
                    else:
                        corresponding_nets.append(net_hashmap[i[j]])
                gates.append(-1)
                associated_nets.append(corresponding_nets)
                gatetype.append(2)
            elif (i[0] == 'nor'):
                corresponding_nets = []
                for j in range(1, 4):
                    if (i[j] not in net_hashmap.keys()):
                        net_hashmap[i[j]] = net_number
                        corresponding_nets.append(net_number)
                        nets.append(-1)
                        net_number += 1
                    else:
                        corresponding_nets.append(net_hashmap[i[j]])
                gates.append(-1)
                associated_nets.append(corresponding_nets)
                gatetype.append(3)
            elif (i[0] == 'xor'):
                corresponding_nets = []
                for j in range(1, 4):
                    if (i[j] not in net_hashmap.keys()):
                        net_hashmap[i[j]] = net_number
                        corresponding_nets.append(net_number)
                        nets.append(-1)
                        net_number += 1
                    else:
                        corresponding_nets.append(net_hashmap[i[j]])
                gates.append(-1)
                associated_nets.append(corresponding_nets)
                gatetype.append(4)
            else:
                pass
    return (net_hashmap, nets, gates, gatetype, inputs, outputs, associated_nets)


@numba.njit
def compiled_code_accelerated_numba(nets, gates, gatetype, associated_nets):
    """Numba-accelerated compiled-code simulation kernel.

    Evaluates all gates repeatedly until all are evaluated (fixed-point iteration).
    """
    x = -1
    while(x < 0):
        for i, j in enumerate(gates):
            if(j == -1):
                if(gatetype[i] == 0):  # AND
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(nets[y[0]] == 1 and nets[y[1]] == 1):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 1):  # OR
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(nets[y[0]] == 1 or nets[y[1]] == 1):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 2):  # NAND
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(not(nets[y[0]] == 1 and nets[y[1]] == 1)):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 3):  # NOR
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(not(nets[y[0]] == 1 or nets[y[1]] == 1)):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 4):  # XOR
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if((nets[y[0]] == 1 and nets[y[1]] == 0) or (nets[y[0]] == 0 and nets[y[1]] == 1)):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                else:  # NOT (gatetype[i] == 5)
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0):
                        if(nets[y[0]] == 1):
                            nets[y[1]] = 0
                        else:
                            nets[y[1]] = 1
                        gates[i] = 0
                        x = np.sum(gates)
            else:
                continue


def simulate(testvector, circuit):
    """Simulate a circuit using Numba-accelerated compiled-code simulation.

    Args:
        testvector: List of input values (0 or 1).
        circuit: Circuit tuple from generate_circuit().

    Returns:
        List of output values.
    """
    net_hashmap = circuit[0]
    nets = np.array(circuit[1], dtype=np.int32)
    gates = np.array(circuit[2], dtype=np.int32)
    gatetype = np.array(circuit[3], dtype=np.int32)
    inputs = circuit[4]
    outputs = circuit[5]
    associated_nets = np.array(circuit[6], dtype=np.int32)

    for i in range(len(testvector)):
        nets[net_hashmap[inputs[i]]] = testvector[i]

    # dummy call for Numba JIT warm-up
    compiled_code_accelerated_numba(
        np.copy(nets), np.copy(gates), np.copy(gatetype), np.copy(associated_nets)
    )

    net_hashmap = circuit[0]
    nets = np.array(circuit[1], dtype=np.int32)
    gates = np.array(circuit[2], dtype=np.int32)
    gatetype = np.array(circuit[3], dtype=np.int32)
    inputs = circuit[4]
    outputs = circuit[5]
    associated_nets = np.array(circuit[6], dtype=np.int32)

    for i in range(len(testvector)):
        nets[net_hashmap[inputs[i]]] = testvector[i]

    compiled_code_accelerated_numba(nets, gates, gatetype, associated_nets)

    output_testvector = []

    for i in outputs:
        output_testvector.append(nets[net_hashmap[i]])

    return output_testvector