"""
Compiled-code digital circuit simulator.

Implements a compiled-code simulation approach for digital circuits.
Gate classes: AND, OR, NOT, NAND, NOR, XOR.
"""

from itertools import product
import pandas as pd


class xorgate:
    """custom class for XOR gate node"""

    def __init__(self, a, b, y):
        self.a = a
        self.b = b
        self.y = y

    def __repr__(self):
        return "XOR Gate"

    def ready(self, nets):
        """check if inputs are available"""
        return (nets[self.a] != 'u' and nets[self.b] != 'u')

    def evaluate(self, nets):
        """evaluate XOR gate node"""
        if ((nets[self.a] and (not nets[self.b])) or ((not nets[self.a]) and nets[self.b])):
            nets[self.y] = 1
        else:
            nets[self.y] = 0


class norgate:
    """custom class for NOR gate node"""

    def __init__(self, a, b, y):
        self.a = a
        self.b = b
        self.y = y

    def __repr__(self):
        return "NOR Gate"

    def ready(self, nets):
        """check if inputs are available"""
        return (nets[self.a] != 'u' and nets[self.b] != 'u')

    def evaluate(self, nets):
        """evaluate NOR gate node"""
        if (not (nets[self.a] or nets[self.b])):
            nets[self.y] = 1
        else:
            nets[self.y] = 0


class nandgate:
    """custom class for NAND gate node"""

    def __init__(self, a, b, y):
        self.a = a
        self.b = b
        self.y = y

    def __repr__(self):
        return "NAND Gate"

    def ready(self, nets):
        """check if inputs are available"""
        return (nets[self.a] != 'u' and nets[self.b] != 'u')

    def evaluate(self, nets):
        """evaluate NAND gate node"""
        if (not (nets[self.a] and nets[self.b])):
            nets[self.y] = 1
        else:
            nets[self.y] = 0


class notgate:
    """custom class for NOT gate node"""

    def __init__(self, a, y):
        self.a = a
        self.y = y

    def __repr__(self):
        return "NOT Gate"

    def ready(self, nets):
        """check if inputs are available"""
        return (nets[self.a] != 'u')

    def evaluate(self, nets):
        """evaluate NOT gate node"""
        if nets[self.a] == 1:
            nets[self.y] = 0
        else:
            nets[self.y] = 1


class andgate:
    """custom class for AND gate node"""

    def __init__(self, a, b, y):
        self.a = a
        self.b = b
        self.y = y

    def __repr__(self):
        return "AND Gate"

    def ready(self, nets):
        """check if inputs are available"""
        return (nets[self.a] != 'u' and nets[self.b] != 'u')

    def evaluate(self, nets):
        """evaluate AND gate node"""
        if (nets[self.a] and nets[self.b]):
            nets[self.y] = 1
        else:
            nets[self.y] = 0


class orgate:
    """custom class for OR gate node"""

    def __init__(self, a, b, y):
        self.a = a
        self.b = b
        self.y = y

    def __repr__(self):
        return "OR Gate"

    def ready(self, nets):
        """check if inputs are available"""
        return (nets[self.a] != 'u' and nets[self.b] != 'u')

    def evaluate(self, nets):
        """evaluate OR gate node"""
        if (nets[self.a] or nets[self.b]):
            nets[self.y] = 1
        else:
            nets[self.y] = 0


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
    """Parse a circuit netlist file and generate the circuit representation.

    Args:
        filename: Path to the circuit netlist file.

    Returns:
        Tuple of (nets, gates, inputs, outputs).
    """
    nets = {}
    gates = {}
    inputs = []
    outputs = []

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
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                gates[gate_number] = andgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'or'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                gates[gate_number] = orgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'inv'):
                for j in range(1, 3):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                gates[gate_number] = notgate(i[1], i[2])
                gate_number += 1
            elif (i[0] == 'nand'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                gates[gate_number] = nandgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'nor'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                gates[gate_number] = norgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'xor'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                gates[gate_number] = xorgate(i[1], i[2], i[3])
                gate_number += 1
            else:
                pass
    return (nets, gates, inputs, outputs)


def simulate(testvector, circuit):
    """Simulate a compiled-code circuit for a given test vector.

    Args:
        testvector: List of input values (0 or 1).
        circuit: Circuit tuple from generate_circuit().

    Returns:
        List of output values.
    """
    nets = circuit[0]
    gates = circuit[1]
    inputs = circuit[2]
    outputs = circuit[3]

    # assign the testvector to input nets
    for i in range(len(testvector)):
        nets[inputs[i]] = testvector[i]

    gate_done = [-1]*len(gates.keys())
    while (sum(gate_done) < 0):
        for i in gates.keys():
            if(gates[i].ready(nets)):
                gates[i].evaluate(nets)
                gate_done[i] = 0
            else:
                continue

    output_testvector = []

    for i in outputs:
        output_testvector.append(nets[i])

    return output_testvector