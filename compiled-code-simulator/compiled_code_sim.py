from itertools import product
import pandas as pd
from argparse import ArgumentParser
import time

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="ccsim")
    parser.add_argument(
        '-c', '--circuit', help="circuit filename")
    parser.add_argument("-t", '--truthtable', help="save final truthtable")
    parser.add_argument('--b', action='store_true', help='print benchmark results')
    return parser.parse_args()


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
    """returns the input and output nodes for the circuit netlist in ./filename as a tuple"""
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
    # we will store all nets/nodes in hashable nets for fast lookup (make sure the nets are named differently and are not and, or, inv as gates named that way)
    # we will store all gates hashed by increasing numbers from 0 to no of gates - 1 for fast lookup

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
                # instantiate the gate
                gates[gate_number] = andgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'or'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                # instantiate the gate
                gates[gate_number] = orgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'inv'):
                for j in range(1, 3):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                # instantiate the gate
                gates[gate_number] = notgate(i[1], i[2])
                gate_number += 1
            elif (i[0] == 'nand'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                # instantiate the gate
                gates[gate_number] = nandgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'nor'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                # instantiate the gate
                gates[gate_number] = norgate(i[1], i[2], i[3])
                gate_number += 1
            elif (i[0] == 'xor'):
                for j in range(1, 4):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 'u'
                # instantiate the gate
                gates[gate_number] = xorgate(i[1], i[2], i[3])
                gate_number += 1
            else:
                pass
    return (nets, gates, inputs, outputs)


def simulate(testvector, circuit):
    """simulates the circuit stored at ./filename for an input of testvector and returns output vector"""
    
    nets = circuit[0]
    gates = circuit[1]
    inputs = circuit[2]
    outputs = circuit[3]

    # assign the testvector to input nets
    for i in range(len(testvector)):
        nets[inputs[i]] = testvector[i]

    # create a list of hash_keys of all gates which are yet to be simulated
    left_over_gates = list(range(len(gates.keys())))
    while (len(left_over_gates) > 0):
        # keep performing until no left over gates
        completed = []

        for i in left_over_gates:
            # check if a gate is ready to be evaluated if so add it to completed
            if (gates[i].ready(nets)):
                completed.append(i)

        for i in completed:
            # evaluate all gates in completed and remove them from left over gates
            gates[i].evaluate(nets)
            left_over_gates.remove(i)

    output_testvector = []

    for i in outputs:
        # perform a hash lookup for the output nodes to get the output corresponding to that net
        output_testvector.append(nets[i])

    # return the output vector
    return output_testvector


if __name__ == '__main__':
    args = command_line_fetcher()
    circuitfile = args.circuit
    truthtablefile = args.truthtable
    bench = args.b

    inputs, outputs = get_input_output_nodes(f"{circuitfile}.txt")

    # make a dict to store truth table
    truthtable = {}

    for j in range(len(inputs)):
        truthtable[inputs[j]] = []

    for j in range(len(outputs)):
        truthtable[outputs[j]] = []

    # list performs cartesian product no of input number of times to provide inputs of generic n variable truth table in order
    testvectors = list(product((0, 1), repeat=len(inputs)))
    circuit = generate_circuit(f"{circuitfile}.txt")

    time_bench = time.perf_counter()
    for i in testvectors:
        for j in range(len(i)):
            truthtable[inputs[j]].append(i[j])
        # simulate the ith truthtable entry
        v = simulate(i, circuit)

        for j in range(len(v)):
            truthtable[outputs[j]].append(v[j])
    time_bench = time.perf_counter() - time_bench

    # use pandas to convert to csv and store
    truthtable = pd.DataFrame(truthtable)
    truthtable.to_csv(f"{truthtablefile}.csv", index=False)

    if bench:
        print(time_bench)