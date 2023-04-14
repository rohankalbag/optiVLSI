from itertools import product
import pandas as pd
from argparse import ArgumentParser
import numba
import numpy as np
import time

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="ccsim")
    parser.add_argument(
        '-c', '--circuit', help="circuit filename")
    parser.add_argument("-t", '--truthtable', help="save final truthtable")
    parser.add_argument('--b', action='store_true', help='print benchmark results')
    return parser.parse_args()

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
                # instantiate the gate
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
                # instantiate the gate
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
                # instantiate the gate
                gates.append(-1)
                corresponding_nets.append(69) #random value for uniformity in size like binary input gates
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
                # instantiate the gate
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

                # instantiate the gate
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
                # instantiate the gate
                gates.append(-1)
                associated_nets.append(corresponding_nets)
                gatetype.append(4)
            else:
                pass
    return (net_hashmap, nets, gates, gatetype, inputs, outputs, associated_nets)

@numba.njit
def compiled_code_accelerated_numba(nets, gates, gatetype, associated_nets):
    x = -1
    while(x < 0):
        for i,j in enumerate(gates):
            if(j == -1):
                if(gatetype[i] == 0):
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(nets[y[0]] == 1 and nets[y[1]] == 1):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 1):
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(nets[y[0]] == 1 or nets[y[1]] == 1):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 2):
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(not(nets[y[0]] == 1 and nets[y[1]] == 1)):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 3):
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if(not(nets[y[0]] == 1 or nets[y[1]] == 1)):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                elif(gatetype[i] == 4):
                    y = associated_nets[i]
                    if(nets[y[0]] >= 0 and nets[y[1]] >= 0):
                        if((nets[y[0]] == 1 and nets[y[1]] == 0) or (nets[y[0]] == 0 and nets[y[1]] == 1)):
                            nets[y[2]] = 1
                        else:
                            nets[y[2]] = 0
                        gates[i] = 0
                        x = np.sum(gates)
                else:
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
    """simulates the circuit stored at ./filename for an input of testvector and returns output vector"""
    
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

    #dummy call for numba
    for i in testvectors:
        for j in range(len(i)):
            truthtable[inputs[j]].append(i[j])
        # simulate the ith truthtable entry
        v = simulate(i, circuit)

        for j in range(len(v)):
            truthtable[outputs[j]].append(v[j])
    
    time_bench = time.perf_counter()
    #actual benchmark
    for i in testvectors:
        for j in range(len(i)):
            truthtable[inputs[j]].append(i[j])
        # simulate the ith truthtable entry
        v = simulate(i, circuit)

        for j in range(len(v)):
            truthtable[outputs[j]].append(v[j])
    time_bench = time.perf_counter() - time_bench

    if bench:
        print(time_bench)

    # use pandas to convert to csv and store
    truthtable = pd.DataFrame(truthtable)
    truthtable.to_csv(f"{truthtablefile}.csv", index=False)