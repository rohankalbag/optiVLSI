from itertools import product
import pandas as pd
import numba
import numpy as np

def get_input_output_nodes(filename):
    """returns the input and output nodes for the circuit netlist in ./filename as a tuple"""
    inputs = []
    outputs = []

    with open(filename, 'r') as t:
        gate_number = 0
        m = [x.split() for x in t.readlines()]
        for i in m:
            if(len(i) == 0):
                pass
            elif(i[0] == 'inp'):
                inputs = i[1:]
            elif(i[0] == 'outp'):
                outputs = i[1:]
            else:
                pass
    return (inputs, outputs)

def find_out(g, in1, in2):
    out=0
    g=int(g)
    in1=int(in1)
    in2=int(in2)
    if (g==0):
        if(in1==1):
            out=0
        else:
            out=1
    
    elif (g==1):
        if(in1 and in2):
            out=1
        else:
            out=0
    
    elif (g==2):
        if(in1 or in2):
            out=1
        else:
            out=0
    
    return out

def simulate(nets_old, filename, testvector):
    """simulates the circuit stored at ./filename for an input of testvector and returns output vector"""
    inputs = []
    outputs = []

    nets = numba.typed.Dict()
    nets=nets_old

    gates = numba.typed.Dict()
    nets_fanout=numba.typed.Dict.empty(key_type=numba.core.types.unicode_type, value_type=numba.core.types.int64[:],)

    # we will store all nets/nodes in hashable nets for fast lookup (make sure the nets are named differently and are not and, or, inv as gates named that way)
    # we will store all gates hashed by increasing numbers from 0 to no of gates - 1 for fast lookup  

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
                for j in range(1,4):
                    if(i[j] not in nets.keys()):
                        nets[i[j]] = 2
                # print(i[2])

                if(i[1] not in nets_fanout.keys()):
                    nets_fanout[i[1]] = np.empty(shape=(1,), dtype=int)
                    nets_fanout[i[1]] = np.asarray([gate_number])
                else:
                    nets_fanout[i[1]]=np.append(nets_fanout[i[1]],np.asarray([gate_number]))

                if(i[2] not in nets_fanout.keys()):
                    nets_fanout[i[2]] = np.empty(shape=(1,), dtype=int)
                    nets_fanout[i[2]] = np.asarray([gate_number])
                else:
                    nets_fanout[i[2]]=np.append(nets_fanout[i[2]],np.asarray([gate_number]))

                # instantiate the gate
                gates[gate_number] = np.asarray([1, i[1], i[2], i[3]])
                gate_number += 1
            elif (i[0] == 'or'):
                for j in range(1,4):
                    if(i[j] not in nets.keys()):
                        nets[i[j]] = 2
                # instantiate the gate

                if(i[1] not in nets_fanout.keys()):
                    nets_fanout[i[1]] = np.empty(shape=(1,), dtype=int)
                    nets_fanout[i[1]] = np.asarray([gate_number])
                else:
                    nets_fanout[i[1]]=np.append(nets_fanout[i[1]],np.asarray([gate_number]))

                if(i[2] not in nets_fanout.keys()):
                    nets_fanout[i[2]] = np.empty(shape=(1,), dtype=int)
                    nets_fanout[i[2]] = np.asarray([gate_number])
                else:
                    nets_fanout[i[2]]=np.append(nets_fanout[i[2]],np.asarray([gate_number]))

                gates[gate_number] = np.asarray([2, i[1], i[2], i[3]])
                gate_number += 1
            elif (i[0] == 'inv'):
                for j in range(1,3):
                    if (i[j] not in nets.keys()):
                        nets[i[j]] = 2
                # instantiate the gate

                if(i[1] not in nets_fanout.keys()):
                    nets_fanout[i[1]] = np.empty(shape=(1,), dtype=int)
                    nets_fanout[i[1]] = np.asarray([gate_number])
                else:
                    nets_fanout[i[1]]=np.append(nets_fanout[i[1]],np.asarray([gate_number]))

                gates[gate_number] = np.asarray([0, i[1], i[1], i[2]])
                gate_number += 1
            else:
                pass

    # assign the testvector to input nets
    gate_queue=[]
    for i in range(len(testvector)):
        if(nets[inputs[i]]!=testvector[i]):
            nets[inputs[i]] = testvector[i]
            for k in nets_fanout[inputs[i]]:
                if k not in gate_queue:
                    gate_queue=np.append(gate_queue, k)
    # print(nets)
    # print(gate_queue)

    while(len(gate_queue)>0):
            # print(nets)
            # print(gate_queue)
            # print(gates[int(gate_queue[0])][0])
            # print(nets[gates[int(gate_queue[0])][1]])
            # print(nets[gates[int(gate_queue[0])][2]])
            temp = find_out(gates[int(gate_queue[0])][0],nets[gates[int(gate_queue[0])][1]],nets[gates[int(gate_queue[0])][2]])
            # print(temp)
            # gate_queue=gate_queue[1:]
            # print(gates[int(gate_queue[0])][3])
            if nets[gates[int(gate_queue[0])][3]] != temp:
                nets[gates[int(gate_queue[0])][3]] = temp
                if gates[int(gate_queue[0])][3] in nets_fanout.keys():
                    for k in nets_fanout[gates[int(gate_queue[0])][3]]:
                        if k not in gate_queue:
                            gate_queue=np.append(gate_queue, k)
            
            gate_queue=gate_queue[1:]
 

    # create a list of hash_keys of all gates which are yet to be simulated
    # left_over_gates = list(range(len(gates.keys())))
    # while(len(left_over_gates) > 0):
    #     # keep performing until no left over gates
    #     completed = []

    #     for i in left_over_gates:
    #         #check if a gate is ready to be evaluated if so add it to completed
    #         if(gates[i].ready(nets)):
    #             completed.append(i)

    #     for i in completed:
    #         # evaluate all gates in completed and remove them from left over gates
    #         gates[i].evaluate(nets)
    #         left_over_gates.remove(i)


    output_testvector = []


    for i in outputs:
        # perform a hash lookup for the output nodes to get the output corresponding to that net
        output_testvector.append(nets[i])

    #return the output vector
    return nets, output_testvector


if __name__ == '__main__':
    inputs, outputs = get_input_output_nodes('circuit.txt')

    # make a dict to store truth table
    truthtable = {}

    for j in range(len(inputs)):
        truthtable[inputs[j]] = []

    for j in range(len(outputs)):
        truthtable[outputs[j]] = []

    # list performs cartesian product no of input number of times to provide inputs of generic n variable truth table in order    
    testvectors = list(product((0, 1), repeat=len(inputs)))
    nets_old = numba.typed.Dict()
    for i in testvectors:
        for j in range(len(i)):
            truthtable[inputs[j]].append(i[j])
        # simulate the ith truthtable entry
        nets_old, v = simulate(nets_old,'circuit.txt', i)

        for j in range(len(v)):
            truthtable[outputs[j]].append(v[j])

    # use pandas to convert to csv and store
    truthtable = pd.DataFrame(truthtable)
    truthtable.to_csv("truthtable.csv", index=False)
