#!/usr/bin/env python3
"""Example: Digital circuit simulation using compiled-code simulator."""

from optivlsi.simulation.compiled_code.simulator import (
    generate_circuit, simulate
)
from itertools import product

# Build a full adder circuit from benchmark file
circuit_path = "compiled-code-simulator/benchmarks/fulladder.txt"
circuit = generate_circuit(circuit_path)

# Get input/output nodes
from optivlsi.simulation.compiled_code.simulator import get_input_output_nodes
inputs, outputs = get_input_output_nodes(circuit_path)

print(f"Inputs: {inputs}, Outputs: {outputs}")

# Simulate all input combinations
for bits in product([0, 1], repeat=len(inputs)):
    testvector = {inputs[i]: bits[i] for i in range(len(inputs))}
    result = simulate(testvector, circuit)
    print(f"Input {bits} -> Output {result}")