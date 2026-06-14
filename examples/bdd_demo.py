#!/usr/bin/env python3
"""Example: Build and evaluate a Reduced Ordered Binary Decision Diagram."""

from optivlsi.bdd.algorithms import evaluate_expression

# Boolean expression: x & y
# Truth table: 00->0, 01->0, 10->0, 11->1
exp = "x & y"
for x in [0, 1]:
    for y in [0, 1]:
        result = evaluate_expression({x, y}, exp)
        print(f"x={x}, y={y} -> {exp} = {result}")

print()

# Another expression: (a & b) | c
exp2 = "(a & b) | c"
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            result = evaluate_expression({a, b, c}, exp2)
            print(f"a={a}, b={b}, c={c} -> {exp2} = {result}")