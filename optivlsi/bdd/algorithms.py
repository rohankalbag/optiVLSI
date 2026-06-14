"""
Reduced Ordered Binary Decision Diagram (ROBDD) implementations.

Provides BDD construction and evaluation for logical expressions.
"""

import numpy as np
import numba
import time


# Global state for variable ordering and expression
order = ""
inp = ""
ord1 = None
inp1 = None


def evaluate_expression(x, exp):
    """Evaluate a logical expression for given variable assignments (Python).

    Args:
        x: Array of variable assignments (0 or 1) in 'order' sequence.
        exp: String logical expression (e.g., "a.b + ~a.b").

    Returns:
        Integer result (0 or 1).
    """
    opr = exp.find('+')
    if opr > -1:
        return (evaluate_expression(x, exp[:opr]) or evaluate_expression(x, exp[opr+1:]))
    else:
        out = 1
        if(len(exp) == 1):
            return x[order.find(exp[0])]
        if(exp[0] == '~'):
            out = out and not(x[order.find(exp[1])])
        else:
            out = out and (x[order.find(exp[0])])
        for i in range(1, len(exp)-1):
            if(exp[i] == '~'):
                out = out and not(x[order.find(exp[i+1])])
            elif(exp[i] == '.'):
                out = out and (x[order.find(exp[i+1])])
        return out


@numba.njit
def evaluate_expression2(x, exp):
    """Evaluate a logical expression using Numba (byte-code encoded).

    Args:
        x: Array of variable assignments.
        exp: Numpy array of ASCII byte codes for the expression.

    Returns:
        Integer result (0 or 1).
    """
    while(len(exp) > 0):
        opr = np.where(exp == ord('+'))[0]
        if len(opr) > 0 and opr[0] > -1:
            opr = opr[0]
            out = np.ones(shape=(1,), dtype=np.int32)
            if(len(exp[:opr]) == 1):
                out = x[np.where(ord1 == (exp[0]))]
                if out == 1:
                    return 1
            if(exp[0] == ord('~')):
                out = out and (np.int32(1) - (x[np.where(ord1 == (exp[1]))[0]]))
            else:
                out = out and (x[np.where(ord1 == (exp[0]))[0]])
            for i in range(1, len(exp[:opr])-1):
                if(exp[i] == ord('~')):
                    out = out and (np.int32(1) - (x[np.where(ord1 == (exp[i+1]))[0]]))
                elif(exp[i] == ord('.')):
                    out = out and (x[np.where(ord1 == (exp[i+1]))[0]])

            if out == 1:
                return 1
            else:
                exp = exp[opr+1:]
        else:
            out = np.ones(shape=(1,), dtype=np.int32)
            if (len(exp) == 1):
                return x[np.where(ord1 == (exp[0]))[0]][0]
            if (exp[0] == ord('~')):
                out = out and (np.int32(1) - (x[np.where(ord1 == (exp[1]))[0]]))
            else:
                out = out and (x[np.where(ord1 == (exp[0]))[0]])
            for i in range(1, len(exp)-1):
                if (exp[i] == ord('~')):
                    out = out and (np.int32(1) - (x[np.where(ord1 == (exp[i+1]))[0]]))
                elif (exp[i] == ord('.')):
                    out = out and (x[np.where(ord1 == (exp[i+1]))[0]])
            return out[0]


@numba.njit
def int_to_binary_numba(integer):
    """Convert integer to binary array for Numba.

    Args:
        integer: Integer to convert.

    Returns:
        Binary array of length len(order).
    """
    binary_string = np.zeros(shape=len(order))
    j = len(order)
    while (integer > 0):
        j = j - 1
        digit = integer % 2
        binary_string[j] = digit
        integer = integer // 2
    return binary_string


def int_to_binary(integer):
    """Convert integer to binary array.

    Args:
        integer: Integer to convert.

    Returns:
        Binary array of length len(order).
    """
    binary_string = np.zeros(shape=len(order))
    j = len(order)
    while (integer > 0):
        j = j - 1
        digit = integer % 2
        binary_string[j] = digit
        integer = integer // 2
    return binary_string


@numba.njit
def main_numba():
    """Build BDD tree using Numba-accelerated evaluation.

    Returns:
        List representing the BDD tree.
    """
    tree = []
    for i in range(len(ord1)):
        for j in range(2**(i) - 1, 2**(i+1) - 1):
            tree.append(ord1[i])
            height = i + 1

    j = 0
    for i in range(2**(height) - 1, 2**(height+1) - 1):
        x1 = np.asarray(int_to_binary_numba(j), dtype=np.int32)
        tree.append(int(evaluate_expression2(x1, inp1)))
        j = j + 1
    return tree


def main():
    """Build BDD tree using Python evaluation."""
    tree = []
    if len(order) == 0:
        return tree
    height = len(order)
    for i in range(len(order)):
        for j in range(2**(i) - 1, 2**(i+1) - 1):
            tree.append(order[i])

    j = 0
    for i in range(2**(height) - 1, 2**(height+1) - 1):
        x1 = np.asarray(int_to_binary(j), dtype=np.int32)
        tree.append(int(evaluate_expression(x1, inp)))
        j = j + 1
    return tree
