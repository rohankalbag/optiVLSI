"""
Reduced Ordered Binary Decision Diagram (ROBDD) implementations.

Provides:
- build(): Build a BDD from a logical expression
- evaluate(): Evaluate a BDD for given variable assignments
- evaluate_numba(): Numba-accelerated evaluation
"""

import numpy as np

from .algorithms import (
    evaluate_expression,
    evaluate_expression2 as _evaluate_expression_numba,
    main as _build_bdd,
    main_numba as _build_bdd_numba,
    int_to_binary,
)


# Global state used by the original implementation
order = ""
inp = ""
ord1 = None
inp1 = None


def build(expression, variable_order):
    """Build a BDD from a logical expression.

    Args:
        expression: String logical expression (e.g., "a.b + ~a.b").
        variable_order: String of variable ordering (e.g., "ab").

    Returns:
        BDD representation as a list.
    """
    global order, inp, ord1, inp1
    order = variable_order
    inp = expression

    inp1_list = []
    for ch in inp:
        inp1_list.append(ord(ch))
    inp1 = np.array(inp1_list, dtype=np.int32)

    ord1_list = []
    for ch in order:
        ord1_list.append(ord(ch))
    ord1 = np.array(ord1_list, dtype=np.int32)

    # Set globals in the algorithms module directly
    import optivlsi.bdd.algorithms as bdd_algo
    bdd_algo.order = variable_order
    bdd_algo.inp = expression
    bdd_algo.ord1 = ord1
    bdd_algo.inp1 = inp1

    _build_bdd()
    return _build_bdd_numba()


__all__ = [
    "build",
    "evaluate_expression",
    "int_to_binary",
]