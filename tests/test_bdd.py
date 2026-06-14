"""Tests for ROBDD (Reduced Ordered Binary Decision Diagram)."""

import pytest
import numpy as np
from optivlsi.bdd import build, evaluate_expression, int_to_binary


def test_build_simple_bdd():
    """Build BDD for simple expression 'a.b' (AND gate)."""
    tree = build("a.b", "ab")
    assert tree is not None
    assert len(tree) > 0


def test_evaluate_expression():
    """evaluate_expression works correctly for AND/OR/NOT."""
    from optivlsi.bdd.algorithms import order
    # Temporarily set global order
    import optivlsi.bdd.algorithms as bdd_algo
    bdd_algo.order = "ab"

    assert evaluate_expression(np.array([0, 0], dtype=np.int32), "a.b") == 0
    assert evaluate_expression(np.array([1, 0], dtype=np.int32), "a.b") == 0
    assert evaluate_expression(np.array([1, 1], dtype=np.int32), "a.b") == 1
    assert evaluate_expression(np.array([1, 1], dtype=np.int32), "a+b") == 1
    assert evaluate_expression(np.array([0, 0], dtype=np.int32), "a+b") == 0
    assert evaluate_expression(np.array([0, 0], dtype=np.int32), "~a") == 1
    assert evaluate_expression(np.array([1, 0], dtype=np.int32), "~a") == 0

    bdd_algo.order = ""


def test_int_to_binary():
    """int_to_binary returns correct binary representation."""
    import optivlsi.bdd.algorithms as bdd_algo
    bdd_algo.order = "abc"

    result = int_to_binary(0)
    assert np.array_equal(result, [0, 0, 0]), f"Expected [0,0,0], got {result}"

    result = int_to_binary(5)
    assert np.array_equal(result, [1, 0, 1]), f"Expected [1,0,1], got {result}"

    result = int_to_binary(7)
    assert np.array_equal(result, [1, 1, 1]), f"Expected [1,1,1], got {result}"

    bdd_algo.order = ""


def test_build_xor():
    """Build BDD for XOR expression 'a.~b + ~a.b'."""
    tree = build("a.~b+~a.b", "ab")
    assert tree is not None
    assert len(tree) > 0