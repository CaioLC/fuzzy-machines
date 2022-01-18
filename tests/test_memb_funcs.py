"""Tests for memb_func.py"""
# pylint: disable=missing-function-docstring, invalid-name
from fuzzy_machines.memb_funcs import Constant


def test_constant():
    const_func = Constant(0.5)
    assert const_func.value == 0.5
    assert const_func(20) == 0.5
