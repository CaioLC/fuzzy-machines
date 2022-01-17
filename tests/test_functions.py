"""Tests for `fuzzy_machines.functions` package."""
from fuzzy_machines.memb_funcs import Constant


def test_constant():
    const_func = Constant(.5)
    assert const_func.value == 0.5
    assert const_func(20) == 0.5

