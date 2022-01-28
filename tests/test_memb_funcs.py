"""Tests for memb_func.py"""
# pylint: disable=missing-function-docstring, invalid-name
import numpy as np
import pytest
from fuzzy_machines.memb_funcs import Constant, Linear

# Singleton
def test_singleton():
    pass


# Constant
def test_constant_init():
    Constant(0, 10)
    Constant("-inf", "inf")
    with pytest.raises(ValueError):
        Constant("a", 1)
    with pytest.raises(ValueError):
        Constant(1, "b")


def test_constant_call():
    cons_func = Constant(0, 10)
    assert cons_func(0) == 1
    assert cons_func([0, 1, 2, 3]).all() == np.array([1, 1, 1, 1]).all()
    assert cons_func(-1) == 0
    assert cons_func([-1, 2, 3, 20, 20]).all() == np.array([0, 1, 1, 0, 0]).all()
    cons_func = Constant(-10, 10)
    assert cons_func([-11, -10, 2, 3, 20, 20]).all() == np.array([0, 1, 1, 1, 0, 0]).all()

    cons_func = Constant(0, 10)
    assert cons_func(0, 0.45) == 0.45
    assert cons_func([0, 1, 2, 3], 0.45).all() == np.array([0.45, 0.45, 0.45, 0.45]).all()
    assert cons_func(-1, 0.45) == 0
    cons_func = Constant(-10, 10)
    assert (
        cons_func([-11, -10, 2, 3, 20, 20], 0.45).all()
        == np.array([0, 0.45, 0.45, 0.45, 0, 0]).all()
    )

    with pytest.raises(ValueError):
        assert cons_func([-1, 2, 3, 20, 20], -0.3).all() == np.array([0, -0.3, -0.3, 0, 0]).all()


def test_constant_describe():
    cons = Constant(0, 10)
    x_arr, y_arr = cons.describe()
    assert x_arr.all() == np.array([0, 10]).all()
    assert y_arr.all() == np.array([1, 1]).all()

    cons = Constant("-inf", "inf")
    x_arr, y_arr = cons.describe()
    assert x_arr.size == y_arr.size
    assert x_arr.all() == np.array([float("-inf"), float("inf")]).all()
    assert y_arr.all() == np.array([1, 1]).all()


def test_constant_area():
    cons = Constant(0, 10)
    assert cons.area() == 10
    cons = Constant(-10, 10)
    assert cons.area() == 20
    cons = Constant("-inf", "inf")
    assert cons.area() == float("inf")

    cons = Constant(0, 10)
    assert cons.area(0.5) == 5
    cons = Constant(-10, 10)
    assert cons.area(0.5) == 10
    cons = Constant("-inf", "inf")
    assert cons.area(0.5) == float("inf")


# Linear
def test_linear_init():
    lin = Linear(0, 1)
    assert lin.min_v == 0
    assert lin.max_v == 1
    assert lin.slope == 1
    assert lin.b == 0

    lin = Linear(1, 0)
    assert lin.min_v == 0
    assert lin.max_v == 1
    assert lin.slope == -1
    assert lin.b == 1

    lin = Linear(-1, 1)
    assert lin.min_v == -1
    assert lin.max_v == 1
    assert lin.slope == 0.5
    assert lin.b == 0.5

    lin = Linear(1, -1)
    assert lin.min_v == -1
    assert lin.max_v == 1
    assert lin.slope == -0.5
    assert lin.b == 0.5

    with pytest.raises(TypeError):
        Linear("a", 1)

    with pytest.raises(ValueError):
        Linear(1, 1)

    with pytest.raises(ValueError):
        Linear("-inf", "+inf")


def test_linear_call():
    lin = Linear(0, 1)
    assert lin(0) == 0
    assert lin([0, 0.4, 0.6, 0.8, 1]).all() == np.array([0, 0.4, 0.6, 0.8, 1]).all()
    assert lin(-1) == 0
    assert lin([-1, 0.4, 0.6, 20, 20]).all() == np.array([0, 0.4, 0.6, 0, 0]).all()

    lin = Linear(0, 10)
    assert lin(0) == 0
    assert lin([0, 4, 6, 8, 10]).all() == np.array([0, 0.4, 0.6, 0.8, 1]).all()
    assert lin(-1) == 0
    assert lin([-1, 0.4, 6, 20, 20]).all() == np.array([0, 0.04, 0.6, 0, 0]).all()

    lin = Linear(10, 0)
    assert lin(0) == 1
    assert lin([0, 4, 6, 8, 10]).all() == np.array([1, 0.8, 0.6, 0.4, 0]).all()
    assert lin(-1) == 0
    assert lin([-1, 0.4, 6, 20, 20]).all() == np.array([0, 0.96, 0.4, 0, 0]).all()


def test_linear_describe():
    lin = Linear(0, 1)
    x_arr, y_arr = lin.describe()
    assert x_arr.all() == np.array([0, 1]).all()
    assert y_arr.all() == np.array([0, 1]).all()

    lin = Linear(-1, 1)
    x_arr, y_arr = lin.describe()
    assert x_arr.all() == np.array([-1, 1]).all()
    assert y_arr.all() == np.array([0, 1]).all()

    lin = Linear(10, 0)
    x_arr, y_arr = lin.describe()
    assert x_arr.all() == np.array([0, 10]).all()
    assert y_arr.all() == np.array([1, 0]).all()

    lin = Linear(10, -10)
    x_arr, y_arr = lin.describe()
    assert x_arr.all() == np.array([-10, 10]).all()
    assert y_arr.all() == np.array([1, 0]).all()


def test_linear_area():
    lin = Linear(0, 1)
    assert lin.area() == 0.5
    lin = Linear(1, 0)
    assert lin.area() == 0.5

    lin = Linear(0, 1)
    assert lin.area(0.5) == 0.375
    lin = Linear(1, 0)
    assert lin.area(0.5) == 0.375

    with pytest.raises(ValueError):
        lin.area(5)


# Triangular

# Trapezoidal
