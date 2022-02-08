"""Tests for memb_func.py"""
# pylint: disable=missing-function-docstring, invalid-name
from re import A
import numpy as np
import py
import pytest
from src.fuzz.memb_funcs import Constant, Linear, MembershipFunction, Singleton, Trapmf, Trimf

# BaseClass
def test_baseclass_init():
    mf = MembershipFunction(0, 1)
    assert mf.min_v == 0
    assert mf.max_v == 1


def test_baseclass_call():
    mf = MembershipFunction(0, 1)
    assert isinstance(mf(0, 1), np.ndarray)
    assert isinstance(mf([0, 1, 2], 1), np.ndarray)
    assert mf(0, 1) == 0
    assert mf(1, 1) == 1
    assert np.isnan(mf(2, 1))
    assert np.isnan(mf(-1, 1))


def test_baseclass_describe():
    mf = MembershipFunction(0, 1)
    with pytest.raises(NotImplementedError):
        mf.describe()


def test_baseclass_opt_integration():
    mf = MembershipFunction(0, 1)
    with pytest.raises(NotImplementedError):
        mf.optimal_integration(1)


def test_baseclass_naive_describe():
    mf = MembershipFunction(0, 1)
    x, y = mf.naive_describe(1, 0.1)
    assert x.all() == np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).all()
    assert y.all() == np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).all()

    x, y = mf.naive_describe(0.5, 0.1)
    assert x.all() == np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).all()
    assert y.all() == np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).all()


def test_baseclass_naive_integration():
    mf = MembershipFunction(0, 1)
    minv, maxv, yarr = mf.naive_integration(1, 0.1)[0]
    assert minv == 0
    assert maxv == 1
    assert yarr.all() == np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).all()


def test_baseclass_area():
    mf = MembershipFunction(0, 1)
    assert mf.area() == 0.5


# Singleton
def test_singleton_init():
    s = Singleton(2)
    assert s.min_v == float("-inf")
    assert s.max_v == float("+inf")


def test_singleton_call():
    s = Singleton(2)
    assert s(1) == 0
    assert s(2) == 1
    assert s(2.000001) == 0


def test_singleton_describe():
    s = Singleton(2)
    x, y = s.describe()
    assert x == 2
    assert y == 1


def test_singleton_area():
    s = Singleton(2)
    with pytest.raises(TypeError):
        s.area()


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
def test_triang_init():
    tri = Trimf(0, 1, 2)
    assert tri.max_v == 2
    assert tri.min_v == 0
    assert tri.up.min_v == 0
    assert tri.up.max_v == 1
    assert tri.down.min_v == 1
    assert tri.down.max_v == 2

    with pytest.raises(ValueError):
        Trimf("a", "b", "c")


def test_triang_call():
    tri = Trimf(0, 1, 2)
    assert tri(0) == 0
    assert tri(1) == 1
    assert tri(2) == 0
    assert tri([0, 1, 2]).all() == np.array([0, 1, 0]).all()
    assert (
        tri([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]).all()
        == np.array([0, 0, 0, 0.5, 1, 0.5, 0, 0, 0]).all()
    )
    assert (
        tri([-1, 0, 0.5, 0.8, 1, 1.2, 1.5, 2, 2.5, 3], 0.6).all()
        == np.array([0, 0, 0.5, 0.6, 0.6, 0.6, 0.5, 0, 0, 0]).all()
    )


def test_triang_describe():
    tri = Trimf(0, 1, 2)
    x, y = tri.describe()
    assert x.all() == np.array([0, 1, 2]).all()
    assert y.all() == np.array([0, 1, 0]).all()


def test_triang_area():
    tri = Trimf(0, 1, 2)
    assert tri.area() == 1
    tri = Trimf(0, 2, 4)
    assert tri.area() == 2


# Trapezoidal
def test_trap_init():
    trap = Trapmf(0, 1, 2, 3)
    assert trap.min_v == 0
    assert trap.max_v == 3
    assert trap.up.min_v == 0
    assert trap.up.max_v == 1
    assert trap.cons.min_v == 1
    assert trap.cons.max_v == 2
    assert trap.down.min_v == 2
    assert trap.down.max_v == 3

    with pytest.raises(ValueError):
        Trapmf("a", 2, 3, 4)


def test_trap_call():
    trap = Trapmf(0, 1, 2, 3)
    assert trap(0) == 0
    assert trap(1) == 1
    assert trap(2) == 1
    assert trap(3) == 0
    assert trap([0, 1, 2, 3]).all() == np.array([0, 1, 1, 0]).all()
    assert (
        trap([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]).all()
        == np.array([0, 0, 0, 0.5, 1, 1, 1, 0.5, 0]).all()
    )
    assert (
        trap([-1, 0, 0.5, 0.8, 1, 1.2, 1.5, 2, 2.5, 3], 0.6).all()
        == np.array([0, 0, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0]).all()
    )


def test_trap_describe():
    trap = Trapmf(0, 1, 2, 3)
    x, y = trap.describe()
    assert x.all() == np.array([0, 1, 2, 3]).all()
    assert y.all() == np.array([0, 1, 1, 0]).all()


def test_trap_area():
    trap = Trapmf(0, 1, 2, 3)
    assert trap.area() == 2
