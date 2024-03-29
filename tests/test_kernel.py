""" tests for kernel.py """
# pylint: disable=missing-function-docstring, invalid-name
from warnings import WarningMessage
import numpy as np
import pytest
from src.fuzz.kernel import Kernel
from src.fuzz.memb_funcs import MembershipFunction, Linear, Trimf


def test_kernel_init():
    with pytest.raises(TypeError):
        Kernel()  # pylint: disable=no-value-for-parameter
    with pytest.raises(TypeError):
        Kernel(0)  # pylint: disable=no-value-for-parameter
    with pytest.raises(ValueError):
        Kernel("str", "1")
    with pytest.raises(ValueError):
        Kernel("0", "1")
    with pytest.raises(ValueError):
        Kernel(0, "1")
    with pytest.raises(ValueError):
        Kernel(2, 1)
    with pytest.raises(ValueError):
        Kernel(float("inf"), float("-inf"))

    Kernel(0, 1)
    Kernel(-1, 20)
    Kernel(float("-inf"), float("inf"))


def test_kernel_add_func():
    food = Kernel(0, 10)
    food.add_memb_func("good", Linear(0, 10))
    assert isinstance(food.input_functions, dict)
    assert food.input_functions["good"]
    food.add_memb_func("bad", Linear(10, 0))
    assert isinstance(food.input_functions, dict)
    assert food.input_functions["good"]
    assert food.input_functions["bad"]
    assert len(food.input_functions) == 2
    with pytest.raises(TypeError):
        food.add_memb_func(Linear(0.1, 2), "good")  # should fail
    with pytest.raises(KeyError):
        food.input_functions["error"]  # pylint: disable=pointless-statement

    assert isinstance(food.input_functions["good"], MembershipFunction)
    assert isinstance(food.input_functions["bad"], MembershipFunction)

    with pytest.raises(TypeError):

        def regular_func_fail():
            return 0.5

        food.add_memb_func("func_error", regular_func_fail)  # should fail

    # corrupt kernel func dict
    with pytest.raises(TypeError):
        food.input_functions = ["wrong type", "will fail"]
        food.add_memb_func("bad", Linear(-0.1, 0))


def test_kernel_del_func():
    food = Kernel(0, 10)
    food.add_memb_func("good", Linear(0.1, 0))
    food.add_memb_func("bad", Linear(-0.1, 0))
    food.del_memb_func("good")
    assert len(food.input_functions) == 1
    assert food.input_functions.keys() == set(["bad"])
    food.del_memb_func("bad")
    assert len(food.input_functions) == 0
    assert food.input_functions.keys() == set([])
    assert isinstance(food.input_functions, dict)
    try:
        food.del_memb_func("error")  # should fail
    except KeyError:
        pass


def test_kernel_call():
    food = Kernel(0, 10)
    food.add_memb_func("good", Linear(0, 10))
    food.add_memb_func("bad", Linear(10, 0))
    print(food.membership_degree)
    food(8)
    assert np.round(food.membership_degree["good"], 1) == 0.8
    assert np.round(food.membership_degree["bad"], 1) == 0.2


def test_kernel_describe():
    food = Kernel(0, 10)
    food.add_memb_func("good", Linear(0, 10))
    food.add_memb_func("bad", Linear(10, 0))
    res = food.describe()
    x_arr, y_arr = res["good"]
    assert x_arr.all() == np.array([0, 10]).all()
    assert y_arr.all() == np.array([0, 1]).all()
    x_arr, y_arr = res["bad"]
    assert x_arr.all() == np.array([10, 0]).all()
    assert y_arr.all() == np.array([0, 1]).all()


def test_kernel_check_coverage():
    food = Kernel(0, 10)
    food.add_memb_func("bad", Trimf(-1, 0, 6))
    food.add_memb_func("average", Trimf(4, 6, 8))
    food.add_memb_func("good", Trimf(8, 10, 11))
    assert food.check_coverage() == True

    food.add_memb_func("bad", Trimf(-1, 0, 3))
    food.add_memb_func("average", Trimf(4, 6, 8))
    food.add_memb_func("good", Trimf(8, 10, 11))
    with pytest.warns(
        UserWarning,
        match=(
            "MembershipFunction 'bad' has no intersections. Variable space may not be fully defined"
        ),
    ):
        assert food.check_coverage() == False

    food.add_memb_func("bad", Trimf(-1, 0, 4))
    food.add_memb_func("average", Trimf(4, 6, 7.9))
    food.add_memb_func("good", Trimf(8, 10, 11))
    with pytest.warns(
        UserWarning,
        match=(
            "MembershipFunction 'good' has no intersections. Variable space may not be fully"
            " defined"
        ),
    ):
        assert food.check_coverage() == False

    food = Kernel(0, 10)
    food.add_memb_func("everything", Trimf(-1, 0, 10))
    assert food.check_coverage() == True

    food.add_memb_func("everything", Trimf(-1, 0, 9))
    assert food.check_coverage() == False

    food.add_memb_func("everything", Trimf(0.01, 1, 10))
    assert food.check_coverage() == False
