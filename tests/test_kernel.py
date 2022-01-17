from numbers import Number
from types import FunctionType
import pytest
from fuzzy_machines.kernel import Kernel, KernelFuncMember, _clamp
from fuzzy_machines.memb_funcs import FunctionBase, Constant, Linear


def test_clamp():
    assert 0 == _clamp(-1.234, 0, 1), "returned value below lower boundary"
    assert 1 == _clamp(1.234, 0, 1), "returned value above higher boundary"
    assert 0.234 == _clamp(0.234, 0, 1), "did not returned value between boundaries"


def test_kernel_func_member_init():
    kmember = KernelFuncMember(Constant(0.5))
    assert isinstance(kmember.func, FunctionBase)
    with pytest.raises(TypeError):

        def generic_function():
            return True

        kmember = KernelFuncMember(generic_function)


def test_kernel_func_member_call():
    kmember = KernelFuncMember(Constant(0.5))
    assert kmember(20) == 0.5
    assert kmember("string") == 0.5
    assert kmember([20, 10]) == 0.5
    assert kmember({"20": 20}) == 0.5
    assert kmember((20, "v")) == 0.5

    kmember = KernelFuncMember(Linear(0.1, 0))
    assert kmember(0) == 0
    assert kmember(1) == 0.1
    assert kmember(5) == 0.5
    assert kmember(10) == 1
    assert kmember(50) == 1
    assert kmember(-1) == 0
    try:
        kmember("str")
    except TypeError:
        pass


def test_kernel_func_member_iterate():
    kmember = KernelFuncMember(Linear(0.5, 0))
    points = kmember.iterate(-1, 1, 5)
    assert [0, 0, 0, 0.25, 0.5] == points


def test_kernel_init():
    with pytest.raises(TypeError):
        Kernel()
    with pytest.raises(TypeError):
        Kernel(0)
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
    food.add_memb_func("good", KernelFuncMember(Linear(0.1, 0)))
    assert isinstance(food.input_functions, dict)
    assert food.input_functions["good"]
    food.add_memb_func("bad", KernelFuncMember(Linear(-0.1, 0)))
    assert isinstance(food.input_functions, dict)
    assert food.input_functions["good"]
    assert food.input_functions["bad"]
    assert len(food.input_functions) == 2
    with pytest.raises(TypeError):
        food.add_memb_func(KernelFuncMember(Linear(0.1, 2)), "good")  # should fail
    with pytest.raises(KeyError):
        food.input_functions["error"]  # should fail

    assert isinstance(food.input_functions["good"], KernelFuncMember)
    assert isinstance(food.input_functions["bad"], KernelFuncMember)

    with pytest.raises(TypeError):

        def regular_func_fail():
            return 0.5

        food.add_memb_func("func_error", regular_func_fail)  # should fail

    # corrupt kernel func dict
    with pytest.raises(TypeError):
        food.input_functions = ["wrong type", "will fail"]
        food.add_memb_func("bad", KernelFuncMember(Linear(-0.1, 0)))


def test_kernel_del_func():
    food = Kernel(0, 10)
    food.add_memb_func("good", KernelFuncMember(Linear(0.1, 0)))
    food.add_memb_func("bad", KernelFuncMember(Linear(-0.1, 0)))
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
    food.add_memb_func("good", KernelFuncMember(Linear(0.1, 0)))
    food.add_memb_func("bad", KernelFuncMember(Linear(-0.1, 1)))
    print(food.input_membership)
    food(8)
    assert round(food.input_membership["good"], 1) == 0.8
    assert round(food.input_membership["bad"], 1) == 0.2


def test_kernel_describe():
    food = Kernel(0, 10)
    food.add_memb_func("good", KernelFuncMember(Linear(0.1, 0)))
    food.add_memb_func("bad", KernelFuncMember(Linear(-0.1, 1)))
    res = food.describe(11)
    assert [round(val, 1) for val in res["good"]] == [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ]
    assert [round(val, 1) for val in res["bad"]] == [
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.1,
        0,
    ]

    food = Kernel(-5, 15)
    food.add_memb_func("good", KernelFuncMember(Linear(0.1, 0)))
    food.add_memb_func("bad", KernelFuncMember(Linear(-0.1, 1)))
    res = food.describe(21)
    assert [round(val, 1) for val in res["good"]] == [
        0,
        0,
        0,
        0,
        0,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert [round(val, 1) for val in res["bad"]] == [
        1,
        1,
        1,
        1,
        1,
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.1,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
