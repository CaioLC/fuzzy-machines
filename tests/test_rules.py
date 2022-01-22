""" tests for rules.py """
# pylint: disable=missing-function-docstring, invalid-name
import pytest
from fuzzy_machines.operators import OperatorEnum
from fuzzy_machines.memb_funcs import Linear
from fuzzy_machines.kernel import Kernel, KernelFuncMember
from fuzzy_machines.rules import AND, OR, NOT, RuleBase

food = (
    Kernel(0, 10)
    .add_memb_func("good", KernelFuncMember(Linear(0.1, 0)))
    .add_memb_func("rancid", KernelFuncMember(Linear(-0.1, 1)))
)
food(8)

service = (
    Kernel(0, 10)
    .add_memb_func("good", KernelFuncMember(Linear(0.1, 0)))
    .add_memb_func("bad", KernelFuncMember(Linear(-0.1, 1)))
)
service(3)

price = (
    Kernel(0, 10)
    .add_memb_func("cheap", KernelFuncMember(Linear(-0.1, 1)))
    .add_memb_func("expensive", KernelFuncMember(Linear(0.1, 0)))
)
price(7)

input_kernel_set = {
    "food": food,
    "service": service,
    "price": price,
}


def test_rule_init():
    op = RuleBase(OperatorEnum.DEFAULT, {"food": "good"})
    op("mock_me")
    with pytest.raises(TypeError):
        RuleBase(OperatorEnum)  # pylint: disable=no-value-for-parameter

    with pytest.raises(TypeError):
        RuleBase("will fail", {"food": "good"})


def test_and():
    op = AND({"food": "good"}, {"service": "good"}, OperatorEnum.DEFAULT)
    assert round(op(input_kernel_set), 1) == 0.3


def test_or():
    op = OR({"food": "good"}, {"service": "good"}, OperatorEnum.DEFAULT)
    assert round(op(input_kernel_set), 1) == 0.8


def test_not():
    op = NOT({"food": "good"}, OperatorEnum.DEFAULT)
    assert round(op(input_kernel_set), 1) == 0.2


def test_nested():
    op = OR(
        AND({"food": "good"}, {"price": "cheap"}, OperatorEnum.DEFAULT),
        AND({"food": "rancid"}, {"service": "good"}, OperatorEnum.DEFAULT),
        OperatorEnum.DEFAULT,
    )
    assert round(op(input_kernel_set), 1) == 0.3

    op = OR(
        AND(
            {"food": "good"},
            AND({"service": "good"}, {"price": "expensive"}, OperatorEnum.DEFAULT),
            OperatorEnum.DEFAULT,
        ),
        AND(
            {"food": "rancid"},
            AND(
                {"service": "good"},
                NOT({"price": "expensive"}, OperatorEnum.DEFAULT),
                OperatorEnum.DEFAULT,
            ),
            OperatorEnum.DEFAULT,
        ),
        OperatorEnum.DEFAULT,
    )
    assert round(op(input_kernel_set), 1) == 0.3
