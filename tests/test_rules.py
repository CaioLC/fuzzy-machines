""" tests for rules.py """
# pylint: disable=missing-function-docstring, invalid-name
import numpy as np
import pytest
from fuzzy_machines.operators import OperatorEnum
from fuzzy_machines.memb_funcs import Linear
from fuzzy_machines.kernel import Kernel
from fuzzy_machines.rules import AND, OR, NOT, IS, RuleBase

OP_LIST = [OperatorEnum.DEFAULT, OperatorEnum.PRODUCT, OperatorEnum.BOUNDED]
food = Kernel(0, 10).add_memb_func("good", Linear(0, 10)).add_memb_func("rancid", Linear(10, 0))

service = Kernel(0, 10).add_memb_func("good", Linear(0, 10)).add_memb_func("bad", Linear(10, 0))

price = (
    Kernel(0, 10).add_memb_func("cheap", Linear(10, 0)).add_memb_func("expensive", Linear(0, 10))
)

input_kernel_set = {
    "food": food,
    "service": service,
    "price": price,
}

food(8)
service(3)
price(7)


def test_rule_init():
    op = RuleBase(OperatorEnum.DEFAULT, {"food": "good"})
    op("mock_me")
    with pytest.raises(TypeError):
        RuleBase(OperatorEnum)  # pylint: disable=no-value-for-parameter

    with pytest.raises(TypeError):
        RuleBase("will fail", {"food": "good"})


def test_and():
    results = [0.3, 0.24, 0.1]
    for operator, result in zip(OP_LIST, results):
        op = AND({"food": "good"}, {"service": "good"}, operator)
        assert np.round(op(input_kernel_set), 2) == result


def test_or():
    results = [0.8, 0.86, 1]
    for operator, result in zip(OP_LIST, results):
        print(operator, result)
        op = OR({"food": "good"}, {"service": "good"}, operator)
        assert np.round(op(input_kernel_set), 2) == result


def test_not():
    op = NOT({"food": "good"}, OperatorEnum.DEFAULT)
    assert np.round(op(input_kernel_set), 1) == 0.2


def test_is():
    op = IS({"food": "rancid"}, OperatorEnum.DEFAULT)
    assert np.round(op(input_kernel_set), 1) == 0.2


def test_nested():
    op = OR(
        AND({"food": "good"}, {"price": "cheap"}, OperatorEnum.DEFAULT),
        AND({"food": "rancid"}, {"service": "good"}, OperatorEnum.DEFAULT),
        OperatorEnum.DEFAULT,
    )
    assert np.round(op(input_kernel_set), 1) == 0.3

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
    assert np.round(op(input_kernel_set), 1) == 0.3


def test_multiple_rules_same_output_rule():
    pass
