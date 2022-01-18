""" tests for engine.py """
# pylint: disable=missing-function-docstring, invalid-name
from typing import cast
import pytest

from fuzzy_machines.engine import Engine
from fuzzy_machines.kernel import Kernel, KernelFuncMember
from fuzzy_machines.memb_funcs import Constant, Linear
from fuzzy_machines.operands import OperandEnum
from fuzzy_machines.rules import AND, NOT, OR, RuleBase

food_quality = (
    Kernel(0, 10)
    .add_memb_func("rancid", KernelFuncMember(Linear(-3, 1)))
    .add_memb_func("good", KernelFuncMember(Linear(2, 1)))
)

food_service = (
    Kernel(0, 10)
    .add_memb_func("bad", KernelFuncMember(Constant(0.1)))
    .add_memb_func("good", KernelFuncMember(Constant(0.8)))
)

tips = (
    Kernel(0, 30)
    .add_memb_func("low", KernelFuncMember(Constant(0.10)))
    .add_memb_func("average", KernelFuncMember(Constant(0.20)))
    .add_memb_func("high", KernelFuncMember(Constant(0.30)))
)

### ENGINE BASE TESTS ###
def test_engine_create():
    eng = Engine()
    assert isinstance(eng, Engine)


def test_repr():
    eng = Engine()
    print(eng)


def test_add_kernel():
    wrong_types = {
        "int": 3,
        "list": ["a", "b", "d"],
        "str": "Caio",
        "reg_func": lambda x: x + 1,
    }
    eng = Engine()
    eng.add_kernel("f_quality", food_quality)
    eng.add_kernel("f_service", food_service)

    # wrong names
    with pytest.raises(TypeError):
        eng.add_kernel(123, food_service)

    # wrong values
    for val in wrong_types.values():
        with pytest.raises(TypeError):
            eng.add_kernel("will_fail", val)

    # eng.rules with corrupted format
    eng.input_kernel_set = [1, 2, 3]
    with pytest.raises(TypeError):
        eng.add_kernel("f_service", food_service)


def test_del_kernel():
    eng = Engine()
    eng.add_kernel("func", food_service)
    eng.del_kernel("func")
    with pytest.raises(KeyError):
        eng.del_kernel("does_not_exist")


def test_inference_kernel():
    eng = Engine().add_inference_kernel(tips)
    assert isinstance(eng.inference_kernel, Kernel)
    eng.del_inference_kernel()
    assert eng.inference_kernel is None
    with pytest.raises(TypeError):
        eng.add_inference_kernel('will_fail')


### ENGINE SUBCLASSES TESTS ###
def test_add_rule():
    # simple dict
    eng = Engine().add_rule("low", {"food": "rancid"})
    assert isinstance(eng.ruleset["low"], dict)

    # one rule
    eng.add_rule("high", AND({"food": "good"}, {"service": "good"}))
    assert len(eng.ruleset) == 2
    simple_rule = cast(RuleBase, eng.ruleset["high"])
    assert simple_rule.operand_set is not None
    assert isinstance(simple_rule.operand_set, OperandEnum)

    # nested rules
    nested_rule = OR(
        AND(
            {"food": "good"},
            AND({"service": "good"}, {"price": "expensive"}),
        ),
        AND(
            {"food": "rancid"},
            AND({"service": "good"}, NOT({"price": "expensive"})),
        ),
    )
    eng.add_rule("average", nested_rule)
    assert len(eng.ruleset) == 3
    lv1_rule = cast(RuleBase, eng.ruleset["average"])
    assert lv1_rule.operand_set is not None
    assert isinstance(lv1_rule.operand_set, OperandEnum)
    lv2_rule = cast(RuleBase, lv1_rule.a)
    assert lv2_rule.operand_set is not None
    assert isinstance(lv2_rule.operand_set, OperandEnum)
    lv3_rule = cast(RuleBase, lv2_rule.b)
    assert lv3_rule.operand_set is not None
    assert isinstance(lv3_rule.operand_set, OperandEnum)

    # same as above, but now engine declares OperandEnum explicitly.
    eng = Engine(OperandEnum.DEFAULT).add_rule("low", {"food": "rancid"})
    assert isinstance(eng.ruleset["low"], dict)
    print(eng.operands)
    eng.add_rule("high", AND({"food": "good"}, {"service": "good"}))
    assert len(eng.ruleset) == 2
    simple_rule = cast(RuleBase, eng.ruleset["high"])
    assert simple_rule.operand_set is not None
    assert isinstance(simple_rule.operand_set, OperandEnum)
    nested_rule = OR(
        AND(
            {"food": "good"},
            AND({"service": "good"}, {"price": "expensive"}),
        ),
        AND(
            {"food": "rancid"},
            AND({"service": "good"}, NOT({"price": "expensive"})),
        ),
    )
    eng.add_rule("average", nested_rule)
    assert len(eng.ruleset) == 3
    lv1_rule = cast(RuleBase, eng.ruleset["average"])
    assert lv1_rule.operand_set is not None
    assert isinstance(lv1_rule.operand_set, OperandEnum)
    lv2_rule = cast(RuleBase, lv1_rule.a)
    assert lv2_rule.operand_set is not None
    assert isinstance(lv2_rule.operand_set, OperandEnum)
    lv3_rule = cast(RuleBase, lv2_rule.b)
    assert lv3_rule.operand_set is not None
    assert isinstance(lv3_rule.operand_set, OperandEnum)


def test_delete_rule():
    # nested rules
    nested_rule = OR(
        AND(
            {"food": "good"},
            AND({"service": "good"}, {"price": "expensive"}),
        ),
        AND(
            {"food": "rancid"},
            AND({"service": "good"}, NOT({"price": "expensive"})),
        ),
    )
    eng = (
        Engine()
        .add_rule("low", {"food": "rancid"})
        .add_rule("high", AND({"food": "good"}, {"service": "good"}))
        .add_rule("average", nested_rule)
    )
    assert len(eng.ruleset) == 3
    eng.delete_rule("low")
    assert len(eng.ruleset) == 2
    assert eng.ruleset.keys() == set(["average", "high"])
    eng.delete_rule("average")
    assert len(eng.ruleset) == 1
    assert eng.ruleset.keys() == set(["high"])
    eng.delete_rule("high")
    assert len(eng.ruleset) == 0
    assert eng.ruleset.keys() == set([])
    with pytest.raises(KeyError):
        eng.delete_rule("error")


def test_fuzzify():
    eng = (
        Engine()
        .add_kernel("food", food_quality)
        .add_kernel("service", food_service)
        .add_inference_kernel(tips)
        .add_rule("low", {"food": "rancid"})
        .add_rule(
            "average",
            OR(
                AND({"food": "rancid"}, {"service": "good"}),
                AND({"food": "good"}, {"service": "bad"}),
            ),
        )
        .add_rule("high", AND({"food": "good"}, {"service": "good"}))
    )

    measurement_data = dict({"food": 3, "service": 9})
    eng.fuzzyfy(measurement_data)
    assert eng.fuzzy_res.keys() == set(["low", "average", "high"])
    for val in eng.fuzzy_res.values():
        assert isinstance(val, float)

    # wrong ruleset_data:
    measurement_data = dict({"food_wrong_name": 3, "service_wrong_name": 9})
    with pytest.raises(KeyError):
        eng.fuzzyfy(measurement_data)


def test_defuzzify():
    eng = Engine()
    with pytest.raises(NotImplementedError):
        eng.defuzzyfy()
