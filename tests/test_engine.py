""" tests for engine.py """
# pylint: disable=missing-function-docstring, invalid-name
from typing import cast
import numpy as np
import pytest

from src.fuzz.engine import Engine
from src.fuzz.kernel import Kernel
from src.fuzz.memb_funcs import Constant, MembershipFunction, Linear
from src.fuzz.operators import DefuzzEnum, OperatorEnum, RuleAggregationEnum
from src.fuzz.rules import AND, NOT, OR, IS, RuleBase

food_quality = (
    Kernel(0, 10).add_memb_func("rancid", Linear(10, 0)).add_memb_func("good", Linear(0, 10))
)

food_service = (
    Kernel(0, 10).add_memb_func("bad", Constant(0, 5)).add_memb_func("good", Constant(5, 10))
)

food_price = (
    Kernel(0, 10).add_memb_func("cheap", Linear(10, 0)).add_memb_func("expensive", Linear(0, 10))
)

tips = (
    Kernel(0, 30)
    .add_memb_func("low", Constant(0, 4))
    .add_memb_func("average", Constant(4, 8))
    .add_memb_func("high", Constant(8, 10))
)


class TKG_Low(MembershipFunction):
    def __init__(self) -> None:
        super().__init__(0, 10)

    def __call__(self, food: np.ndarray, price: np.ndarray, service: np.ndarray) -> np.ndarray:
        return 10 + 0.4 * food + 0.3 * price + 0.3 * service


class TKG_Medium(MembershipFunction):
    def __init__(self) -> None:
        super().__init__(0, 10)

    def __call__(self, food: np.ndarray, price: np.ndarray, service: np.ndarray) -> np.ndarray:
        return 0.8 * food + 0.6 * price + 0.6 * service


class TKG_High(MembershipFunction):
    def __init__(self) -> None:
        super().__init__(0, 10)

    def __call__(self, food: np.ndarray, price: np.ndarray, service: np.ndarray) -> np.ndarray:
        return 1.2 * food + 0.9 * price + 0.9 * service


takagi_set = (
    Kernel(0, 10)
    .add_memb_func("low", TKG_Low())
    .add_memb_func("medium", TKG_Medium())
    .add_memb_func("high", TKG_High())
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
        eng.add_inference_kernel("will_fail")


### ENGINE SUBCLASSES TESTS ###
def test_add_rule():

    eng = Engine().add_rule("low", IS({"food": "rancid"}))
    assert isinstance(eng.ruleset["low"], list)
    assert len(eng.ruleset["low"]) == 1
    eng.add_rule("low", IS({"service": "poor"}))
    assert isinstance(eng.ruleset["low"], list)
    print(eng.ruleset["low"])
    assert len(eng.ruleset["low"]) == 2

    with pytest.raises(TypeError):
        eng.add_rule("high", {"food": "good"})

    eng.add_rule("high", AND({"food": "good"}, {"service": "good"}))
    assert len(eng.ruleset) == 2
    simple_rule = cast(RuleBase, eng.ruleset["high"])
    assert simple_rule[0].operand_set is not None
    assert isinstance(simple_rule[0].operand_set, OperatorEnum)

    # nested rules
    nested_rule = OR(
        AND(
            IS({"food": "good"}),
            AND({"service": "good"}, {"price": "expensive"}),
        ),
        AND(
            IS({"food": "rancid"}),
            AND({"service": "good"}, NOT({"price": "expensive"})),
        ),
    )
    eng.add_rule("average", nested_rule)
    assert len(eng.ruleset) == 3
    lv1_rule = cast(RuleBase, eng.ruleset["average"][0])
    assert lv1_rule.operand_set is not None
    assert isinstance(lv1_rule.operand_set, OperatorEnum)
    lv2_rule = cast(RuleBase, lv1_rule.a)
    assert lv2_rule.operand_set is not None
    assert isinstance(lv2_rule.operand_set, OperatorEnum)
    lv3_rule = cast(RuleBase, lv2_rule.b)
    assert lv3_rule.operand_set is not None
    assert isinstance(lv3_rule.operand_set, OperatorEnum)


def test_delete_rule():
    # nested rules
    nested_rule = OR(
        AND(
            IS({"food": "good"}),
            AND({"service": "good"}, {"price": "expensive"}),
        ),
        AND(
            IS({"food": "rancid"}),
            AND({"service": "good"}, NOT({"price": "expensive"})),
        ),
    )
    eng = (
        Engine()
        .add_rule("low", IS({"food": "rancid"}))
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
    eng = Engine().add_kernel("food", food_quality).add_kernel("service", food_service)
    measurement_data = dict({"food": 4, "service": 8})
    input_fuzz = eng._fuzzyfy(measurement_data)
    assert input_fuzz.keys() == set(["food", "service"])
    print(eng.input_kernel_set)
    assert eng.input_kernel_set["food"].membership_degree == {
        "good": np.array(0.4),
        "rancid": np.array(0.6),
    }
    assert eng.input_kernel_set["service"].membership_degree == {
        "good": np.array(1.0),
        "bad": np.array(0.0),
    }

    with pytest.raises(KeyError):
        wrong_data = dict({"typo_food": 4, "typo_service": 8})
        eng._fuzzyfy(wrong_data)


def test_aggregation():
    aggregation_method = [
        (
            RuleAggregationEnum.MAX,
            {"food": 4, "service": 8, "price": 2},
            {"low": np.array(0.6), "average": np.array(0.6), "high": np.array(0.4)},
        ),
        (
            RuleAggregationEnum.MAX,
            {"food": np.array(4), "service": np.array(8), "price": np.array(2)},
            {"low": np.array([0.6]), "average": np.array([0.6]), "high": np.array([0.4])},
        ),
        (
            RuleAggregationEnum.MAX,
            {
                "food": np.array([4, 8, 10]),
                "service": np.array([8, 10, 3]),
                "price": np.array([2, 2, 8]),
            },
            {
                "low": np.array([0.6, 0.2, 0.8]),
                "average": np.array([0.6, 0.2, 1.0]),
                "high": np.array([0.4, 0.8, 0.0]),
            },
        ),
    ]
    for agg, measurement_data, result in aggregation_method:
        eng = (
            Engine(rule_agg=agg)
            .add_kernel("food", food_quality)
            .add_kernel("service", food_service)
            .add_kernel("price", food_price)
            .add_rule("low", IS({"food": "rancid"}))
            .add_rule("low", IS({"price": "expensive"}))
            .add_rule(
                "average",
                OR(
                    AND({"food": "rancid"}, {"service": "good"}),
                    AND({"food": "good"}, {"service": "bad"}),
                ),
            )
            .add_rule("high", AND({"food": "good"}, {"service": "good"}))
        )
        eng._fuzzyfy(measurement_data)
        res = eng._aggregate()
        assert np.all(list(res.values())) == np.all(list(result.values()))


tips_lin = Kernel(10, 30).add_memb_func("low", Linear(25, 10)).add_memb_func("high", Linear(20, 30))


def test_accumulation():
    eng = (
        Engine(rule_agg=RuleAggregationEnum.MAX, defuzz_method=DefuzzEnum.LINGUISTIC)
        .add_kernel("food", food_quality)
        .add_kernel("service", food_service)
        .add_kernel("price", food_price)
        .add_rule("low", IS({"food": "rancid"}))
        .add_rule("low", IS({"price": "expensive"}))
        .add_rule("high", AND({"food": "good"}, {"service": "good"}))
    )
    measurement_data = dict({"food": 9, "service": 8, "price": 5})
    eng._fuzzyfy(measurement_data)
    eng._aggregate()

    # with no inference system
    with pytest.raises(ValueError):
        eng._accumulate(0.1)

    eng.add_inference_kernel(tips_lin)
    x_r, y_r = eng._accumulate(0.1)
    assert (y_r <= 1).all()
    assert (y_r >= 0).all()
    mask_low = x_r <= 20
    assert (y_r[mask_low] <= 0.6).all()
    mask_high = x_r >= 15
    assert (y_r[mask_high] <= 0.9).all()

    # with wrong defuzz method.
    eng = (
        Engine(rule_agg=RuleAggregationEnum.MAX, defuzz_method=DefuzzEnum.TAKAGI_SUGENO)
        .add_kernel("food", food_quality)
        .add_kernel("service", food_service)
        .add_kernel("price", food_price)
        .add_rule("low", IS({"food": "rancid"}))
        .add_rule("low", IS({"price": "expensive"}))
        .add_rule("high", AND({"food": "good"}, {"service": "good"}))
        .add_inference_kernel(tips_lin)
    )
    eng._fuzzyfy(measurement_data)
    eng._aggregate()
    with pytest.raises(ValueError):
        eng._accumulate(0.1)


def test_run_deffuz():
    defuzz = [(DefuzzEnum.LINGUISTIC, tips_lin), (DefuzzEnum.TAKAGI_SUGENO, takagi_set)]
    measurement_data = dict({"food": 8, "service": 8, "price": 10})
    for d, inf_sys in defuzz:
        eng = (
            Engine(rule_agg=RuleAggregationEnum.MAX, defuzz_method=d)
            .add_kernel("food", food_quality)
            .add_kernel("service", food_service)
            .add_kernel("price", food_price)
            .add_rule("low", IS({"food": "rancid"}))
            .add_rule("low", IS({"price": "expensive"}))
            .add_rule("medium", OR({"food": "good"}, AND({"service": "good"}, {"price": "cheap"})))
            .add_rule("high", AND({"food": "good"}, {"service": "good"}))
            .add_inference_kernel(inf_sys)
        )
        eng.run_defuzz(measurement_data, 0.1)

    eng = (
        Engine(defuzz_method="SUPERWRONG_METHOD")
        .add_kernel("food", food_quality)
        .add_kernel("service", food_service)
        .add_kernel("price", food_price)
        .add_rule("low", IS({"food": "rancid"}))
        .add_rule("low", IS({"price": "expensive"}))
        .add_rule("medium", OR({"food": "good"}, AND({"service": "good"}, {"price": "cheap"})))
        .add_rule("high", AND({"food": "good"}, {"service": "good"}))
        .add_inference_kernel(inf_sys)
    )
    with pytest.raises(NotImplementedError):
        eng.run_defuzz(measurement_data, 0.1)


def test_run_fuzz():
    eng = (
        Engine()
        .add_kernel("food", food_quality)
        .add_kernel("service", food_service)
        .add_rule("low", IS({"food": "rancid"}))
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
    eng.run_fuzz(measurement_data)
    assert eng.actuation_signal.keys() == set(["low", "average", "high"])
    for val in eng.actuation_signal.values():
        assert isinstance(val, np.ndarray)

    # wrong ruleset_data:
    measurement_data = dict({"food_wrong_name": 3, "service_wrong_name": 9})
    with pytest.raises(KeyError):
        eng.run_fuzz(measurement_data)


def test_gen_surface():
    defuzz = [(DefuzzEnum.LINGUISTIC, tips_lin), (DefuzzEnum.TAKAGI_SUGENO, takagi_set)]
    for d, inf_sys in defuzz:
        eng = (
            Engine(rule_agg=RuleAggregationEnum.MAX, defuzz_method=d)
            .add_kernel("food", food_quality)
            .add_kernel("service", food_service)
            .add_rule("low", IS({"food": "rancid"}))
            .add_rule("medium", OR({"food": "good"}, {"service": "good"}))
            .add_rule("high", AND({"food": "good"}, {"service": "good"}))
            .add_inference_kernel(inf_sys)
        )
        with pytest.raises(NotImplementedError):
            eng.gen_surface(20, 0.1)  # TODO: this should work.
