""" The machine to run the fuzzy logic """
from typing import cast
from fuzzy_machines import engine
from fuzzy_machines.engine import Engine, Engine
from fuzzy_machines.kernel import Kernel, KernelFuncMember
from fuzzy_machines.memb_funcs import Constant, Linear
from fuzzy_machines.operand import OperandEnum
from fuzzy_machines.rules import AND, NOT, OR, RuleBase

food_quality = (
    Kernel(0, 10)
    .add_memb_func('rancid', KernelFuncMember(Linear(-3, 1)))
    .add_memb_func('good', KernelFuncMember(Linear(2, 1)))
    )

food_service = (
    Kernel(0, 10)
    .add_memb_func('bad', KernelFuncMember(Constant(.1)))
    .add_memb_func('good', KernelFuncMember(Constant(.8)))
)

tips = (
    Kernel(0, 30)
    .add_memb_func('low', KernelFuncMember(Constant(.10)))
    .add_memb_func('average', KernelFuncMember(Constant(.20)))
    .add_memb_func('high', KernelFuncMember(Constant(.30)))
)

### ENGINE BASE TESTS ###
def test_engine_create():
    eng = Engine()
    assert isinstance(eng, Engine)


def test_repr():
    eng = Engine()
    print(eng)


def test_add_kernel():
    wrong_types = {"int": 3, "list": ["a", "b", "d"], "str": "Caio", "reg_func": lambda x: x+1}
    eng = Engine()
    eng.add_kernel("f_quality", food_quality)
    eng.add_kernel("f_service", food_service)

    # wrong names
    try:
        eng.add_kernel(123, food_service)
    except TypeError:
        pass

    # wrong values
    for val in wrong_types.values():
        try:
            eng.add_kernel("will_fail", val)
        except TypeError:
            pass

    # eng.rules with corrupted format
    eng.input_kernel_set = [1, 2, 3]
    try:
        eng.add_kernel("f_service", food_service)
    except TypeError:
        pass


def test_del_kernel():
    eng = Engine()
    eng.add_kernel("func", food_service)
    eng.del_kernel("func")
    try:
        eng.del_kernel("does_not_exist")
    except KeyError:
        pass

def test_inference_kernel():
    eng = (Engine()
        .add_inference_kernel("tips", tips)
    )
    assert isinstance(eng.inference_kernel, Kernel)
    eng.del_inference_kernel()
    assert eng.inference_kernel == None

### ENGINE SUBCLASSES TESTS ###
def test_add_rule():
    #simple dict
    eng = (Engine()
        .add_rule('low', {'food': 'rancid'}))
    assert isinstance(eng.ruleset['low'], dict)
    print(eng.operands)

    # one rule
    eng.add_rule('high', AND({'food':'good'}, {'service':'good'}))
    assert len(eng.ruleset) == 2
    simple_rule = cast(RuleBase, eng.ruleset['high'])
    assert simple_rule.operand_set is not None
    assert isinstance(simple_rule.operand_set, OperandEnum)

    # nested rules
    nested_rule = OR(
            AND(
                {'food':'good'},
                AND({'service': 'good'}, {'price': 'expensive'}),
            ),
            AND(
                {'food': 'rancid'},
                AND({'service': 'good'},
                    NOT({'price': 'expensive'})
                ),
            ),
        )
    eng.add_rule('average', nested_rule)
    assert len(eng.ruleset) == 3
    lv1_rule = cast(RuleBase, eng.ruleset['average'])
    assert lv1_rule.operand_set is not None
    assert isinstance(lv1_rule.operand_set, OperandEnum)
    lv2_rule = cast(RuleBase, lv1_rule.a)
    assert lv2_rule.operand_set is not None
    assert isinstance(lv2_rule.operand_set, OperandEnum)
    lv3_rule = cast(RuleBase, lv2_rule.b)
    assert lv3_rule.operand_set is not None
    assert isinstance(lv3_rule.operand_set, OperandEnum)

    # same as above, but now engine declares OperandEnum explicitly.
    eng = (Engine(OperandEnum.DEFAULT)
        .add_rule('low', {'food': 'rancid'}))
    assert isinstance(eng.ruleset['low'], dict)
    print(eng.operands)
    eng.add_rule('high', AND({'food':'good'}, {'service':'good'}))
    assert len(eng.ruleset) == 2
    simple_rule = cast(RuleBase, eng.ruleset['high'])
    assert simple_rule.operand_set is not None
    assert isinstance(simple_rule.operand_set, OperandEnum)
    nested_rule = OR(
            AND(
                {'food':'good'},
                AND({'service': 'good'}, {'price': 'expensive'}),
            ),
            AND(
                {'food': 'rancid'},
                AND({'service': 'good'},
                    NOT({'price': 'expensive'})
                ),
            ),
        )
    eng.add_rule('average', nested_rule)
    assert len(eng.ruleset) == 3
    lv1_rule = cast(RuleBase, eng.ruleset['average'])
    assert lv1_rule.operand_set is not None
    assert isinstance(lv1_rule.operand_set, OperandEnum)
    lv2_rule = cast(RuleBase, lv1_rule.a)
    assert lv2_rule.operand_set is not None
    assert isinstance(lv2_rule.operand_set, OperandEnum)
    lv3_rule = cast(RuleBase, lv2_rule.b)
    assert lv3_rule.operand_set is not None
    assert isinstance(lv3_rule.operand_set, OperandEnum)

def test_fuzzify():
    eng = (Engine()
        .add_kernel("f_quality", food_quality)
        .add_kernel("f_service", food_service)
    )
    ruleset_data = dict({"f_quality": 3, "f_service": 9})
    eng.membership_strengh(ruleset_data)

    # mismatch between ruleset and rule_data
    try:
        eng.del_kernel("f_quality")
        eng.membership_strengh(ruleset_data)
    except ValueError:
        eng.add_kernel("f_quality", food_quality)

    # ruleset data is not dictionary
    wrong_types = {
        "int": 3,
        "list": ["a", "b", "d"],
        "num_list": [1, 2, 3],
        "str": "Caio",
    }
    for val in wrong_types.values():
        try:
            eng.membership_strengh(val)
        except AttributeError:  # types are missing the .keys() attribute of dict.
            pass

# def test_defuzzify():
#     eng = Engine()
#     eng.add_kernel("f_quality", food_quality)
#     eng.add_kernel("f_service", food_service)
#     ruleset_data = dict({"f_quality": 10, "f_service": 3})
#     eng.fuzzify(ruleset_data)
#     assert eng.defuzzyfy() == 1

#     # corrupt fuzzy_results:
#     eng.fuzzy_res = []
#     try:
#         eng.defuzzyfy()
#     except ValueError:
#         pass

