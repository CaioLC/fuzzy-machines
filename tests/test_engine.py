""" The machine to run the fuzzy logic """
from fuzzy_machines import engine
from fuzzy_machines.engine import Engine


def food_quality(score: str) -> int:
    if score == "bad":
        return 0
    if score == "average":
        return 0.5
    if score == "good":
        return 1


def food_service(score: str) -> int:
    if score == "bad":
        return 0
    if score == "average":
        return 0.5
    if score == "good":
        return 1


def test_engine_create():
    eng = Engine()
    assert isinstance(eng, Engine)
    eng = Engine(True)
    assert isinstance(eng, Engine)


def test_repr():
    eng = Engine()
    print(eng)


def test_add_rule():
    wrong_types = {"int": 3, "list": ["a", "b", "d"], "str": "Caio"}
    eng = Engine(False)
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


def test_delete_rule():
    eng = Engine(False)
    eng.add_kernel("func", food_service)
    eng.delete_kernel("func")
    try:
        eng.delete_kernel("does_not_exist")
    except KeyError:
        pass


def test_fuzzify():
    eng = Engine()
    eng.add_kernel("f_quality", food_quality)
    eng.add_kernel("f_service", food_service)
    ruleset_data = dict({"f_quality": "bad", "f_service": "good"})
    eng.fuzzify(ruleset_data)

    # mismatch between ruleset and rule_data
    try:
        eng.delete_kernel("f_quality")
        eng.fuzzify(ruleset_data)
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
            eng.fuzzify(val)
        except AttributeError:  # types are missing the .keys() attribute of dict.
            pass


def test_defuzzify():
    eng = Engine()
    eng.add_kernel("f_quality", food_quality)
    eng.add_kernel("f_service", food_service)
    ruleset_data = dict({"f_quality": "bad", "f_service": "good"})
    eng.fuzzify(ruleset_data)
    assert eng.defuzzyfy() == 1

    # corrupt fuzzy_results:
    eng.fuzzy_res = []
    try:
        eng.defuzzyfy()
    except ValueError:
        pass
