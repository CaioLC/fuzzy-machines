"""Tests for `fuzzy_machines.functions` package."""
from fuzzy_machines import memb_funcs


def test_convert(capsys):
    """Correct my_name argument prints"""
    wrong_types = {"int": 3, "list": ["a", "b", "d"], "dict": {"a": 1, "b": 2}}
    for w_type in wrong_types.values():
        try:
            memb_funcs.constant(w_type)
        except TypeError:
            print(f"successfully blocked type {w_type}")

    memb_funcs.constant("Jill")
    captured = capsys.readouterr()
    assert "Jill" in captured.out


def test_a_second_function():
    assert memb_funcs.linear(5)
