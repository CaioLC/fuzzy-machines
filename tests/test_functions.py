"""Tests for `fuzzy_machines.functions` package."""
from fuzzy_machines import functions


def test_convert(capsys):
    """Correct my_name argument prints"""
    wrong_types = {"int": 3, "list": ["a", "b", "d"], "dict": {"a": 1, "b": 2}}
    for w_type in wrong_types.values():
        try:
            functions.convert(w_type)
        except TypeError:
            print(f"successfully blocked type {w_type}")

    functions.convert("Jill")
    captured = capsys.readouterr()
    assert "Jill" in captured.out


def test_a_second_function():
    assert functions.a_second_function(5)
