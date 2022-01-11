"""Tests for `notebookc` package."""
import pytest
from fuzzy_machines import functions


def test_convert(capsys):
    """Correct my_name argument prints"""
    wrong_types = {
        'int': 3,
        'list': ['a', 'b', 'd'],
        'dict': {'a': 1, 'b': 2}
    }
    for wt in wrong_types.values():
        try:
            functions.convert(wt)
        except TypeError:
            print(f'successfully blocked type {wt}')

    functions.convert('Jill')
    captured = capsys.readouterr()
    assert "Jill" in captured.out