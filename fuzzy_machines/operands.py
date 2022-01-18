""" Fuzzy Logic Operators """
# pylint: disable=invalid-name, missing-function-docstring
from enum import Enum


def and_default(a, b):
    return min(a, b)


def or_default(a, b):
    return max(a, b)


def not_default(a):
    return 1 - a


class OperandEnum(Enum):
    """Defines the behaviour of AND, OR and NOT rules method"""

    DEFAULT = [and_default, or_default, not_default]
