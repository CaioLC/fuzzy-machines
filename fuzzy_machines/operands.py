""" Fuzzy Logic Operators """
from enum import Enum


def and_default(a, b):
    return min(a, b)


def or_default(a, b):
    return max(a, b)


def not_default(a):
    return 1 - a


class OperandEnum(Enum):
    DEFAULT = [and_default, or_default, not_default]
