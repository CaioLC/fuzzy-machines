""" Fuzzy Logic Operators """
# pylint: disable=invalid-name, missing-function-docstring
from enum import Enum, auto
from math import prod
from itertools import product
from typing import Iterable

from fuzzy_machines.kernel import Kernel
from fuzzy_machines.memb_funcs import FunctionBase


def and_default(a, b):
    return min(a, b)


def and_product(a, b):
    return a * b


def and_bounded_diff(a, b):
    return max(0, a + b - 1)


def or_default(a, b):
    return max(a, b)


def or_algebraic_product(a, b):
    return a + b - a * b


def or_bounded_sum(a, b):
    return min(1, a + b)


def not_default(a):
    return 1 - a


def is_default(a):
    return a


class OperatorEnum(Enum):
    """
    Defines the behaviour of AND, OR and NOT rules method. \
    The outcome of a fuzzy system is strongly dependent on the specific choice of operators: \
    1. For classification tasks, the min/max operators (DEFAULT) are popular.\
    2. For approximation and identification, the product and algebraic product are better suited. \
    3. For some neuro-fuzzy learning schemes, the bounded difference offer several advantages.
    (See NELLES, Oliver, 2020. Nonlinear System Identification, 2nd edition)
    """

    DEFAULT = [and_default, or_default, not_default, is_default]
    PRODUCT = [and_product, or_algebraic_product, not_default, is_default]
    BOUNDED = [and_bounded_diff, or_bounded_sum, not_default, is_default]


class RuleAggregationEnum(Enum):
    MAX = max


class DefuzzEnum(Enum):
    TAKAGI_SUGENO = auto()
    LINGUISTIC = auto()
