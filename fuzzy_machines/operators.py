""" Fuzzy Logic Operators """
# pylint: disable=invalid-name, missing-function-docstring
from enum import Enum, auto
import numpy as np


def and_default(a, b):
    # np.asfarray(a, b)
    return np.minimum(a, b)


def and_product(a, b):
    return a * b


def and_bounded_diff(a, b):
    return max(0, a + b - 1)


def or_default(a, b):
    return np.maximum(a, b)


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
    """
    Defines the rules aggregation methods available when joining partial rules mapping to the \
    same output inference system function
    """

    MAX = max


class DefuzzEnum(Enum):
    """
    Defines the defuzzification methods available at Engine.
    """

    TAKAGI_SUGENO = auto()
    LINGUISTIC = auto()
