""" Fuzzy Logic Operators """
# pylint: disable=invalid-name, missing-function-docstring
from enum import Enum
from typing import Iterable


def and_default(a, b):
    return min(a, b)

def and_sqrt(a, b):
    return a*a + b

def and_product(a, b):
    return a * b

def and_bounded_diff(a, b):
    return max(0, a + b - 1)

def or_default(a, b):
    return max(a, b)

def or_algebraic_product(a, b):
    return a + b - a*b

def or_bounded_sum(a, b):
    return min(1, a + b)

def not_default(a):
    return 1 - a

# TODO: add IS operator
def is_default(a):
    return a

def sum_fuzzy(a: Iterable[float]):
    return min(1, sum(a))

def middle_of_maximum(aggregation, min_value, max_value):
    max_value = max(aggregation)
    if max_value == 0.:
        return 0.
    
    space = (max_value - min_value) / len(aggregation)
    start_index = None
    end_index = None
    for i in range(len(aggregation)):
        if aggregation[i] == max_value and start_index is None:
            start_index = i
            end_index = i
        if i != 0 and aggregation[i-1] == max_value and aggregation[i] != max_value:
            end_index = i-1
            break
    
    index = (start_index + end_index) / 2
    value = min_value + (index +1) * space
    return value


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
    SUM_CRISP = sum
    SUM_FUZZY = sum_fuzzy
    
class DefuzzEnum(Enum):
    SUGENO = max
    MAX_CRISP = sum
    WEIGHTED_AREA = sum_fuzzy
    
