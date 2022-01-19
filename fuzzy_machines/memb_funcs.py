""" Membership function types, ranging from constant, linear or other more complex shape mapping """
# pylint: disable=fixme, invalid-name, R0903
from typing import Any


class FunctionBase:
    """Function Meta"""


class Singleton(FunctionBase):
    """Boolean function. Return 1 when x == value and 0 otherwise"""
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, data: Any) -> float:
        if self.value == data: return 1 
        else: return 0


class Constant(FunctionBase):
    """Constant function. Returns the initialization value"""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, data: Any) -> float:
        return self.value


class Linear(FunctionBase):
    """Linear function"""

    def __init__(self, slope: float, b: float) -> None:
        self.slope = slope
        self.b = b

    def __call__(self, data) -> float:
        return self.slope * data + self.b


class Smf(FunctionBase):
    """S-shaped membership function"""


class Pimf(FunctionBase):
    """Pi-shaped membership function"""


class Zmf(FunctionBase):
    """Z-shaped membership function"""


class Trimf(FunctionBase):
    """Triangular membership function"""


class Trapmf(FunctionBase):
    """Trapezoidal membership function"""


class Gaussmf(FunctionBase):
    """Gaussian membership function"""


class Gauss2mf(FunctionBase):
    """Gaussian combination membership function"""


class Gbellmf:
    """Generalized bell-shaped membership function"""
