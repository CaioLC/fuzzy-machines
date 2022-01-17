""" Membership function types, ranging from constant, linear or other more complex shape mapping """
from typing import Any, Tuple

class FunctionBase:
    def describe():
        pass


class Constant(FunctionBase):
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, data: Any) -> float:
        return self.value

class Linear(FunctionBase):
    def __init__(self, slope: float, b: float) -> None:
        self.slope = slope
        self.b = b

    def __call__(self, data) -> float:
        return self.slope * data + self.b

def smf():
    """ S-shaped membership function """
    raise NotImplementedError

def pimf():
    """ Pi-shaped membership function """
    raise NotImplementedError

def zmf():
    """ Z-shaped membership function """
    raise NotImplementedError

def trimf():
    """ Triangular membership function """
    raise NotImplementedError

def trapmf():
    """ Trapezoidal membership function """
    raise NotImplementedError

def gaussmf():
    """ Gaussian membership function """
    raise NotImplementedError

def gauss2mf():
    """ Gaussian combination membership function """
    raise NotImplementedError

def gbellmf():
    """ Generalized bell-shaped membership function """
    raise NotImplementedError
