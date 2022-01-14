""" Membership function types, ranging from constant, linear or other more complex shape mapping """
from typing import Any


def constant(data: Any, cons: float) -> float:
    """
    """
    return cons

def linear(data: Any, slope: float, b: float) -> float:
    """
    """
    return slope * data + b

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
