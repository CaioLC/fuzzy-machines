""" Membership function types, ranging from constant, linear or other more complex shape mapping """
from typing import Any, Tuple


class FunctionBase:
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


class Smf(FunctionBase):
    """S-shaped membership function"""

    pass


class Pimf(FunctionBase):
    """Pi-shaped membership function"""

    pass


class Zmf(FunctionBase):
    """Z-shaped membership function"""

    pass


class Trimf(FunctionBase):
    """Triangular membership function"""

    pass


class Trapmf(FunctionBase):
    """Trapezoidal membership function"""

    pass


class Gaussmf(FunctionBase):
    """Gaussian membership function"""

    pass


class Gauss2mf(FunctionBase):
    """Gaussian combination membership function"""

    pass


class Gbellmf:
    """Generalized bell-shaped membership function"""

    pass
