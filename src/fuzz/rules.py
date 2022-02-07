""" AND, NOT and OR operators for fuzzy logic inference systems """
# pylint: disable=invalid-name, R0903

from typing import Union, Dict
import numpy as np

from .kernel import Kernel
from .operators import OperatorEnum


class Rules:
    """Rules MetaClass"""


class RuleBase(Rules):
    """Base class for all declarative rules"""

    def __init__(self, operand_set: OperatorEnum, a, b=None) -> None:
        if operand_set is not None and not isinstance(operand_set, OperatorEnum):
            raise TypeError(f"Expected type OperatorEnum for 'operand'. Got {type(operand_set)}")
        self.operand_set = operand_set
        self.a = a
        self.b = b

    def __call__(self, input_kernel_membership) -> np.ndarray:
        pass


def _resolve(x: Union[Rules, Dict[str, str]], input_kernel_set: Dict[str, Kernel]) -> np.ndarray:
    if isinstance(x, RuleBase):
        return x(input_kernel_set)
    assert len(x) == 1
    variable, membership_val = x.popitem()
    if variable in input_kernel_set.keys():
        return input_kernel_set[variable].membership_degree[membership_val]
    raise KeyError(f"Cannot find kernel named '{variable}' in the kernel set")


class AND(RuleBase):
    """AND Operator. Performs the AND function as defined by the OperatorEnum of choice"""

    def __init__(
        self,
        a: Union[RuleBase, Dict[str, str]],
        b: Union[RuleBase, Dict[str, str]],
        operand_set: OperatorEnum = None,
    ) -> float:
        super().__init__(operand_set, a, b)

    def __call__(self, input_kernel_membership) -> np.ndarray:
        a = _resolve(self.a, input_kernel_membership)
        b = _resolve(self.b, input_kernel_membership)
        func = self.operand_set.value[0]
        # print("AND:", a, b, "->", func(a, b))
        return func(a, b)


class OR(RuleBase):
    """OR Operator. Performs the OR function as defined by the OperatorEnum of choice"""

    def __init__(
        self,
        a: Union[RuleBase, Dict[str, str]],
        b: Union[RuleBase, Dict[str, str]],
        operand_set: OperatorEnum = None,
    ) -> float:
        super().__init__(operand_set, a, b)

    def __call__(self, input_kernel_membership) -> np.ndarray:
        a = _resolve(self.a, input_kernel_membership)
        b = _resolve(self.b, input_kernel_membership)
        func = self.operand_set.value[1]
        # print("OR:", a, b, "->", func(a, b))
        return func(a, b)


class NOT(RuleBase):
    """NOT Operator. Performs the NOT function as defined by the OperatorEnum of choice"""

    def __init__(
        self, a: Union[RuleBase, Dict[str, str]], operand_set: OperatorEnum = None
    ) -> float:
        super().__init__(operand_set, a)

    def __call__(self, input_kernel_set) -> np.ndarray:
        a = _resolve(self.a, input_kernel_set)
        func = self.operand_set.value[2]
        # print("NOT:", a, "->", func(a))
        return func(a)


class IS(RuleBase):
    """IS Operator. Performs the IS function as defined by the OperatorEnum of choice"""

    def __init__(
        self, a: Union[RuleBase, Dict[str, str]], operand_set: OperatorEnum = None
    ) -> float:
        super().__init__(operand_set, a)

    def __call__(self, input_kernel_set) -> np.ndarray:
        a = _resolve(self.a, input_kernel_set)
        func = self.operand_set.value[3]
        # print("IS:", a, "->", func(a))
        return func(a)
