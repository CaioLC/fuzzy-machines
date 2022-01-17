from types import FunctionType
from typing import Union, Dict

from fuzzy_machines.operand import OperandEnum, and_default, or_default

class Rules:
    pass

class RuleBase(Rules):
    """ Base class for all declarative rules. 
    - If SERVICE is GOOD then -> tip is AVERAGE:  (output_kernel: avg_kernel_func(good))
    - If SERVICE (input_kernel) is POOR (input_memb) OR (operator) FOOD (input_kernel) is RANCID (input_memb) then -> tip is LOW (output_kernel: low_kernel_func(OR(poor, rancid)))
    this function defines a dict of {'low': low_kernel_func, 'average': avg_kernel_func, 'high': high_kernel_func }
    returns the "firing strenght" or "weight" of each inference_kernel
    """
    def __init__(self, operand_set: OperandEnum, a, b=None) -> None:
        if operand_set is not None and not isinstance(operand_set, OperandEnum):
            raise TypeError(f"Expected type OperandEnum for 'operand'. Got {type(operand_set)}")
        self.operand_set = operand_set
        self.a = a
        self.b = b

    def __call__(self, input_kernel_membership) -> float:
        pass

    def _resolve(self, x: Union[Rules, Dict[str, str]], input_kernel_set) -> float:
        if isinstance(x, RuleBase):
            return x(input_kernel_set)
        variable, membership = x.popitem()
        return input_kernel_set[variable][membership]

class AND(RuleBase):

    def __init__(self, a: Union[RuleBase, Dict[str, str]], b: Union[RuleBase, Dict[str, str]], operand_set: OperandEnum = None) -> float:
        super().__init__(operand_set, a, b)

    def __call__(self, input_kernel_membership) -> float:
        a = self._resolve(self.a, input_kernel_membership)
        b = self._resolve(self.b, input_kernel_membership)
        func = self.operand_set.value[0]
        print('AND:', a, b, '->', func(a, b))
        return func(a, b)

class OR(RuleBase):
    def __init__(self, a: Union[RuleBase, Dict[str, str]], b: Union[RuleBase, Dict[str, str]], operand_set: OperandEnum = None) -> float:
        super().__init__(operand_set, a, b)

    def __call__(self, input_kernel_membership) -> float:
        a = self._resolve(self.a, input_kernel_membership)
        b = self._resolve(self.b, input_kernel_membership)
        func = self.operand_set.value[1]
        print('OR:', a, b, '->', func(a, b))
        return func(a, b)

class NOT(RuleBase):
    def __init__(self, a: Union[RuleBase, Dict[str, str]], operand_set: OperandEnum = None) -> float:
        super().__init__(operand_set, a)

    def __call__(self, input_kernel_set) -> float:
        a = self._resolve(self.a, input_kernel_set)
        func = self.operand_set.value[2]
        print('NOT:', a, '->', func(a))
        return func(a)
