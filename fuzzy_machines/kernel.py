"""Inference System and Membership Function classes. Building blocks of a Fuzzy Machine"""

from typing import List, Dict, Any
import numpy as np

from fuzzy_machines.memb_funcs import FunctionBase

def _clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value

class KernelFuncMember:
    """
    KernelFuncMember is a wrapper to FunctionBase, providing two additional methods:
        - __call__: clamps the results of any function between 0 and 1 (inclusive ends)
        - iterate(): a for_loop traversing the KernelFuncMember for a range of values. Required for generating surfaces at the Engine. 
    """
    def __init__(self, func: FunctionBase) -> None:
        if not isinstance(func, FunctionBase):
            raise TypeError(f"Expected type FunctionBase for 'func'. Got {type(func)}")
        self.func = func

    def __call__(self, val) -> float:
        return _clamp(self.func(val), 0.,1.)

    def iterate(self, min_v: float, max_v: float, sample_size: int) -> List[float]:
        res = []
        for val in np.linspace(min_v, max_v, sample_size):
            res.append(self.__call__(val))
        return res

class Kernel:
    def __init__(self, min_v: float, max_v: float) -> None:
        self.min_v = min_v
        self.max_v = max_v
        self.input_functions: Dict[str, KernelFuncMember] = None
        self.input_membership: Dict[str, float] = None

    # NOTE: https://www.sciencedirect.com/topics/engineering/fuzzification
    def __call__(self, measurement: Any):
        # NOTE: all input_membership_functions must "consume" the same type of data.
        self.input_membership = dict()
        for key, func in self.input_functions.items():
            res = func(measurement)
            print(res)
            self.input_membership[key] = res
        return self.input_membership

    def add_memb_func(self, var_name: str, func: KernelFuncMember):
        # NOTE: Do all membership functions must have some overlapping areas??
        if not isinstance(var_name, str):
            raise TypeError(f"Expected type str for 'variable'. Got {type(var_name)}")
        if not isinstance(func, KernelFuncMember):
            raise TypeError(
                f"Expected type KernelFuncMember for 'func'. Got {type(func)}"
            )
        if not self.input_functions:
            self.input_functions = dict({var_name: func})
        elif isinstance(self.input_functions, dict):
            self.input_functions[var_name] = func
        else:
            raise TypeError(
                f"Expected self.input_functions to be None or dict. Found {type(self.input_functions)}"
        )
        return self

    def del_memb_func(self, var_name):
        try:
            del self.input_functions[var_name]
        except KeyError:
            raise KeyError(f"{var_name} not found in rules dict")
    
    def describe(self, sample_size):
        """ Plots 1-d function outputs for every memb function """
        res = {}
        for name, kernel_func in self.input_functions.items():
            res[name] = kernel_func.iterate(self.min_v, self.max_v, sample_size)
        return res
