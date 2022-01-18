"""Inference System and Membership Function classes. Building blocks of a Fuzzy Machine"""

from numbers import Number
from typing import List, Dict, Any
import numpy as np

from fuzzy_machines.memb_funcs import FunctionBase


def _clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


class KernelFuncMember:
    """
    KernelFuncMember is a wrapper to FunctionBase, providing two additional methods: \
        - __call__: clamps the results of any function between 0 and 1 (inclusive ends) \
        - iterate(): a for_loop traversing the KernelFuncMember for a range of values. \
    Required for generating surfaces at the Engine.
    """

    def __init__(self, func: FunctionBase) -> None:
        if not isinstance(func, FunctionBase):
            raise TypeError(f"Expected type FunctionBase for 'func'. Got {type(func)}")
        self.func = func

    def __call__(self, val) -> float:
        return _clamp(self.func(val), 0.0, 1.0)

    def iterate(self, min_v: float, max_v: float, sample_size: int) -> List[float]:
        """Calls the KernelFuncMember with a list of values, which is a linear iteration
        between min value and max value given a certain sample_size.

        Args:
            min_v (float): iteration start
            max_v (float): iteration end (inclusive)
            sample_size (int): number os datapoints

        Returns:
            List[float]: the results for calling the function for each datapoint
        """
        res = []
        for val in np.linspace(min_v, max_v, sample_size):
            res.append(self.__call__(val))
        return res


class Kernel:
    """
    A wrapper that represents all manners a particular variable is mapped to KernelFuncMembers.
    """

    def __init__(self, min_v: float, max_v: float) -> None:
        if not isinstance(min_v, Number):
            raise ValueError(f"expected numeric for 'min_v'. Found {type(min_v)}")
        if not isinstance(max_v, Number):
            raise ValueError(f"expected numeric for 'max_v'. Found {type(max_v)}")
        if not max_v >= min_v:
            raise ValueError("'max_v' must be greater or equal than 'min_v'")
        self.min_v = min_v
        self.max_v = max_v
        self.input_functions: Dict[str, KernelFuncMember] = None
        self.input_membership: Dict[str, float] = None

    # NOTE: https://www.sciencedirect.com/topics/engineering/fuzzification
    def __call__(self, measurement: Any):
        # NOTE: all input_membership_functions must "consume" the same type of data.
        self.input_membership = {}
        for key, func in self.input_functions.items():
            res = func(measurement)
            print(res)
            self.input_membership[key] = res
        return self.input_membership

    def add_memb_func(self, var_name: str, func: KernelFuncMember):
        """Registers a KernelFuncMember as part of the Kernel

        Args:
            var_name (str): the name of the mapping KernelFuncMember
            func (KernelFuncMember): a KernelFuncMember object

        Raises:
            TypeError: if var_name is not str
            TypeError: if func is not KernelFuncMember
            TypeError: if self.input_functions got corrupted and is not Dict

        Returns:
            Kernel: self
        """
        # NOTE: Do all membership functions must have some overlapping areas??
        if not isinstance(var_name, str):
            raise TypeError(f"Expected type str for 'variable'. Got {type(var_name)}")
        if not isinstance(func, KernelFuncMember):
            raise TypeError(f"Expected type KernelFuncMember for 'func'. Got {type(func)}")
        if not self.input_functions:
            self.input_functions = dict({var_name: func})
        elif isinstance(self.input_functions, dict):
            self.input_functions[var_name] = func
        else:
            raise TypeError(
                "Expected self.input_functions to be None or dict. Found"
                f" {type(self.input_functions)}"
            )
        return self

    def del_memb_func(self, var_name):
        """Deletes a registered KernelFuncMember

        Args:
            var_name ([type]): the name of the registered KernelFuncMember

        Raises:
            KeyError: is var_name can't be found at self.input_functions.keys()
        """
        try:
            del self.input_functions[var_name]
        except KeyError as error:
            raise KeyError(f"{var_name} not found in rules dict") from error

    def describe(self, sample_size):
        """Plots 1-d function outputs for every memb function"""
        res = {}
        for name, kernel_func in self.input_functions.items():
            res[name] = kernel_func.iterate(self.min_v, self.max_v, sample_size)
        return res
