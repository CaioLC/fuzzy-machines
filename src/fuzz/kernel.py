"""Inference System and Membership Function classes. Building blocks of a Fuzzy Machine"""

from numbers import Number
from typing import Dict, Any, Tuple
from warnings import warn
import numpy as np

from .memb_funcs import MembershipFunction


class Kernel:
    """
    A wrapper that represents all manners a particular variable is mapped its MFs.
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
        self.input_functions: Dict[str, MembershipFunction] = None
        self.membership_degree: Dict[str, float] = None

    def __call__(self, measurement: Any):
        self.membership_degree = {}
        for key, func in self.input_functions.items():
            res = func(measurement)
            self.membership_degree[key] = res
        return self.membership_degree

    def add_memb_func(self, var_name: str, func: MembershipFunction):
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
        if not isinstance(var_name, str):
            raise TypeError(f"Expected type str for 'variable'. Got {type(var_name)}")
        if not isinstance(func, MembershipFunction):
            raise TypeError(f"Expected type FunctionBase for 'func'. Got {type(func)}")
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

    def describe(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Plots 1-d function outputs for every memb function"""
        res = {}
        for name, kernel_func in self.input_functions.items():
            res[name] = kernel_func.describe()
        return res

    def check_coverage(self) -> bool:
        """Checks if registered MFS cover the entire universe data range"""
        min_k_value = min(v.min_v for v in self.input_functions.values())
        max_k_value = max(v.max_v for v in self.input_functions.values())
        outer_bounds = self.min_v >= min_k_value and self.max_v <= max_k_value
        inners = []
        if len(self.input_functions) > 1:
            for key, func in self.input_functions.items():
                intersect_count = len(
                    [
                        f
                        for k, f in self.input_functions.items()
                        if func.max_v >= f.min_v and func.min_v <= f.max_v and k != key
                    ]
                )
                if intersect_count == 0:
                    warn(
                        UserWarning(
                            f"MembershipFunction '{key}' has no intersections. Variable space may"
                            " not be fully defined"
                        )
                    )
                inners.append(intersect_count)
            inners = np.array(inners)
        else:
            # if there's only one MF, then there's no sense in checking intersections
            # inners defaults to true.
            inners = np.array([True])
        return outer_bounds and inners.all()
