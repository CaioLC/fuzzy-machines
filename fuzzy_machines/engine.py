""" The machine to run the fuzzy logic """

from ast import operator
import math
from types import FunctionType
from typing import Any, Callable, Dict


class KernelFuncMember:
    def __init__(self, func, min_v, max_v, discrete, granularity) -> None:
        self.func = func,
        self.min_v = min_v,
        self.max_v = max_v,
        self.discrete = discrete,
        self.granularity = granularity

    def __call__(self) -> float:
        return math.clamp(self.func(), 0,1) # TODO: clamp function result between 0 a 1

    def max_centroid_val(self) -> float:
        raise NotImplementedError


class Kernel:
    def __init__(self) -> None:
        self.input_functions = None
        self.input_membership = None

    # NOTE: https://www.sciencedirect.com/topics/engineering/fuzzification
    def __call__(self, measurement: Any):
        # NOTE: all input_membership_functions must "consume" the same type of data.
        self.input_membership = []
        for func in self.input_functions.values():
            self.input_membership.append(func(measurement))
        return self.input_membership

    def add_memb_func():
        # NOTE: Do all membership functions must have some overlapping areas??
        raise NotImplementedError

    def del_memb_func():
        raise NotImplementedError
    
    def describe():
        """ Plots 1-d function outputs for every memb function """
        raise NotImplementedError

class Rule:
    pass

class Engine:
    def __init__(self, operators: str ='simple') -> None:
        self.input_kernel_set: Dict[str, Kernel] = None
        self.output_kernel: Kernel = None
        self.rule_set = Dict[str, Rule] # do we need a class rule?
        self.operators = operators
        self.fuzzy_res = []

    def __repr__(self) -> str:
        return str(self.__dict__)

    def add_kernel(self, variable: str, rule_map: Callable[..., float]):
        """[summary]

        Args:
            name (str): [description]
            func (Callable[..., float]): [description]

        Raises:
            TypeError: [description]
        """
        if not isinstance(variable, str):
            raise TypeError(f"Expected type str for 'variable'. Got {type(variable)}")
        if not isinstance(rule_map, FunctionType):
            raise TypeError(
                f"Expected type FunctionType for 'rule_map'. Got {type(rule_map)}"
            )
        if not self.input_kernel_set:
            self.input_kernel_set = dict({variable: rule_map})
        elif isinstance(self.input_kernel_set, dict):
            self.input_kernel_set[variable] = rule_map
        else:
            raise TypeError(
                f"Expected self.rules to be None or dict. Found {type(self.input_kernel_set)}"
            )

    def delete_kernel(self, name: str):
        try:
            del self.input_kernel_set[name]
        except KeyError:
            raise KeyError(f"{name} not found in rules dict")

    def add_rule(self, name: str):
        """ 
        examples: 
            - If SERVICE (input_kernel) is GOOD (input_memb) then -> tip is AVERAGE (output_kernel: avg_kernel_func(good))
            - If SERVICE (input_kernel) is POOR (input_memb) OR (operator) FOOD (input_kernel) is RANCID (input_memb) then -> tip is LOW (output_kernel: low_kernel_func(OR(poor, rancid)))
        this function defines a dict of {'low': low_kernel_func, 'average': avg_kernel_func, 'high': high_kernel_func }
        returns the "firing strenght" or "weight" of each inference_kernel"""        

        self.rule_set['tip_poor'] =  (self.input_kernel_set['service']['poor'] || self.kernel_set['food']['rancid'])
        self.rule_set['tip_average'] = self.input_kernel_set['service']['good']
        self.rule_set['tip_poor'] =  (self.input_kernel_set['service']['poor'] && self.kernel_set['food']['rancid'])
        raise NotImplementedError

    def delete_rule(self):
        raise NotImplementedError

    def fuzzify(self, measurements: Dict[str, Any]):
        # TODO: add recursive functions with a depends_on parameter. This will require adding an EngineMeta class or other interface-like function declaration.
        if self.input_kernel_set.keys() != measurements.keys():
            raise ValueError(
                f"Could not match the ruleset data to registered ruleset functions.\nruleset_data: {measurements.keys()}\nruleset: {self.input_kernel_set.keys()}"
            )
        self.fuzzy_res = []
        for key, func in self.input_kernel_set.items():
            self.fuzzy_res.append(func(measurements[key]))
        return self.fuzzy_res

    def infer_membership(self):
        if len(self.input_kernel_set) != len(self.fuzzy_res):
            raise ValueError(
                f"Could not match fuzzy results to registered ruleset.\nruleset: {self.input_kernel_set.keys()}\nfuzzy_res: {self.fuzzy_res}"
            )
        # TODO: check how to actually defuzzyfy
        return sum(self.fuzzy_res)

    def defuzzyfy(self):
        # TODO: 
        # 1 - draw line at percentage area of each membership figure
        # 2 - build poligon area of all 'filled' areas
        # 3 - return X coordinate of the centroid
        # see https://www.mathworks.com/help/fuzzy/defuzzification-methods.html
        return NotImplementedError

    # TODO: generate surface points.
    # def gen_surface(self):
    #     pass
