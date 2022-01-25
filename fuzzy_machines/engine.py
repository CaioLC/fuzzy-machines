""" The machine to run the fuzzy logic """
# pylint: disable=invalid-name, fixme
from typing import Any, Dict, List, Tuple
from warnings import warn

from black import Union
import numpy as np

from fuzzy_machines.operators import DefuzzEnum, OperatorEnum, RuleAggregationEnum
from fuzzy_machines.kernel import Kernel
from fuzzy_machines.rules import RuleBase


class Engine:
    """
    The Fuzzy Engine wraps the input kernels, rules and inference system to provide the
    fuzzyfy, defuzzyfy and generate surface methods.

    Usage steps:
    1: create engine \n
    2: add input kernels (see Kernel for further reference) \n
    3: add inference system kernel (see Kernel for further reference) \n
    4: add rules to map the input kernels to the inference system \n
    5: call engine.fuzzyfy() to run the system \n
    6: call engine.defuzzyfy() to reduce the fuzzy result to a single float number \n
    7: call engine.gen_surface() to build a iterable cache-like map to greatly reduce time compute 
    """

    def __init__(
            self,
            operands: OperatorEnum = OperatorEnum.DEFAULT,
            rule_agreggation: RuleAggregationEnum = RuleAggregationEnum.MAX,
            defuzz_method: DefuzzEnum = DefuzzEnum.WEIGHTED_AREA
        ) -> None:
        """Initializes a new engine object

        Args:
            operands (OperandEnum, optional): Operation definitions for AND, OR and NOT methods. \
        Defaults to OperandEnum.DEFAULT.
        """
        # initialization
        self.operands = operands
        self.rule_agreggation = rule_agreggation
        self.defuzz_method = defuzz_method

        # builder
        self.input_kernel_set: Dict[str, Kernel] = {}
        self.inference_kernel: Kernel = None
        self.ruleset: Dict[str, List[RuleBase]] = {}

        # results
        self.actuation_signal: Dict[str, float] = {}
        self.membership_degree: Dict[str, float] = {}
        self.defuzzy_res: float = None

    def __repr__(self) -> str:
        """String representation

        Returns:
            str: the engine object
        """
        return str(self.__dict__)

    def add_kernel(self, name: str, kernel: Kernel):
        """Adds a Kernel object, to map a particular variable of interest to its membership
        functions

        Args:
            name (str): The name of the kernel. Various methods will call the kernel using this \
        name as the key to a dict of type {name: kernel}
            kernel (Kernel): Kernel object, mapping the variable to its many membership functions

        Raises:
            TypeError: if the internal dictionary self.input_kernel_set gets corrupted by direct \
        user manipulation

        Returns:
            Engine: self
        """
        _typecheck(name, kernel)
        # if not kernel.check_normalized():
        #     warn(f"Kernel for {name} is not normalized")
        if not self.input_kernel_set:
            self.input_kernel_set = dict({name: kernel})
        elif isinstance(self.input_kernel_set, dict):
            self.input_kernel_set[name] = kernel
        else:
            raise TypeError(
                f"Expected self.rules to be None or dict. Found {type(self.input_kernel_set)}"
            )
        return self

    def del_kernel(self, name: str) -> None:
        """Deletes a registered input kernel

        Args:
            name (str): the name of the registered kernel

        Raises:
            KeyError: if name is not found in self.input_kernel_set.keys()
        """
        try:
            del self.input_kernel_set[name]
        except KeyError as error:
            raise KeyError(f"{name} not found in rules dict") from error

    def add_inference_kernel(self, kernel: Kernel):
        """Adds a Kernel to map the inference system to its membership functions fuzzy output

        Args:
            kernel (Kernel): the kernel object mapping the inference system to membership functions

        Raises:
            TypeError: if object type != type(Kernel)

        Returns:
            Engine: self
        """
        if not isinstance(kernel, Kernel):
            raise TypeError(f"Expected type Kernel for 'kernel'. Got {type(kernel)}")
        self.inference_kernel = kernel
        return self

    def del_inference_kernel(self):
        """Deletes the registered inference kernel, if there is one"""
        self.inference_kernel = None

    def add_rule(self, name: str, rule: RuleBase):
        """Add a declarative rule, mapping each input kernel membership values to the inference
        system membership functions.

        Args:
            name (str): must match the name of a KernelFuncMember registered at the Inference \
        Kernel System.
            rule (Union[RuleBase, Dict[str, str]]): a rule object (AND, OR, NOT), or a dictionary \
        to get the direct value of a specific membership function.

        Examples:
            - (i) If FOOD is GOOD then tip is High (direct access): \
        fm.add_rule('High', {'food': 'good'})
            - (ii) if SERVICE is BAD AND FOOD is RANCID then tip is Low (rule base access): \
        fm.add_rule('Low', AND({'service': 'bad'}, {'food': 'rancid'}))

        See More:
            - RuleBase documentation

        Returns:
            Engine: self
        """
        if not isinstance(rule, RuleBase):
            raise TypeError(f"Expected type RuleBase for 'rule'. Got {type(rule)}")
        if isinstance(rule, RuleBase):
            self._inject_operands(rule)
        if name in self.ruleset.keys():
            self.ruleset[name].append(rule)
        else: 
            self.ruleset[name] = [rule]
        return self

    def delete_rule(self, name: str):
        """Deletes a registered rule at self.ruleset

        Args:
            name (str): the rule name

        Raises:
            KeyError: if name is not in self.ruleset.keys()
        """
        try:
            del self.ruleset[name]
        except KeyError as error:
            raise KeyError(f"{name} not found in rules dict") from error

    def _fuzzyfy(self, measurements: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Fuzzyfy the crisp measurement data for all registered kernels. Result is returned \
            to the user as a nested dictionary but also stored in each kernel object.

        Args:
            measurements for each registered registered input kernel \
        (in the format {'kernel_name': data})

        Raises:
            KeyError: if there's a mismatch between input_kernel_set.keys() and measurements.keys()

        Returns:
            Dict[str, Dict[str, float]]: dict('kernel_name': dict('function_member': value))
        """
        if self.input_kernel_set.keys() != measurements.keys():
            raise KeyError(
                "Could not match the ruleset data to registered ruleset functions.\nruleset_data:"
                f" {measurements.keys()}\nruleset: {self.input_kernel_set.keys()}"
            )

        res = {}
        for kkey, kernel in self.input_kernel_set.items():
            res[kkey] = kernel(measurements[kkey])
        return res

    def _aggregate(self) -> Dict[str, np.ndarray]: # aggregation (running all rules) and returning one value per rule
        for rkey, rulelist in self.ruleset.items():
            agg_actuation = self.rule_agreggation.value(rule(self.input_kernel_set) for rule in rulelist)
            self.actuation_signal[rkey] = np.asfarray(agg_actuation)
        return self.actuation_signal

    def _accumulate(self, granularity) -> Tuple[np.ndarray, np.ndarray]:
        # step 1: find non overlapping area and return minimum xy pairs
        # step 2: iterate through all overlapping areas and return granular xy pairs
        # step 3: join everything and calculate area
        sample_size = round((self.inference_kernel.max_v - self.inference_kernel.min_v) / granularity)
        x_range = np.linspace(self.inference_kernel.min_v, self.inference_kernel.max_v, sample_size)
        y_range = np.zeros(sample_size)
        for rule, func in self.inference_kernel.input_functions.items():
            acc = self.actuation_signal[rule]
            y_proponent = func(x_range, acc)
            y_range = np.maximum(y_range, y_proponent)
        
        return x_range, y_range


    def _defuzzyfy(self, x_range, y_range):
        """Transform the fuzzy result to a numerical float value."""
        y1_range = np.array([np.nan]) + y_range[:-1]
        x1_range = np.array([np.nan]) + x_range[:-1]
        moment_area = (x1_range - x_range) * np.mean(np.array(y1_range, y_range))
        total_area = (self.inference_kernel.max_v - self.inference_kernel.min_v) * np.mean(y_range)
        self.defuzzy_res = moment_area / total_area
        return self.defuzzy_res


    def run_fuzzy(self, measurements: Dict[str, Any]) -> Dict[str, np.ndarray]:
        self._fuzzyfy(measurements)
        return self._aggregate()

    def run_defuzz(self, measurements: Dict[str, Any], granularity) -> Dict[str, np.ndarray]:
        # TODO: add cache
        self._fuzzyfy(measurements)
        self._aggregate()
        x_range, y_range = self._accumulate(granularity)
        self._defuzzyfy(x_range, y_range)
        return self.defuzzy_res

    # TODO: generate surface points.
    # def gen_surface(self):
    #     pass

    def _inject_operands(self, rule: RuleBase):
        rule.operand_set = self.operands
        if isinstance(rule.a, RuleBase):
            self._inject_operands(rule.a)
        if isinstance(rule.b, RuleBase):
            self._inject_operands(rule.b)


def _typecheck(variable: str, kernel: Kernel):
    if not isinstance(variable, str):
        raise TypeError(f"Expected type str for 'variable'. Got {type(variable)}")
    if not isinstance(kernel, Kernel):
        raise TypeError(f"Expected type Kernel for 'kernel'. Got {type(kernel)}")
