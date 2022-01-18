""" The machine to run the fuzzy logic """
# pylint: disable=invalid-name, fixme
from typing import Any, Dict

from black import Union

from fuzzy_machines.operands import OperandEnum

from .kernel import Kernel
from .rules import RuleBase


class Engine:
    """
    The Fuzzy Engine wraps the input kernels, rules and inference system to provide the
    fuzzyfy, defuzzyfy and generate surface methods.

    Usage steps:
    1: create engine
    2: add input kernels (see Kernel for further reference)
    3: add inference system kernel (see Kernel for further reference)
    4: add rules to map the input kernels to the inference system
    5: call engine.fuzzyfy() to run the system
    6: call engine.defuzzyfy() to reduce the fuzzy result to a single float number
    7: call engine.gen_surface() to build a iterable cache-like map to greatly reduce time compute
    """

    def __init__(self, operands: OperandEnum = OperandEnum.DEFAULT) -> None:
        """Initializes a new engine object

        Args:
            operands (OperandEnum, optional): Operation definitions for AND, OR and NOT methods.
        Defaults to OperandEnum.DEFAULT.
        """
        # initialization
        self.operands = operands

        # builder
        self.input_kernel_set: Dict[str, Kernel] = {}
        self.inference_kernel: Kernel = None
        self.ruleset: Dict[str, Union[RuleBase, Dict[str, str]]] = {}

        # results
        self.actuation_signal: Dict[str, Dict[str, float]] = {}
        self.fuzzy_res: Dict[str, float] = {}
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
            name (str): The name of the kernel. Various methods will call the kernel using this
        name as the key to a dict of type {name: kernel}
            kernel (Kernel): Kernel object, mapping the variable to its many membership functions

        Raises:
            TypeError: if the internal dictionary self.input_kernel_set gets corrupted by direct
        user manipulation

        Returns:
            Engine: self
        """
        _typecheck(name, kernel)
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

    def add_rule(self, name: str, rule: Union[RuleBase, Dict[str, str]]):
        """Add a declarative rule, mapping each input kernel membership values to the inference
        system membership functions.

        Args:
            name (str): must match the name of a KernelFuncMember registered at the Inference
        Kernel System.
            rule (Union[RuleBase, Dict[str, str]]): a rule object (AND, OR, NOT), or a dictionary
        to get the direct value of a specific membership function.

        Examples:
            - (i) If FOOD is GOOD then tip is High (direct access):
        fm.add_rule('High', {'food': 'good'})
            - (ii) if SERVICE is BAD AND FOOD is RANCID then tip is Low (rule base access):
        fm.add_rule('Low', AND({'service': 'bad'}, {'food': 'rancid'}))

        See More:
            - RuleBase documentation

        Returns:
            Engine: self
        """
        if isinstance(rule, RuleBase):
            self._inject_operands(rule)
        self.ruleset[name] = rule
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

    def fuzzyfy(self, measurements: Dict[str, Any]) -> Dict[str, float]:
        """
        Main fuzzyfication method. Passing a dictionary of measurements for each registered
        input kernel (in the format {'kernel_name': data}), this methods runs all kernel membership
        functions and store the corresponding values.

        Next, it parses all rules and construct the "actuation signal", which is the actual values
        sent to the inference system.

        Finally, the inference system is run based on the actuation signal.

        Args:
            measurements (Dict[str, Any]): measurements for each registered registered input kernel
        (in the format {'kernel_name': data})

        Raises:
            KeyError: if there's a mismatch between input_kernel_set.keys() and measurements.keys()

        Returns:
            Dict[str, float]: {'fuzzy_func': 'fuzzy_result'}
        """
        if self.input_kernel_set.keys() != measurements.keys():
            raise KeyError(
                "Could not match the ruleset data to registered ruleset functions.\nruleset_data:"
                f" {measurements.keys()}\nruleset: {self.input_kernel_set.keys()}"
            )

        # run kernel and store membership values for all KernelFuncMember
        for kkey, kernel in self.input_kernel_set.items():
            kernel(measurements[kkey])

        for rkey, rule in self.ruleset.items():
            if isinstance(rule, dict):
                assert len(rule) == 1
                variable, membership = rule.popitem()
                self.actuation_signal[rkey] = self.input_kernel_set[variable].input_membership[
                    membership
                ]
            elif isinstance(rule, RuleBase):
                self.actuation_signal[rkey] = rule(self.input_kernel_set)
        self.fuzzy_res = self.inference_kernel(self.actuation_signal)
        return self.fuzzy_res

    def defuzzyfy(self):
        """Transform the fuzzy result to a numerical float value.

        Raises:
            NotImplementedError: [description]
        """
        # TODO: defuzzyfy not yet implemented
        # 1 - draw line at percentage area of each membership figure
        # 2 - build poligon area of all 'filled' areas
        # 3 - return X coordinate of the centroid
        # see https://www.mathworks.com/help/fuzzy/defuzzification-methods.html
        raise NotImplementedError

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
