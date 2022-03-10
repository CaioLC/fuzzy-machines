""" The machine to run the fuzzy logic """
# pylint: disable=invalid-name, fixme
from typing import Any, Dict, List, Tuple
from warnings import warn
import numpy as np

from .operators import DefuzzEnum, OperatorEnum, RuleAggregationEnum
from .kernel import Kernel
from .rules import RuleBase


class Engine:
    """
    The Fuzzy Engine wraps the input kernels, rules and inference system to provide the
    fuzzyfy, defuzzyfy and generate surface methods.

    Usage steps:
    1: create engine \n
    2: add input kernels (see Kernel for further reference) \n
    3: add inference system (Kernel or Takagi-Sugeno based) \n
    4: add rules to map the input kernels to the inference system \n
    5: call engine.fuzzyfy() to run the system \n
    6: call engine.defuzzyfy() to reduce the fuzzy result to a single float number \n
    7: call engine.gen_surface() to build a iterable cache-like map to greatly reduce time compute
    """

    def __init__(
        self,
        operands: OperatorEnum = OperatorEnum.DEFAULT,
        rule_agg: RuleAggregationEnum = RuleAggregationEnum.MAX,
        defuzz_method: DefuzzEnum = DefuzzEnum.LINGUISTIC,
    ) -> None:
        """Initializes a new engine object

        Args:
            operands (OperandEnum, optional): Operation definitions for AND, OR and NOT methods. \
        Defaults to OperandEnum.DEFAULT.
        """
        # initialization
        self.operands = operands
        self.rule_agreggation = rule_agg
        self.defuzz_method = defuzz_method

        # builder
        self.input_kernel_set: Dict[str, Kernel] = {}
        self.inference_kernel: Kernel = None
        self.ruleset: Dict[str, List[RuleBase]] = {}

        # results
        self.actuation_signal: Dict[str, float] = {}
        self.membership_degree: Dict[str, float] = {}
        self.defuzzy_res: np.ndarray = None

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
        if name in self.ruleset:
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
        data_len = set()
        for data in measurements.values():
            data_len.add(np.asarray(data).size)
        assert len(data_len) == 1, "Fuzzification expect all measurement data to be of equal lenght"
        res = {}
        for kkey, kernel in self.input_kernel_set.items():
            res[kkey] = kernel(measurements[kkey])
        return res

    def _aggregate(
        self,
    ) -> Dict[str, np.ndarray]:  # aggregation (running all rules) and returning one value per rule
        for rkey, rulelist in self.ruleset.items():
            rule_res = np.asfarray([rule(self.input_kernel_set) for rule in rulelist])
            assert len(rule_res) >= 1, f"rule {rkey} returned no value"
            if rule_res[0].size > 1:
                agg_actuation = np.asfarray(
                    [
                        self.rule_agreggation.value(rule_res[:, col])
                        for col in range(rule_res.shape[1])
                    ]
                )
            else:
                agg_actuation = self.rule_agreggation.value(rule_res)
            self.actuation_signal[rkey] = np.asfarray(agg_actuation)
        return self.actuation_signal

    def _accumulate(self, granularity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        In the accumulation phase, all inference rules are joined to form a single shape. \
        The method traverses the inference system with granularity A and run each of its KMF with \
        max activation set to aggregated rule value for each KMF. \

        This method is implemented for Linguistic Inference Systems only and will raise exception \
            if run with any other defuzzyfication method.

        Args:
            granularity (float): the "step size" of the iterator function

        Raises:
            ValueError: if _accumulate is called by an engine not running linguistic inference sys

        Returns:
            Tuple[np.ndarray, np.ndarray]: a mapping of x_values to y_values as a tuple of ndarray.
        """
        if self.inference_kernel is None:
            raise ValueError("Engine is missing the inference kernel system.")
        if self.defuzz_method != DefuzzEnum.LINGUISTIC:
            raise ValueError(
                f"self._accumulate is not valid for defuzzification method {self.defuzz_method}"
            )
        sample_size = round(
            (self.inference_kernel.max_v - self.inference_kernel.min_v) / granularity
        )
        x_range = np.linspace(self.inference_kernel.min_v, self.inference_kernel.max_v, sample_size)
        y_stack = None
        acc_array = np.array(list(self.actuation_signal.values()))
        if acc_array.ndim == 1:
            y_range = np.zeros(sample_size)
            for rule, func in self.inference_kernel.input_functions.items():
                acc = self.actuation_signal[rule]
                y_proponent = func(x_range, acc)
                y_range = np.maximum(y_range, y_proponent)
            if y_stack is not None:
                y_stack = np.vstack((y_stack, y_range))
            else:
                y_stack = y_range
        else:
            keys = list(self.actuation_signal.keys())
            for acc_point in acc_array.T:
                y_range = np.zeros(sample_size)
                point = dict(zip(keys, acc_point))
                print(point)
                for rule, func in self.inference_kernel.input_functions.items():
                    acc = point[rule]
                    y_proponent = func(x_range, acc)
                    y_range = np.maximum(y_range, y_proponent)
                if y_stack is not None:
                    y_stack = np.vstack((y_stack, y_range))
                else:
                    y_stack = y_range
        # print(y_stack)
        return x_range, y_stack

    def _takagi_sugeno(self, data: Any):
        """Transform the fuzzy result to a numerical float value."""
        return sum(
            func(**data) * self.actuation_signal[rule]
            for rule, func in self.inference_kernel.input_functions.items()
        ) / sum(self.actuation_signal.values())

    def run_fuzz(self, measurements: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Passing a dictionary of data, runs the Engine and return a fuzzy output mapped to the \
            inference system

        Args:
            measurements (Dict[str, Any]): Dictionary mapping each input kernel to its data

        Returns:
            Dict[str, np.ndarray]: the fuzzy output of the inference system.
        """
        self._check_coverage()
        self._fuzzyfy(measurements)
        return self._aggregate()

    def run_defuzz(self, measurements: Dict[str, Any], granularity: float = None) -> np.ndarray:
        """Passing a dictionary of data and granularity (mandatory for Linguistic Defuzz) \
            returns a crisp result of the inference system.

        Args:
            measurements (Dict[str, Any]): Dictionary mapping each input kernel to its data
            granularity (float): iteration granularity. Required for Linguistic Defuzz only.

        Raises:
            NotImplementedError: If Defuzz method is not implemented.

        Returns:
            np.ndarray: crisp values of the inference system.
        """
        self.run_fuzz(measurements)
        if self.defuzz_method == DefuzzEnum.LINGUISTIC:
            assert granularity is not None, "Linguistic Defuzz requires a granularity param"
            x_range, y_range = self._accumulate(granularity)
            self.defuzzy_res = _centroid(x_range, y_range)

        elif self.defuzz_method == DefuzzEnum.TAKAGI_SUGENO:
            self.defuzzy_res = self._takagi_sugeno(measurements)

        else:
            raise NotImplementedError(f"defuzzyfy for {self.defuzz_method} is not implemented")

        return self.defuzzy_res

    def gen_surface(self, map_size: int, granularity: float):
        """Very expensive operation, if used with Linguistic Fuzzy Systems.

        Args:
            granularity (Union[float, Dict[str, float]]): [description]

        Returns:
            [type]: [description]
        """
        raise NotImplementedError  # TODO: gen_surface needs to be redone

    def _inject_operands(self, rule: RuleBase):
        rule.operand_set = self.operands
        if isinstance(rule.a, RuleBase):
            self._inject_operands(rule.a)
        if isinstance(rule.b, RuleBase):
            self._inject_operands(rule.b)

    def _check_coverage(self) -> None:
        coverage = np.array([f.check_coverage() for f in self.input_kernel_set.values()])
        if not coverage.all():
            warn(
                UserWarning(f"Variable description is incomplete. Coverage check: {list(coverage)}")
            )

def _typecheck(variable: str, kernel: Kernel):
    if not isinstance(variable, str):
        raise TypeError(f"Expected type str for 'variable'. Got {type(variable)}")
    if not isinstance(kernel, Kernel):
        raise TypeError(f"Expected type Kernel for 'kernel'. Got {type(kernel)}")


def _centroid(x_range, y_range: np.ndarray):
    """Transform the fuzzy result to a numerical float value."""
    if y_range.ndim == 1:
        return _func1d(y_range, x_range)
    if y_range.ndim == 2:
        return np.array([_func1d(y_row, x_range) for y_row in y_range])
    raise NotImplementedError("_centroid is not defined for arrays with 3 or more dimensions")

def _func1d(y_row, x_values):
    return np.asfarray(np.sum(x_values * y_row) / np.sum(y_row))
