""" The machine to run the fuzzy logic """

from typing import Any, Dict

from black import Union

from fuzzy_machines.operands import OperandEnum

from .kernel import Kernel
from .rules import RuleBase


class Engine:
    def __init__(self, operands: OperandEnum = OperandEnum.DEFAULT) -> None:
        # initialization
        self.operands = operands

        # builder
        self.input_kernel_set: Dict[str, Kernel] = {}
        self.inference_kernel: Kernel = None
        self.ruleset: Dict[str, Union[RuleBase, Dict[str, str]]] = {}

        # results
        self.input_kernel_set_values: Dict[str, Dict[str, float]] = {}
        self.fuzzy_res: Dict[str, float] = {}
        self.defuzzy_res: float = None

    def __repr__(self) -> str:
        return str(self.__dict__)

    def add_kernel(self, variable: str, kernel: Kernel):
        """[summary]

        Args:
            name (str): [description]
            func (Callable[..., float]): [description]

        Raises:
            TypeError: [description]
        """
        _typecheck(variable, kernel)
        if not self.input_kernel_set:
            self.input_kernel_set = dict({variable: kernel})
        elif isinstance(self.input_kernel_set, dict):
            self.input_kernel_set[variable] = kernel
        else:
            raise TypeError(
                f"Expected self.rules to be None or dict. Found {type(self.input_kernel_set)}"
            )
        return self

    def del_kernel(self, name: str):
        try:
            del self.input_kernel_set[name]
        except KeyError:
            raise KeyError(f"{name} not found in rules dict")

    def add_inference_kernel(self, variable, kernel: Kernel):
        _typecheck(variable, kernel)
        self.inference_kernel = kernel
        return self

    def del_inference_kernel(self):
        self.inference_kernel = None

    def add_rule(self, name: str, rule: Union[RuleBase, Dict[str, str]]):
        """Add a declarative rule, mapping each input kernel membership values to the inference system fuzzy results. A rule maps the key value of the inference system kernel to a specific RuleBase.

        Args:
            rules (List[Dict[str, RuleBase]]): The list of rules to be added. Each rule has the format {'inference_membership_key': Rule()}.

        Example:
            - (i) If FOOD is GOOD then tip is High: fm.add_rule('High', {'food': 'good'})
            - (ii) if SERVICE is BAD AND FOOD is RANCID then tip is Low: fm.add_rule('Low', AND({'service': 'bad'}, {'food': 'rancid'}))

        See More:
            - RuleBase documentation

        Returns:
            [Engine]: self
        """
        if isinstance(rule, RuleBase):
            self._inject_operands(rule)
        self.ruleset[name] = rule
        return self

    def delete_rule(self, name: str):
        try:
            del self.ruleset[name]
        except KeyError:
            raise KeyError(f"{name} not found in rules dict")

    def fuzzyfy(self, measurements: Dict[str, Any]):
        # TODO: add recursive functions with a depends_on parameter. This will require adding an EngineMeta class or other interface-like function declaration.
        if self.input_kernel_set.keys() != measurements.keys():
            raise KeyError(
                "Could not match the ruleset data to registered ruleset functions.\nruleset_data:"
                f" {measurements.keys()}\nruleset: {self.input_kernel_set.keys()}"
            )

        # run kernel and store membership values for all KernelFuncMember
        input_kernel_membership = {}
        for kkey, kernel in self.input_kernel_set.items():
            input_kernel_membership[kkey] = kernel(measurements[kkey])

        for rkey, rule in self.ruleset.items():
            if isinstance(rule, dict):
                assert len(rule) == 1
                variable, membership = rule.popitem()
                self.fuzzy_res[rkey] = input_kernel_membership[variable][membership]
            elif isinstance(rule, RuleBase):
                self.fuzzy_res[rkey] = rule(input_kernel_membership)

        return self.fuzzy_res

    def defuzzyfy(self):
        # TODO:
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
