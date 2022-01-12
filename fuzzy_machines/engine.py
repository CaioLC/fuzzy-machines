""" The machine to run the fuzzy logic """

from types import FunctionType
from typing import Any, Callable, Dict


class Kernel:
    # NOTE: https://www.sciencedirect.com/topics/engineering/fuzzification
    pass


class Engine:
    def __init__(self, parallel: bool = False) -> None:
        self.ruleset = None
        self.parallel = parallel
        self.fuzzy_res = []

    def __repr__(self) -> str:
        return str(self.__dict__)

    def add_rule(self, variable: str, rule_map: Callable[..., float]):
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
        if not self.ruleset:
            self.ruleset = dict({variable: rule_map})
        elif isinstance(self.ruleset, dict):
            self.ruleset[variable] = rule_map
        else:
            raise TypeError(
                f"Expected self.rules to be None or dict. Found {type(self.ruleset)}"
            )

    def delete_rule(self, name: str):
        try:
            del self.ruleset[name]
        except KeyError:
            raise KeyError(f"{name} not found in rules dict")

    def fuzzify(self, ruleset_data: Dict[str, Any]):
        # TODO: add recursive functions with a depends_on parameter. This will require adding an EngineMeta class or other interface-like function declaration.
        if self.ruleset.keys() != ruleset_data.keys():
            raise ValueError(
                f"Could not match the ruleset data to registered ruleset functions.\nruleset_data: {ruleset_data.keys()}\nruleset: {self.ruleset.keys()}"
            )
        self.fuzzy_res = []
        for key, func in self.ruleset.items():
            self.fuzzy_res.append(func(ruleset_data[key]))
        return self.fuzzy_res

    def defuzzyfy(self):
        if len(self.ruleset) != len(self.fuzzy_res):
            raise ValueError(
                f"Could not match fuzzy results to registered ruleset.\nruleset: {self.ruleset.keys()}\nfuzzy_res: {self.fuzzy_res}"
            )
        # TODO: check how to actually defuzzyfy
        return sum(self.fuzzy_res)

    # TODO: generate surface points.
    # def gen_surface(self):
    #     pass
