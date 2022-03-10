""" Membership function types, ranging from constant, linear or other more complex shape mapping """
# pylint: disable=fixme, invalid-name, R0903
from typing import Any, List, Tuple
import numpy as np


def _integral_approximation(a: float, b: float, f: np.ndarray):
    return (b - a) * np.mean(f)


def _call_end(data: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(data), 0, data)


class MembershipFunction:
    """Function Meta"""

    def __init__(self, min_v: float, max_v: float) -> None:
        self.min_v = float(min_v)
        self.max_v = float(max_v)

    def __call__(self, data: Any, activation: float) -> np.ndarray:
        if not 0.0 <= activation <= 1.0:
            raise ValueError(f"Expected activation to be between 0 and 1. Received {activation}")
        data = np.asfarray(data)
        data = np.where(data > self.max_v, np.nan, data)
        data = np.where(data < self.min_v, np.nan, data)
        return data

    def describe(self) -> Tuple[np.ndarray, np.ndarray]:
        """describe is an function-optimized iterator, from min_v to max_v returning a tuple of \
            ndarrays as X and Y values. Useful for plotting a line graph and visualizing \
            the membership function shape

        Raises:
            NotImplementedError: means the membership function is missing the describe method

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple of X-Y value arrays
        """
        raise NotImplementedError

    def naive_describe(
        self, activation: float, granularity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """same as describe, but not optmize. Draws an iterator with linear values between \
            min_v and max_v and a certain granularity.

        Args:
            activation (float): max y-value for any x-value
            granularity (float): defines the linear iteration sample size

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple of X-Y value arrays.
        """
        sample_size = round((self.max_v - self.min_v) / granularity)
        x_range = np.linspace(self.min_v, self.max_v, sample_size)
        y_range = self.__call__(x_range, activation)
        return x_range, y_range

    def optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        """Optimal integration strategy. Defined at subclass"""
        raise NotImplementedError

    def naive_integration(
        self,
        activation: float,
        granularity: float,
    ) -> List[Tuple[float, float, np.ndarray]]:
        """naive linear integration traversing the entire function.

        Args:
            granularity (float): sample size to numerical integration approximation

        Returns:
            List[Tuple[float, float, np.ndarray]]: List of Tuple object containing (a, b, f(x)) \
                for numerical integration
        """
        desc = self.naive_describe(activation, granularity)
        return [(self.min_v, self.max_v, desc[1])]

    def area(self, activation=1.0, granularity=0.01):
        """Numerical integration to calculate area

        Args:
            granularity ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        try:
            res = self.optimal_integration(activation)
        except NotImplementedError:
            res = self.naive_integration(activation, granularity)

        area = 0.0
        for v_min, v_max, y_res in res:
            area += _integral_approximation(v_min, v_max, y_res)

        return area


class Singleton(MembershipFunction):
    """Boolean function. Return 1 when x == value and 0 otherwise"""

    def __init__(self, value: float) -> None:
        super().__init__(float("-inf"), float("+inf"))
        self.value = value

    def __call__(self, data: np.ndarray, activation=1.0) -> np.ndarray:
        data = super().__call__(data, activation)
        res_val = activation if activation is not None else 1.0
        return _call_end(np.where(data == self.value, res_val, 0))

    def describe(self):
        x_array = np.array([self.value])
        y_array = self.__call__(x_array)
        return x_array, y_array

    def optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        raise TypeError("Cannot calculate area or perform integration on singleton type")


class Constant(MembershipFunction):
    """Constant function. Returns the initialization value"""

    # def __init__(self, v_min: float, v_max: float) -> None:
    #     super().__init__(v_min, v_max)

    def __call__(self, data: np.ndarray, activation=1.0) -> np.ndarray:
        data = super().__call__(data, activation)
        y_res = np.where(np.isnan(data), np.nan, activation)
        return _call_end(y_res)

    def describe(self):
        x_array = np.array([self.min_v, self.max_v])
        y_array = self.__call__(x_array)
        return x_array, y_array

    def optimal_integration(self, activation) -> List[Tuple[float, float, np.ndarray]]:
        y_arr = self.__call__(self.min_v, activation)
        return [(self.min_v, self.max_v, np.full(1, y_arr))]


class Linear(MembershipFunction):
    """Linear function"""

    def __init__(self, y_eq_zero: float, y_eq_one: float) -> None:
        if y_eq_zero == y_eq_one:
            raise ValueError("x_intersect cannot be equal to y_eq_one")
        if y_eq_zero < y_eq_one:
            super().__init__(y_eq_zero, y_eq_one)
        else:
            super().__init__(y_eq_one, y_eq_zero)
        if abs(float(y_eq_zero)) == float("inf") or abs(float(y_eq_one)) == float("inf"):
            raise ValueError("Linear function is not defined for infinite intersects")

        self.slope = 1 / (y_eq_one - y_eq_zero)
        self.b = 1 - (self.slope * y_eq_one)

    def __call__(self, data: np.ndarray, activation=1.0) -> np.ndarray:
        data = super().__call__(data, activation)
        y_res = data * self.slope + self.b
        return _call_end(np.where(y_res > activation, activation, y_res))

    def describe(self):
        x_array = np.array([self.min_v, self.max_v])
        y_array = self.__call__(x_array)
        return x_array, y_array

    def optimal_integration(self, activation) -> List[Tuple[float, float, np.ndarray]]:
        x = (activation - self.b) / self.slope
        if x in (self.min_v, self.max_v):
            y_arr = self.__call__(np.array([self.min_v, self.max_v]), activation)
            return [(self.min_v, self.max_v, y_arr)]
        if self.min_v < x < self.max_v:
            res = []
            y_arr1 = self.__call__(np.array([self.min_v, x]), activation)
            res.append([self.min_v, x, y_arr1])
            y_arr2 = self.__call__(np.array([x, self.max_v]), activation)
            res.append([x, self.max_v, y_arr2])
            return res
        raise NotImplementedError


# class Smf(FunctionBase):
#     """S-shaped membership function"""


# class Pimf(FunctionBase):
#     """Pi-shaped membership function"""


# class Zmf(FunctionBase):
#     """Z-shaped membership function"""


class Trimf(MembershipFunction):
    """Triangular membership function"""

    def __init__(self, bottom1: float, peak: float, bottom2: float) -> None:
        super().__init__(bottom1, bottom2)
        assert bottom1 < peak < bottom2, "Cannot draw triangule from initialization parameters"
        self.up = Linear(bottom1, peak)
        self.down = Linear(bottom2, peak)

    def __call__(self, data: np.ndarray, activation=1.0) -> np.ndarray:
        data = np.asfarray(data)
        index = np.arange(data.size)
        indexed_data = np.column_stack((index, data))
        mask = indexed_data[:, 1] <= self.up.max_v
        dt_up = indexed_data[mask, :]
        res_up = np.column_stack((dt_up[:, 0], self.up(dt_up[:, 1], activation)))
        dt_down = indexed_data[~mask, :]
        res_down = np.column_stack((dt_down[:, 0], self.down(dt_down[:, 1], activation)))
        all_res = np.append(res_up, res_down, axis=0)
        all_res_sorted = all_res[all_res[:, 0].argsort()]
        return _call_end(all_res_sorted[:, 1])

    def describe(self):
        x_up, y_up = self.up.describe()
        x_down, y_down = self.down.describe()
        return np.append(x_up, x_down), np.append(y_up, y_down)

    def optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        up_opt = self.up.optimal_integration(activation)
        down_opt = self.down.optimal_integration(activation)
        return up_opt + down_opt


class Trapmf(MembershipFunction):
    """Trapezoidal membership function"""

    def __init__(self, bottom1: float, top1: float, top2: float, bottom2: float) -> None:
        super().__init__(bottom1, bottom2)
        assert (
            bottom1 < top1 < top2 < bottom2
        ), "Cannot draw trapezoid from initialization parameters"
        self.up = Linear(bottom1, top1)
        self.cons = Constant(top1, top2)
        self.down = Linear(bottom2, top2)

    def __call__(self, data: np.ndarray, activation=1.0) -> np.ndarray:
        data = np.asfarray(data)
        index = np.arange(data.size)
        indexed_data = np.column_stack((index, data))
        mask1 = indexed_data[:, 1] <= self.up.max_v
        mask2 = np.logical_and(
            self.up.max_v < indexed_data[:, 1], indexed_data[:, 1] <= self.down.min_v
        )
        mask3 = indexed_data[:, 1] > self.down.min_v
        dt_up = indexed_data[mask1, :]
        res_up = np.column_stack((dt_up[:, 0], self.up(dt_up[:, 1], activation)))
        dt_cons = indexed_data[mask2, :]
        res_cons = np.column_stack((dt_cons[:, 0], self.cons(dt_cons[:, 1], activation)))
        dt_down = indexed_data[mask3, :]
        res_down = np.column_stack((dt_down[:, 0], self.down(dt_down[:, 1], activation)))
        all_res = np.concatenate((res_up, res_cons, res_down), axis=0)
        all_res_sorted = all_res[all_res[:, 0].argsort()]
        return _call_end(all_res_sorted[:, 1])

    def describe(self):
        x_up, y_up = self.up.describe()
        x_cons, y_cons = self.cons.describe()
        x_down, y_down = self.down.describe()
        xs = np.concatenate((x_up, x_cons, x_down))
        ys = np.concatenate((y_up, y_cons, y_down))
        return xs, ys

    def optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        up_opt = self.up.optimal_integration(activation)
        cons_opt = self.cons.optimal_integration(activation)
        down_opt = self.down.optimal_integration(activation)
        return up_opt + cons_opt + down_opt


# class Gaussmf(FunctionBase):
#     """Gaussian membership function"""

#     def __init__(self, min_v: float, max_v: float) -> None:
#         raise NotImplementedError

#     def __call__(self, data: Any, activation: float) -> np.ndarray:
#         raise NotImplementedError

#     def _call_end(self, data: np.ndarray) -> np.ndarray:
#         raise NotImplementedError

#     def describe(self) -> Tuple[np.ndarray, np.ndarray]:
#         raise NotImplementedError

#     def optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
#         """Optimal integration strategy. Defined at subclass"""
#         raise NotImplementedError


# class Gauss2mf(FunctionBase):
#     """Gaussian combination membership function"""

#     def __init__(self, min_v: float, max_v: float) -> None:
#         raise NotImplementedError

#     def __call__(self, data: Any, activation: float) -> np.ndarray:
#         raise NotImplementedError

#     def _call_end(self, data: np.ndarray) -> np.ndarray:
#         raise NotImplementedError

#     def describe(self) -> Tuple[np.ndarray, np.ndarray]:
#         raise NotImplementedError

#     def optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
#         """Optimal integration strategy. Defined at subclass"""
#         raise NotImplementedError


# class Gbellmf:
#     """Generalized bell-shaped membership function"""

#     def __init__(self, min_v: float, max_v: float) -> None:
#         raise NotImplementedError

#     def __call__(self, data: Any, activation: float) -> np.ndarray:
#         raise NotImplementedError

#     def _call_end(self, data: np.ndarray) -> np.ndarray:
#         raise NotImplementedError

#     def describe(self) -> Tuple[np.ndarray, np.ndarray]:
#         raise NotImplementedError

#     def optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
#         """Optimal integration strategy. Defined at subclass"""
#         raise NotImplementedError
