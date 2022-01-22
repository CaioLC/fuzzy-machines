""" Membership function types, ranging from constant, linear or other more complex shape mapping """
# pylint: disable=fixme, invalid-name, R0903
from typing import Any, List, Tuple, cast
from warnings import warn
import numpy as np

def _integral_approximation(a:float, b:float, f:np.ndarray):
    return (b-a)*np.mean(f)

class FunctionBase:
    """Function Meta"""
    def __init__(self, v_min:float, v_max:float) -> None:
        self.v_min = v_min
        self.v_max = v_max

    def __call__(self, data: Any) -> np.ndarray:
        data = np.asfarray(data)
        data = np.where(data > self.v_max, np.nan, data)
        data = np.where(data < self.v_min, np.nan, data)
        return data

    def _call_end(self, data: np.ndarray) -> np.ndarray:
        return np.where(data == np.nan, 0, data)

    def _optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        """Optimal integration strategy. Defined at subclass"""
        return NotImplementedError

    def _naive_integration(self, granularity: float, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        """naive linear integration traversing the entire function.

        Args:
            granularity (float): sample size to numerical integration approximation
        
        Returns:
            List[Tuple[float, float, np.ndarray]]: List of Tuple object containing (a, b, f(x)) for numerical integration
        """
        sample_size = round((self.v_max - self.v_min) / granularity)
        x_range = np.linspace(self.v_min, self.v_max, sample_size)
        y_range = self.__call__(x_range, activation)
        return [(self.v_min, self.v_max, y_range)]

    def area(self, granularity):
        """Numerical integration to calculate area

        Args:
            granularity ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        try:
            y_range_list = self._optimal_integration()
        except NotADirectoryError:
            y_range_list = self._naive_integration(granularity)

        area = 0.
        for a, b, y_range in y_range_list:
            area += _integral_approximation(y_range, a, b)
        
        return area

class Singleton(FunctionBase):
    """Boolean function. Return 1 when x == value and 0 otherwise"""
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, data: np.ndarray, activation: float = None) -> np.ndarray:
        data = super().__call__(data)
        res_val = activation if activation is not None else 1.
        return super()._call_end(np.where(data == self.value, res_val, 0))

    def _optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        raise TypeError("Cannot calculate area or perform integration on singleton type")


class Constant(FunctionBase):
    """Constant function. Returns the initialization value"""

    def __init__(self, const_value: float, v_min: float, v_max: float) -> None:
        super().__init__(v_min, v_max)
        self.const_value = const_value

    def __call__(self, data: np.ndarray, activation: float = None) -> np.ndarray:
        data = super().__call__(data)
        res_val = activation if activation is not None else self.const_value
        return super()._call_end(np.where(data == np.nan, np.nan, res_val))

    def _optimal_integration(self, activation) -> List[Tuple[float, float, np.ndarray]]:
        y_arr = (self.v_max - self.v_min) * activation
        return [(self.v_min, self.v_max, np.full(1, y_arr))]


class Linear(FunctionBase):
    """Linear function"""
    def __init__(self, x_intersect: float, y_eq_one: float) -> None:
        if x_intersect == y_eq_one:
            raise ValueError("x_intersect cannot be equal to y_eq_one")
        if x_intersect < y_eq_one:
            super().__init__(x_intersect, y_eq_one)
        else:
            super().__init__(y_eq_one, x_intersect)
        self.slope = 1/(y_eq_one - x_intersect)
        self.b = self.slope * x_intersect

    def __call__(self, data: np.ndarray, activation: float = None) -> np.ndarray:
        data = super().__call__(data)
        y_res = data * self.slope + self.b
        if activation:
            return super()._call_end(np.where(y_res > activation, activation, y_res))
        return super()._call_end(y_res)

    def _optimal_integration(self, activation) -> List[Tuple[float, float, np.ndarray]]:
        x = (activation - self.b) / self.slope
        if (x == self.v_min and self.slope > 0.) or (x == self.v_max and self.slope < 0.):
            y_arr = (self.v_max - self.v_min) * activation
            return [(self.v_min, self.v_max, np.full(1, y_arr))]
        if (x == self.v_max and self.slope > 0) or (x == self.v_min and self.slope < 0.):
            y_arr = self.__call__(np.array([self.v_min, self.v_max]))
            return [(self.v_min, self.v_max, np.full(1, y_arr))]
        if self.v_min < x < self.v_max:
            res = []
            if self.slope > 0:
                y_arr1 = self.__call__(np.array([self.v_min, x]))
                res.append(self.v_min, x, y_arr1)
                y_arr2 = (self.v_max - x) * activation
                res.append(x, self.v_max, y_arr2)
                return res
            if self.slope < 0:
                y_arr1 = (x - self.v_min) * activation
                res.append(self.v_min, x, y_arr1)
                y_arr2 = self.__call__(np.array([x, self.v_max]))
                res.append(x, self.v_max, y_arr2)
                return res
        return NotImplementedError()


class Smf(FunctionBase):
    """S-shaped membership function"""


class Pimf(FunctionBase):
    """Pi-shaped membership function"""


class Zmf(FunctionBase):
    """Z-shaped membership function"""


class Trimf(FunctionBase):
    """Triangular membership function"""
    def __init__(self, bottom1: float, peak:float, bottom2: float) -> None:
        super().__init__(bottom1, bottom2)
        self.up = Linear(bottom1, peak)
        self.down = Linear(bottom2, peak)

    def __call__(self, data: np.ndarray, activation: float = None) -> np.ndarray:
        data = super().__call__(data)
        index = np.arange(data.size)
        indexed_data = np.append(index, data).reshape(2,len(data))
        mask = indexed_data[1] <= self.up.v_max
        dt_up = indexed_data[:, mask]
        res_up = np.array([dt_up[0],self.up(dt_up[1], activation)]) 
        dt_down = indexed_data[:, ~mask]
        res_down = np.array([dt_down[0],self.down(dt_down[1], activation)])
        all_res = np.sort(np.append(res_up, res_down, axis=1))
        return super()._call_end(all_res[1])
    
    def _optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        up_opt = self.up._optimal_integration(activation)
        down_opt = self.down._optimal_integration(activation)
        return up_opt + down_opt


class Trapmf(FunctionBase):
    """Trapezoidal membership function"""
    def __init__(self, bottom1: float, top1: float, top2: float, bottom2: float) -> None:
        super().__init__(bottom1, bottom2)
        self.up = Linear(bottom1, top1)
        self.cons = Constant(1, top1, top2)
        self.down = Linear(bottom2, top2)

    def __call__(self, data: np.ndarray, activation: float = None) -> np.ndarray:
        data = super().__call__(data)
        index = np.arange(data.size)
        indexed_data = np.append(index, data).reshape(2,len(data))
        mask1 = indexed_data[1] <= self.up.v_max
        mask2 = self.up.v_max < indexed_data[1] <= self.down.v_min
        mask3 = self.down.v_min < indexed_data[1] <= self.down.v_max
        dt_up = indexed_data[:, mask1]
        res_up = np.array([dt_up[0],self.up(dt_up[1], activation)]) 
        dt_cons = indexed_data[:, mask2]
        res_cons = np.array([dt_cons[0],self.cons(dt_cons[1], activation)]) 
        dt_down = indexed_data[:, mask3]
        res_down = np.array([dt_down[0],self.down(dt_down[1], activation)])
        all_res = np.sort(np.concatenate((res_up, res_cons, res_down), axis=1))
        return super()._call_end(all_res[1])

    def _optimal_integration(self, activation: float) -> List[Tuple[float, float, np.ndarray]]:
        up_opt = self.up._optimal_integration(activation)
        cons_opt = self.cons._optimal_integration(activation)
        down_opt = self.down._optimal_integration(activation)
        return up_opt + cons_opt + down_opt

class Gaussmf(FunctionBase):
    """Gaussian membership function"""


class Gauss2mf(FunctionBase):
    """Gaussian combination membership function"""


class Gbellmf:
    """Generalized bell-shaped membership function"""
