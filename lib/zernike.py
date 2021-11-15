

from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sy



def mn_to_j(
    m: int,
    n: int,
) -> int:
    

    return int((n * (n + 2) + m) / 2)


def j_to_mn(
    j: int,
) -> Tuple[int, int]:
   

    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = int(2 * j - n * (n + 2))

    return m, n


def polar_to_cartesian(
    expression: sy.Expr,
) -> sy.Expr:
    

    rho, phi = sy.symbols('rho'), sy.symbols('phi')
    x, y = sy.symbols('x'), sy.symbols('y')

    substitute_rho = sy.sqrt(x**2 + y**2)
    substitute_phi = sy.atan2(y, x)

    result = deepcopy(expression)
    result = result.subs(rho, substitute_rho)
    result = result.subs(phi, substitute_phi)

    return result


def derive(
    expression: sy.Expr,
    wrt: Union[str, sy.Symbol]
) -> sy.Expr:
    

    if isinstance(wrt, sy.Symbol) and (wrt in expression.free_symbols):
        return sy.diff(expression, wrt)


    elif isinstance(wrt, str):
        for symbol in expression.free_symbols:
            if wrt == symbol.name:
                return sy.diff(expression, symbol)

    return sy.sympify(0)


def is_cartesian(
    expression: sy.Expr
) -> bool:
    

    return {_.name for _ in expression.free_symbols}.issubset({'x', 'y'})


def is_polar(
    expression: sy.Expr
) -> bool:
    

    return {_.name for _ in expression.free_symbols}.issubset({'rho', 'phi'})


def eval_cartesian(
    expression: sy.Expr,
    x_0: Union[float, np.ndarray],
    y_0: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    
    assert is_cartesian(expression), \
        '"expression" is not in Cartesian coordinates!'
    assert ((isinstance(x_0, float) and isinstance(y_0, float)) or
            (isinstance(x_0, np.ndarray) and isinstance(y_0, np.ndarray) and
             x_0.shape == y_0.shape)), \
        '"x_0" and "y_0" must be either both float, or both numpy array ' \
        'with the same shape!'


    if not expression.is_constant():

        numpy_func: Callable[..., Union[float, np.ndarray]] = \
            sy.utilities.lambdify(args=sy.symbols('x, y'),
                                  expr=expression,
                                  modules='numpy')

    
    else:

        def numpy_func(_: float, __: float) -> float:
            return float(expression) * _ / _ * __ / __
        numpy_func = np.vectorize(numpy_func)

    return numpy_func(x_0, y_0)




class ZernikePolynomial:


    def __init__(self,
                 m: Optional[int] = None,
                 n: Optional[int] = None,
                 j: Optional[int] = None):

        error_msg = 'ZernikePolynomial must be instantiated either with ' \
                    'double indices (m, n) *or* a single index j!'
        if j is not None:
            assert m is None, error_msg
            assert n is None, error_msg
            self.j = j
            self.m, self.n = j_to_mn(self.j)
        else:
            assert m is not None, error_msg
            assert n is not None, error_msg
            self.m, self.n = m, n
            self.j = mn_to_j(self.m, self.n)

        assert (-self.n <= self.m <= self.n), \
            'Zernike polynomials are only defined for -n <= m <= n!'
        assert self.j >= 0, \
            'Zernike polynomials are only defined for j >= 0!'

    def __repr__(self) -> str:
        return f'Z^{self.m}_{self.n}'

    @property
    def radial_part(self) -> sy.Expr:
        

        rho = sy.Symbol('rho')

        if (self.n - self.m) % 2 == 1:
            return sy.sympify(0)

        else:
            return sum(sy.Pow(-1, k) * sy.binomial(self.n - k, k) *
                       sy.binomial(self.n - 2 * k, (self.n - self.m) / 2 - k) *
                       sy.Pow(rho, self.n - 2 * k)
                       for k in range(0, int((self.n - self.m) / 2) + 1))

    @property
    def azimuthal_part(self) -> sy.Expr:
        

        phi = sy.Symbol('phi')

        if self.m > 0:
            return sy.cos(self.m * phi)
        elif self.m < 0:
            return sy.sin(-self.m * phi)
        else:
            return sy.sympify(1)

    @property
    def normalization(self) -> sy.Expr:
       
        if self.m == 0:
            return sy.sqrt(self.n + 1)
        else:
            return sy.sqrt(self.n + 1) * sy.sqrt(2)

    @property
    def polar(self) -> sy.Expr:
        
        return self.normalization * self.radial_part * self.azimuthal_part

    @property
    def cartesian(self) -> sy.Expr:
        
        return polar_to_cartesian(self.polar)

    @property
    def fourier_transform(self) -> sy.Expr:
       

        k1 = sy.Symbol('k1')
        k2 = sy.Symbol('k2')

        factor_1 = (sy.Pow(-1, self.n) * sy.sqrt(self.n + 1) / (sy.pi * k1) *
                    sy.besselj(2 * sy.pi * k1, self.n + 1))

        if self.m == 0:
            factor_2 = sy.Pow(-1, self.n / 2)
        elif self.m > 0:
            factor_2 = (sy.sqrt(2) * sy.Pow(-1, (self.n - self.m) / 2) *
                        sy.Pow(sy.I, self.m) * sy.cos(self.m * k2))
        else:
            factor_2 = (sy.sqrt(2) * sy.Pow(-1, (self.n + self.m) / 2) *
                        sy.Pow(sy.I, -self.m) * sy.sin(-self.m * k2))

        return sy.nsimplify(sy.simplify(factor_1 * factor_2))


class Wavefront:
    
    def __init__(self,
                 coefficients: Union[Sequence[float], Dict[int, float]]):

        self.coefficients = coefficients

    @property
    def polar(self) -> sy.Expr:


        if isinstance(self.coefficients, dict):
            return sum(coefficient * ZernikePolynomial(j=j).polar
                       for j, coefficient in self.coefficients.items())
        else:
            return sum(coefficient * ZernikePolynomial(j=j).polar
                       for j, coefficient in enumerate(self.coefficients))

    @property
    def cartesian(self) -> sy.Expr:

        return polar_to_cartesian(self.polar)
