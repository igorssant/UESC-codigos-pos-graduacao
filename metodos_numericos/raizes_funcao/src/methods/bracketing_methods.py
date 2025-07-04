import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable
from utils.math_problems import is_indertemination

def bissection(
    func: Union[str, Callable],
    xl: np.float64,
    xu: np.float64,
    tol: np.float64,
    max_iter: int,
) -> np.float64:
    iter: int = 0
    relative_error: np.float64 = np.float64(100.0)
    x: np.float64 = xl

    fl: np.float64 = evaluate_one_variable(func, xl)

    while (relative_error > tol) and (iter < max_iter):
        x_old: np.float64 = x

        x = (xl + xu) / 2
        fx: np.float64 = evaluate_one_variable(func, x)

        iter = iter + 1

        if x != np.float64(0.0):
            relative_error = abs((x - x_old) / x) * 100

        if (fl * fx) < 0:
            xu = x
        else:
            xl = x
            fl = fx

    return x, xl, xu, relative_error, iter


def false_position(
    func: Union[str, Callable],
    xl: np.float64,
    xu: np.float64,
    tol: np.float64,
    max_iter: int,
) -> np.float64:
    x: np.float64 = np.float64(0.0)
    iter: int = 0
    relative_error: np.float64 = np.float64(100.0)
    
    fl: np.float64 = evaluate_one_variable(func, xl)
    fu: np.float64 = evaluate_one_variable(func, xu)

    if abs(fl) < abs(fu):
        x = xl
    else:
        x = xu

    while (relative_error > tol) and (iter < max_iter):
        x_old: np.float64 = x
        iter = iter + 1
        
        if is_indertemination(fu, fl):
            return np.inf, xl, xu, relative_error, iter
        
        x = xu + (fu * (xl - xu)) / (fu - fl)

        fX: np.float64 = evaluate_one_variable(func, x)

        if x != np.float64(0.0):
            relative_error = abs((x - x_old) / x) * 100

        if abs(fl) < abs(fu):
            xl = x
            fl = fX
        else:
            xu = x
            fu = fX

    return x, xl, xu, relative_error, iter
