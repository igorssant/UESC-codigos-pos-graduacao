import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable
from utils.math_problems import is_indertemination

def fixed_point(
    func: Union[str, Callable], x0: np.float64, tol: np.float64, max_iter: int
) -> tuple[np.float64, np.float64, np.float64, int]:
    """
    Método do Ponto Fixo para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        x0 (np.float64): O ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        - A raiz encontrada (np.float64)
        - O último valor de x (np.float64)
        - O erro relativo (np.float64)
        - O número de iterações (int)
    """

    iter: int = 0
    relative_error: np.float64 = np.float64(100.0)
    x: np.float64 = x0
    x_old: np.float64 = x

    while (relative_error > tol) and (iter < max_iter):
        x_old: np.float64 = x
        iter = iter + 1

        x = evaluate_one_variable(func, x_old)

        if x != np.float64(0.0):
            relative_error = abs((x - x_old) / x) * 100

    return x, x_old, relative_error, iter


def newton_raphson(
    func: Union[str, Callable],
    derivative: Union[str, Callable],
    x0: np.float64,
    tol: np.float64,
    max_iter: int,
) -> tuple[np.float64, np.float64, np.float64, int]:
    """
    Método de Newton-Raphson para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        derivative (Union[str, Callable]): A derivada da função.
        x0 (np.float64): O ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        - A raiz encontrada (np.float64)
        - O último valor de x (np.float64)
        - O erro relativo (np.float64)
        - O número de iterações (int)
    """

    iter: int = 0
    relative_error: np.float64 = np.float64(100.0)
    x: np.float64 = x0

    while (relative_error > tol) and (iter < max_iter):
        x0 = x
        iter = iter + 1

        x = x0 - (
            evaluate_one_variable(func, x0) / evaluate_one_variable(derivative, x0)
        )

        if x != np.float64(0.0):
            relative_error = abs((x - x0) / x) * 100

    return x, x0, relative_error, iter


def secant(
    func: Union[str, Callable],
    x0: np.float64,
    x1: np.float64,
    tol: np.float64,
    max_iter: int,
) -> tuple[np.float64, np.float64, np.float64, int]:
    """
    Método da Secante para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        x0 (np.float64): O ponto inicial.
        x1 (np.float64): O segundo ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        - A raiz encontrada (np.float64)
        - O último valor de x0 (np.float64)
        - O último valor de x1 (np.float64)
        - O erro relativo (np.float64)
        - O número de iterações (int)
    """

    relative_error: np.float64 = np.float64(100.0)
    iter: int = 0
    x: np.float64 = np.float64(0.0)

    while (relative_error > tol) and (iter < max_iter):
        iter = iter + 1

        f0: np.float64 = evaluate_one_variable(func, x0)
        f1: np.float64 = evaluate_one_variable(func, x1)
        
        if is_indertemination(f0, f1):
            return np.inf, x0, x1, relative_error, iter

        x: np.float64 = x1 - f1 * ((x0 - x1) / (f0 - f1))

        if x != np.float64(0.0):
            relative_error = abs((x - x1) / x) * 100

        x0 = x1
        x1 = x

    return x, x0, x1, relative_error, iter
