from typing import Callable, Union
from numpy.typing import NDArray
import numpy as np
import sys
import os
import re
# Adicionar o diretório 'src' (o diretório raiz do projeto de módulos) ao Python path
# Agora ele vai um nível acima do diretório 'methods' para chegar em 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.parser import evaluate_one_variable
from utils.quadrature_tables import *


def trapezoid_integral(func :Union[str, Callable[[np.float64], np.float64]],
                       a :np.float64,
                       b:np.float64) -> np.float64:
    return (b - a) *\
           (evaluate_one_variable(func, a) +\
           evaluate_one_variable(func, b)) / 2

def multiple_trapezoid_integral(func :Union[str, Callable[[np.float64], np.float64]],
                                a :np.float64,
                                b :np.float64,
                                n :int) -> np.float64:
    if n < 1:
        raise ValueError("O número de pontos (n) deve ser pelo menos 1.")

    h :int = int((b - a) / n)
    x :np.float64 = a
    summ :np.float64 = evaluate_one_variable(func, x)

    for i in range(n - 1):
        x += h
        summ = summ + 2 * evaluate_one_variable(func, x)

    summ += evaluate_one_variable(func, b)

    return (b - a) * summ / (2 * n)

def simpson13_integral(func :Union[str, Callable[[np.float64], np.float64]],
                       a :np.float64,
                       b :np.float64) -> np.float64:
    mid_point :np.float64 = np.float64((b + a) / 2.0)

    return (b - a) *\
        (evaluate_one_variable(func, a) +\
        4 * evaluate_one_variable(func, mid_point) +\
        evaluate_one_variable(func, b)) / 6

def multiple_simpson13_integral(func :Union[str, Callable[[np.float64], np.float64]],
                                a :np.float64,
                                b :np.float64,
                                n :int) -> np.float64:
    if n < 1:
        raise ValueError("O número de pontos (n) deve ser pelo menos 1.")

    # Calcula h e pega os pontos
    h :np.float64 = np.float64((b - a) / n)
    points :NDArray[np.float64] = np.arange(a, b + h, h)

    if n % 2 != 0:
        print("Número de intervalos inválido")
        return np.NaN

    odd_sum :np.float64 = np.array(
        [evaluate_one_variable(func, point) for point in points[1:-1:2]],
        dtype=np.float64
        ).sum()

    even_sum :np.float64 = np.array(
        [evaluate_one_variable(func, point) for point in points[2:-2:2]],
        dtype=np.float64
        ).sum()

    return (b - a) *\
        (evaluate_one_variable(func, a) +\
        4 * odd_sum + 2 * even_sum +\
        evaluate_one_variable(func, b)) / (3 * n)

def simpson38_integral(func :Union[str, Callable[[np.float64], np.float64]],
                       a :np.float64,
                       b :np.float64) -> np.float64:
    h :np.float64 = np.float64((b - a) / 3.0)
    mid_points :NDArray[np.float64] = [(a + h), (b - h)]

    return (b - a) *\
           (evaluate_one_variable(func, a) + 3 *\
           evaluate_one_variable(func, mid_points[0]) +\
           3 * evaluate_one_variable(func, mid_points[1]) +\
           evaluate_one_variable(func, b)) / 8

def richards_extrapolation(func :Union[str, Callable[[np.float64], np.float64]],
                           a :np.float64,
                           b :np.float64,
                           h1 :np.float64,
                           h2 :np.float64) -> np.float64:
    i_h1 :np.float64 = multiple_trapezoid_integral(func, a, b, h1)
    i_h2 :np.float64 = multiple_trapezoid_integral(func, a, b, h2)

    return i_h2 + (1.0 / ((h1 / h2)**2 - 1.0)) * (i_h2 - i_h1)

def gauss_quadrature(func :str,
                     a :np.float64,
                     b :np.float64,
                     n :int) -> np.float64:
    if not isinstance(func, str):
        raise ValueError("A função (func) deve ser uma string.")

    if n < 1:
        raise ValueError("O número de pontos (n) deve ser pelo menos 1.")

    tables :list[Callable] = [legendre_table, tchebyshev_table, laguerre_table, hermite_table]
    table :Callable = tables[0]

    if re.match(r"\s*\(?\s*1\s*/\s*(np\.sqrt|sqrt)\(\s*1\s*-\s*x\s*\*\*\s*2\s*\)\s*\)?\s*\*\s*", func):
        print("Tchebyshev")
        table = tables[1]
    elif re.match(r"\s*\(?\s*((np|math)\.exp\(\s*-x\s*\*\*\s*2\s*\)|math\.e\s*\*\*\s*-x\s*\*\*\s*2)\s*\)?\s*\*\s*", func):
        print("Hermite")
        table = tables[3]
    elif re.match(r"\s*\(?\s*((np|math)\.exp\(\s*-x\s*\)|math\.e\s*\*\*\s*-x\s*)\s*\)?\s*\*\s*", func):
        print("Laguerre")
        table = tables[2]

    # Verifica se precisa fazer mudança de variável
    if a != np.float64(-1.0) or b != np.float64(1.0):
        x :str = f"({(b + a) / 2 } + {((b - a) / 2)} * x)"
        dx :np.float64 = np.float64((b - a) / 2.0)

        func = func.replace("x", x)
        func = f"({func}) * {dx}"

    args :dict = table(n)

    if args == None:
        return np.NaN

    x :np.array = np.array(args["x"])
    c :np.array = np.array(args["c"])
    final_result :NDArray[np.flaot64] = np.zeros((x.shape[0], 1),
                                                 dtype=np.float64)

    for i in range(x.shape[0]):
        final_result[i] = c[i] * evaluate_one_variable(func, x[i])

    return final_result.sum()
