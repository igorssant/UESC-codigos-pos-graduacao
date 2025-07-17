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


def progressive_derivative_1(func :Union[str, Callable[[np.float64], np.float64]],
                             x :np.float64,
                             h :np.float64) -> np.float64:
    if h <= np.float64(0.0):
        raise ValueError("O tamanho do passo *h* não pode ser zero.")

    x_plus_h :np.float64 = x + h
    x_plus_2h :np.float64 = np.float64(x + 2.0 * h)

    return (-evaluate_one_variable(func, x_plus_2h) +
            4 * evaluate_one_variable(func, x_plus_h) -
            3 * evaluate_one_variable(func, x)) / (2 * h)

def progressive_derivative_2(func :Union[str, Callable[[np.float64], np.float64]],
                             x :np.float64,
                             h :np.float64) -> np.float64:
    if h <= np.float64(0.0):
        raise ValueError("O tamanho do passo *h* não pode ser zero.")

    x_plus_h :np.float64 = x + h
    x_plus_2h :np.float64 = np.float64(x + 2.0 * h)
    x_plus_3h :np.float64 = np.float64(x + 3.0 * h)

    return (-evaluate_one_variable(func, x_plus_3h) +
            4 * evaluate_one_variable(func, x_plus_2h) -
            5 * evaluate_one_variable(func, x_plus_h) +
            2 * evaluate_one_variable(func, x)) / (h**2)

def regressive_derivative_1(func :Union[str, Callable[[np.float64], np.float64]],
                            x :np.float64,
                            h :np.float64) -> np.float64:
    if h >= np.float64(0.0):
        raise ValueError("O tamanho do passo *h* não pode ser zero.")

    x_menos_h :np.float64 = x - h
    x_menos_2h :np.float64 = np.float64(x - 2.0 * h)

    return (3 * evaluate_one_variable(func, x) -
            4 * evaluate_one_variable(func, x_menos_h) +
            evaluate_one_variable(func, x_menos_2h)) / (2 * h)

def regressive_derivative_2(func :Union[str, Callable[[np.float64], np.float64]],
                            x :np.float64,
                            h :np.float64) -> np.float64:
    if h >= np.float64(0.0):
        raise ValueError("O tamanho do passo *h* não pode ser zero.")
    
    x_menos_h :np.float64 = x - h
    x_menos_2h :np.float64 = np.float64(x - 2.0 * h)
    x_menos_3h :np.float64 = np.float64(x - 3.0 * h)
    
    return (2 * evaluate_one_variable(func, x) -
            5 * evaluate_one_variable(func, x_menos_h) +
            4 * evaluate_one_variable(func, x_menos_2h) -
            evaluate_one_variable(func, x_menos_3h)) / (h**2)

def central_derivative_1(func :Union[str, Callable[[np.float64], np.float64]],
                         x :np.float64,
                         h :np.float64) -> np.float64:
    if h == np.float64(0.0):
        raise ValueError("O tamanho do passo *h* não pode ser zero.")
    
    x_mais_h :np.float64 = x + h
    x_mais_2h :np.float64 = np.float64(x + 2.0 * h)
    x_menos_h :np.float64 = x - h
    x_menos_2h :np.float64 = np.float64(x - 2.0 * h)

    return (-evaluate_one_variable(func, x_mais_2h) +
            8 * evaluate_one_variable(func, x_mais_h) -
            8 * evaluate_one_variable(func, x_menos_h) +
            evaluate_one_variable(func, x_menos_2h)) / (12 * h)

def central_derivative_2(func :Union[str, Callable[[np.float64], np.float64]],
                         x :np.float64,
                         h :np.float64) -> np.float64:
    if h == np.float64(0.0):
        raise ValueError("O tamanho do passo *h* não pode ser zero.")
        
    x_mais_h :np.float64 = x + h
    x_mais_2h :np.float64 = np.float64(x + 2.0 * h)
    x_menos_h :np.float64 = x - h
    x_menos_2h :np.float64 = np.float64(x - 2.0 * h)

    return (-evaluate_one_variable(func, x_mais_2h) +
            16 * evaluate_one_variable(func, x_mais_h) -
            30 * evaluate_one_variable(func, x) +
            16 * evaluate_one_variable(func, x_menos_h) -
            evaluate_one_variable(func, x_menos_2h)) / (12 * h**2)

def forward_difference_euler(func :Union[str, Callable[[np.float64], np.float64]],
                             x0 :np.float64,
                             h :np.float64) -> np.float64:
    if h <= np.float64(0.0):
        raise ValueError("O tamanho do passo 'h' deve ser um valor positivo.")

    fx_0 :np.float64 = evaluate_one_variable(func, x0)
    fx_plus_h :np.float64 = evaluate_one_variable(func, x0 + h)

    return (fx_plus_h - fx_0) / h

def backward_difference_euler(func :Union[str, Callable[[np.float64], np.float64]],
                              x0 :np.float64,
                              h :np.float64) -> np.float64:
    if h >= np.float64(0.0):
        raise ValueError("O tamanho do passo 'h' deve ser um valor menor que zero.")

    fx_0 :np.float64 = evaluate_one_variable(func, x0)
    fx_minus_h :np.float64 = evaluate_one_variable(func, x0 - h)

    return (fx_0 - fx_minus_h) / h

def central_derivative_three_points(func :Union[str, Callable[[np.float64], np.float64]],
                                    x0 :np.float64,
                                    h :np.float64) -> np.float64:
    if h == np.float64(0.0):
        raise ValueError("O tamanho do passo *h* não pode ser zero.")

    return (evaluate_one_variable(func, x0 + h) - evaluate_one_variable(func, x0 - h)) / (2.0 * h)

def second_central_derivative_three_points(func :Union[str, Callable[[np.float64], np.float64]],
                                           x0 :np.float64,
                                           h :np.float64) -> np.float64:
    if h == np.float64(0.0):
        raise ValueError("O tamanho do passo 'h' não pode ser zero.")

    fx_0 = evaluate_one_variable(func, x0)
    fx_plus_h = evaluate_one_variable(func, x0 + h)
    fx_minus_h = evaluate_one_variable(func, x0 - h)

    return (fx_plus_h - 2.0 * fx_0 + fx_minus_h) / (h**2)
