import numpy as np
import sympy as sy
from typing import Union, Callable

def __parse(function :str) -> sy.Expr:
    """
    Converte uma string de função em um objeto sympy.
    A função deve ser uma string que representa uma expressão matemática.
    Exemplo: "np.sin(x) + np.exp(x)".
    """
    function = function.replace("np.", "")

    local_dict = {
        # SymPy functions
        "exp": sy.exp,
        "sin": sy.sin,
        "cos": sy.cos,
        "tan": sy.tan,
        "log": sy.log
    }
    parsed_function:sy.Expr = sy.parse_expr(function, local_dict=local_dict)
    return parsed_function

def evaluate_one_variable(function :Union[str, Callable[[np.float64], np.float64]],
                          x0 :np.float64) -> np.float64:
    """
    Avalia uma função de uma variável em um ponto específico.
    A função pode ser passada como uma string ou um callable.
    Se for uma string, deve conter a variável 'x'.
    """
    result:np.float64 = np.float64(0.0)

    if isinstance(function, str):
        if "x" not in function:
            raise ValueError("A função deve conter a variável 'x'.")

        symbol_x :sy.Symbol = sy.Symbol("x")
        parsed_function :sy.Expr = __parse(function)
        result = parsed_function.subs(symbol_x, x0).evalf(17)

    elif isinstance(function, Callable):
        result = function(x0)
    else:
        raise ValueError("A função deve ser uma string ou um callable.")

    return np.float64(result)
