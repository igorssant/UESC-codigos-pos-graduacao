import numpy as np
import sympy as sy

def _parse(function:str) -> sy.Expr:
    parsed_function:sy.Expr = sy.parse_expr(function)
    return parsed_function

def evaluate_one_variable(function:str, x0:np.float64) -> np.float64:
    symbol_x:sy.Symbol = sy.Symbol("x")
    parsed_function:sy.Expr = _parse(function)
    return parsed_function.subs(symbol_x, x0).evalf()
