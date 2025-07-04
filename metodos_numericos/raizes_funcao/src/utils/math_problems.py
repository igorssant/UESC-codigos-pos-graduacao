import numpy as np

def is_indertemination(a:np.float64, b:np.float64) -> bool:
    """Esta função verifica se a subtração
    de dois valores do tipo numpy.float64
    gera o valor 0.0.
    Retorna *true* se a - b = 0.0
    Retorna *false* caso contrário 

    Args:
        a (np.float64): um dos divisores
        b (np.float64): um dos divisores

    Returns:
        bool: o resultado da subtração é 0.0
    """
    
    return (a - b) == np.float64(0.0)
