import numpy as np
import pandas as pd


def mean_squared_error(y_true :np.ndarray, y_pred :np.ndarray) -> np.float64:
    return np.mean((y_true - y_pred)**2)


def r_squared(y_true :np.ndarray, y_pred :np.ndarray) -> np.float64:
    """
    Calcula o Coeficiente de Determinação (R^2)
    """

    ss_res :np.float64 = np.sum((y_true - y_pred)**2)
    ss_tot :np.float64 = np.sum((y_true - np.mean(y_true))**2)
    
    if ss_tot == 0:
        return 0
    
    return np.float64(1.0) - (ss_res / ss_tot)

