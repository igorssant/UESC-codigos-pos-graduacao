from typing import Tuple
import numpy as np

def linear_regression(x:list[np.float64], y:list[np.float64]) -> Tuple[np.float64, np.float64,
                                                                       np.float64, np.float64]:
    data_size :int = len(x)
    total_sum_squares: np.float64 = np.float64(0.0)
    sum_square_residuals: np.float64 = np.float64(0.0)
    # a0 -> coeficiente linear
    a0: np.float64 = np.float64(0.0)
    # a1 -> coeficiente angular
    a1: np.float64 = np.float64(0.0)
    # syx -> Erro padrão da estimativa
    syx: np.float64 = np.float64(0.0)
    # r2 -> coeficiente de determinacao
    r2: np.float64 = np.float64(0.0)
    sum_x: np.float64 = np.float64(0.0)
    sum_y: np.float64 = np.float64(0.0)
    sum_xy: np.float64 = np.float64(0.0) # <- acumula a soma dos produtos x[i​] * y[i]
    sum_x2: np.float64 = np.float64(0.0) # <- acumula a soma dos quadrados dos valores de x[i]
    
    for i in range(data_size):
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += x[i] * y[i]
        sum_x2 += x[i]**2
    
    mean_x :np.float64 = sum_x / np.float64(data_size)
    mean_y :np.float64 = sum_y / np.float64(data_size)
    a1 = np.sum(np.float64(data_size) * sum_xy - sum_x * sum_y) / np.sum(np.float64(data_size) * sum_x2 - np.dot(sum_x, sum_x))
    a0 = mean_y - a1 * mean_x
    
    for i in range(data_size):
        total_sum_squares += (y[i] - mean_x)**2
        sum_square_residuals += (y[i] - a1 * x[i] - a0)**2
    
    syx = np.sqrt(sum_square_residuals / (data_size - 2))
    r2  = (total_sum_squares - sum_square_residuals) / total_sum_squares
    
    return (a0, a1, syx, r2)
