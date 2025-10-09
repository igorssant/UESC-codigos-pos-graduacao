import numpy as np


def tanh(z :np.float64) -> np.float64:
    return np.tanh(z)


def tanh_derivative(a :np.float64) -> np.float64:
    return np.float64(1.0) - a**2


def linear(z :np.float64) -> np.float64:
    return z


def linear_derivative(a :np.float64) -> np.float64:
    return np.float64(1.0)

