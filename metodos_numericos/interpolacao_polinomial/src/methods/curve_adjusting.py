import sys
import os
from typing import Tuple
from numpy.typing import NDArray
import numpy as np

# Adicionar o diretório 'src' (o diretório raiz do projeto de módulos) ao Python path
# Agora ele vai um nível acima do diretório 'methods' para chegar em 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.gauss_seidel import (
    gauss_seidel, get_augmented_matrix
)

def natural_splines(X: NDArray[np.float64],
                    Y: NDArray[np.float64]) -> Tuple[NDArray[np.float64],
                                                     NDArray[np.float64],
                                                     NDArray[np.float64],
                                                     NDArray[np.float64]]:
    # Número de splines
    n = X.shape[0] - 1

    # Inicializar os coeficientes dos splines
    a, b, c, d = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    # a é o valor de Y nos pontos X
    a = Y[:-1]

    # h é a diferença entre os pontos X
    # h[i] = X[i+1] - X[i]
    h = np.diff(X)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    # Construir o sistema de equações para c nos pontos internos
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

        B[i] = (3 / h[i]) * (Y[i + 1] - Y[i]) - (3 / h[i - 1]) * (Y[i] - Y[i - 1])

    augmented_matrix = get_augmented_matrix(A, B)

    # Resolver o sistema para encontrar os coeficientes c
    c = gauss_seidel(augmented_matrix, 1e-6, 200)

    # Calcular os coeficientes b
    for i in range(n):
        b[i] = (1 / h[i]) * (Y[i + 1] - Y[i]) - (h[i] / 3) * (2 * c[i] + c[i + 1])

    # Calcular os coeficientes d
    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Remover o último elemento, pois não é necessário
    c = c[:-1]

    return (a, b, c, d)

def fixed_splines(X :NDArray[np.float64],
                  Y :NDArray[np.float64],
                  dx_0 :np.float64,
                  dx_n :np.float64
                  ) -> Tuple[NDArray[np.float64],
                             NDArray[np.float64],
                             NDArray[np.float64],
                             NDArray[np.float64]]:
    # Número de splines
    n = X.shape[0] - 1

    # Inicializar os coeficientes dos splines
    a, b, c, d = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    # a é o valor de Y nos pontos X
    a = Y[:-1]

    # h é a diferença entre os pontos X
    # h[i] = X[i+1] - X[i]
    h = np.diff(X)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1
    # A derivada do primeiro ponto *dx_0*
    B[0] = dx_0

    A[n, n] = 1
    # A derivada do ultimo ponto *dx_n*
    B[n] = dx_n

    # Construir o sistema de equações para c nos pontos internos
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

        B[i] = (3 / h[i]) * (Y[i + 1] - Y[i]) - (3 / h[i - 1]) * (Y[i] - Y[i - 1])

    augmented_matrix = get_augmented_matrix(A, B)

    # Resolver o sistema para encontrar os coeficientes c
    c = gauss_seidel(augmented_matrix, 1e-6, 200)

    # Calcular os coeficientes b
    for i in range(n):
        b[i] = (1 / h[i]) * (Y[i + 1] - Y[i]) - (h[i] / 3) * (2 * c[i] + c[i + 1])

    # Calcular os coeficientes d
    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Remover o último elemento, pois não é necessário
    c = c[:-1]

    return (a, b, c, d)

def cubic_splines(X :NDArray[np.float64],
                  Y :NDArray[np.float64],
                  *,
                  dx_0 :np.float64 | None = None,
                  dx_n :np.float64 | None = None
                  ) -> Tuple[NDArray[np.float64],
                             NDArray[np.float64],
                             NDArray[np.float64],
                             NDArray[np.float64]]:
    if dx_0 == None and dx_n == None:
        return natural_splines(X, Y)
    elif dx_0 != None and dx_n != None:
        return fixed_splines(X, Y, dx_0=dx_0, dx_n=dx_n)
    else:
        raise RuntimeError("""
                           Para utilizar os SPLines fixados
                           as derivadas *dx_0* e *dx_n* devem ser
                           definidas.
                           """)

if __name__  == "__main__":
    # Exemplo de uso
    X = np.array([0, 1, 2, 3], dtype=np.float64)
    Y = np.array([1, np.e, np.e**2, np.e**3], dtype=np.float64)

    a, b, c, d = natural_splines(X, Y)

    print("Coeficientes dos splines:")
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
