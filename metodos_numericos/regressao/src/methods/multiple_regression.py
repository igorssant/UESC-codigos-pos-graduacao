from numpy.typing import NDArray
import numpy as np


def determinate_elements(X: NDArray, b: NDArray) -> NDArray:
    """Determina os elementos da matriz aumentada para regressão polinomial múltipla.
    Args:
        X (NDArray): Array bidimensional de pontos x.
        b (NDArray): Array unidimensional de pontos y.
    Returns:
        NDArray: Matriz aumentada para regressão polinomial múltipla.
    Raises:
        ArithmeticError: Se o número de pontos for menor que a ordem do polinômio + 1.
    """

    data_size: int = len(b)
    num_variables: int = X.shape[1] if X.ndim > 1 else 1
    poly_order: int = num_variables

    if data_size < poly_order + 1:
        raise ArithmeticError(f"""
                               Erro: O número de pontos deve ser maior ou igual a ordem do polinômio + 1.
                               Quantidade de dados = {data_size}.
                               Ordem + 1 = {poly_order + 1}.
                               Regressão impossível.
                               """)

    aux_matrix: NDArray = np.hstack((np.ones((data_size, 1)), X))
    augmented_matrix: NDArray = np.zeros(
        (poly_order + 1, poly_order + 2), dtype=np.float64
    )

    for i in range(poly_order + 1):
        for j in range(i + 1):
            summ: np.float64 = np.float64(0.0)

            for l in range(data_size):
                summ += aux_matrix[l, i] * aux_matrix[l, j]

            augmented_matrix[i, j] = summ

            if i != j:
                augmented_matrix[j, i] = summ

        summ = np.float64(0.0)

        for l in range(data_size):
            summ += b[l] * aux_matrix[l, i]

        augmented_matrix[i, poly_order + 1] = summ

    return augmented_matrix


def retrieve_multiple_coef(A: NDArray, b: NDArray) -> NDArray:
    return np.linalg.solve(A, b)
