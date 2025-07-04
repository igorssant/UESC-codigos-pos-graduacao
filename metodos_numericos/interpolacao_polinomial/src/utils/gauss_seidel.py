from numpy.typing import NDArray # type: ignore
import numpy as np # type: ignore


def __calculate_error(xi: NDArray, x0: NDArray) -> np.float64:
    """
    Calcula o erro relativo entre duas soluções.
    :param xi: Solução atual
    :param x0: Solução anterior
    :return: Erro relativo máximo
    """
    #error_array: NDArray = np.abs((xi - x0) / xi)
    error_array: NDArray = np.abs((xi - x0))

    return np.max(error_array)


def get_augmented_matrix(
    A: NDArray, B: NDArray
) -> NDArray:
    """
    Generate an augmented matrix from matrix A and vector B.

    Parameters:
    A (NDArray): The coefficient matrix.
    B (NDArray): The constant terms vector.

    Returns:
    NDArray: The augmented matrix.
    """
    return np.hstack((A, B.reshape(-1, 1)))


def calculate_initial_solution(augmented_matrix: NDArray) -> NDArray:
    """
    Calcula a solução inicial para o método iterativo.
    :param augmented_matrix: Matriz aumentada do sistema
    :return: Solução inicial
    """
    n: int = augmented_matrix.shape[0]
    initial_guess: NDArray = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if augmented_matrix[i, i] == 0:
            raise ValueError("Elemento nulo encontrado na diagonal.")

        initial_guess[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]

    return initial_guess


def gauss_seidel(augmented_matrix: NDArray, tol: np.float64 | float, max_iter: int) -> NDArray[np.float64]:
    """
    Método de Gauss-Seidel para resolver sistemas lineares.
    """

    def calculate_solution_gs(matrix, prev_solution):
        # Implementação específica para Gauss-Seidel
        # (adaptação da sua implementação atual)
        n = matrix.shape[0]
        solution = np.copy(prev_solution)

        for i in range(n):
            sum_value = 0.0
            for j in range(n):
                if j != i:
                    sum_value += matrix[i, j] * solution[j]

            solution[i] = (matrix[i, -1] - sum_value) / matrix[i, i]

        return solution

    return iterative_method(
        augmented_matrix,
        tol,
        max_iter,
        calculate_solution_gs,
        "O método de Gauss-Seidel",
    )

def iterative_method(
    augmented_matrix: NDArray,
    tol: np.float64 | float,
    max_iter: int,
    calculate_solution_func,
    method_name: str = "O método iterativo",
) -> NDArray[np.float64]:
    """
    Método iterativo genérico para resolver sistemas lineares.

    :param augmented_matrix: Matriz aumentada do sistema
    :param tol: Tolerância para convergência
    :param max_iter: Número máximo de iterações
    :param calculate_solution_func: Função que calcula a próxima solução
    :param method_name: Nome do método para mensagens de erro
    :return: Solução do sistema
    """
    n: int = augmented_matrix.shape[0]
    iter_count: int = 0
    previous_solution: NDArray = calculate_initial_solution(augmented_matrix)
    current_solution: NDArray = np.zeros(n, dtype=np.float64)

    while iter_count < max_iter:
        current_solution = calculate_solution_func(augmented_matrix, previous_solution)

        if np.any(np.isnan(current_solution)):
            raise ValueError("Solução inválida encontrada.")

        error = __calculate_error(current_solution, previous_solution)

        if error < tol:
            return current_solution

        iter_count += 1
        previous_solution = np.copy(current_solution)

    raise RuntimeError(f"{method_name} não convergiu após {max_iter} iterações.")
