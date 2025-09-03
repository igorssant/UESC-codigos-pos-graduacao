from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def functional(x :NDArray[np.float64],
               A :NDArray[np.float64],
               c :NDArray[np.float64],
               b :np.float64) -> NDArray[np.float64]:
    """Segundo a atividade:
    A -> matriz (m x n)
    b -> constante escalar
    c -> constante escalar
    x -> vetor (n)
    """
    x_minus_c :NDArray[np.float64] = np.subtract(x, c)

    # trasformar em vetor coluna
    x_minus_c = x_minus_c.reshape(-1, 1)

    return np.add(np.dot(x_minus_c.T, np.dot(A, x_minus_c)), b)


def calculate_gradient(x :NDArray[np.float64],
                       A :NDArray[np.float64],
                       c :NDArray[np.float64],
                       matrix_is_assymetrical :bool = False) -> NDArray[np.float64]:
    """Segundo o problema:
    A matriz pode ser simétrica, ou seja,
        grad_J(x) = 2 * A * (x - c)
    Ou pode ser não simétrica, ou seja,
        grad_J(x) = (A + A^T) * (x - c)
    """
    x_minus_c :NDArray[np.float64] = np.subtract(x, c)
    
    if matrix_is_assymetrical:
        return np.dot(np.add(A, A.T), x_minus_c)
    
    return np.multiply(np.float64(2.0), np.dot(A, x_minus_c))


def gradient_descent(x0 :np.float64,
                     A :NDArray[np.float64],
                     c :NDArray[np.float64],
                     b :np.float64,
                     learing_rate :np.float64,
                     max_iter :int = 500,
                     tolerance :np.float64 = np.float64(0.0001),
                     matrix_is_assymetrical :bool = False) -> Tuple[NDArray[np.float64], int]:
    total_iter :int = 0
    x_optimal :NDArray[np.float64] = x0.copy()
    previous_cost :NDArray[np.float64] = functional(x0, A, c, b)
    
    for i in range(max_iter):
        gradient :NDArray[np.float64] = calculate_gradient(x_optimal, A, c, matrix_is_assymetrical)

        x_optimal = np.subtract(x_optimal, np.multiply(learing_rate, gradient))

        current_cost :NDArray[np.float64] = functional(x_optimal, A, c, b)
        
        if np.abs(current_cost - previous_cost) < tolerance:
            total_iter = i
            break
        
        previous_cost = current_cost
        
    return (x_optimal, total_iter)


def test() -> None:
    print(functional(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
        np.float64(3.0)))
    
    print(calculate_gradient(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64)))
    
    print(calculate_gradient(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([[1.0, 3.0],
                  [2.0, 5.0]], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
        True))


def main() -> None:
    x0 :NDArray[np.float64] = np.array([3.0, 1.0], dtype=np.float64)
    A :NDArray[np.float64] = np.array([[2.0, 1.0],
                                       [1.0, 2.0]], dtype=np.float64)
    c :NDArray[np.float64] = np.array([1.0, -1.0], dtype=np.float64)
    b :np.float64 = np.float64(1.0)
    alpha :np.float64 = np.float64(0.01)
    max_iter :int = 1000
    tolerance :np.float64 = np.float64(0.00001)
    result :NDArray[np.float64]
    actual_max_iter :int
    info_text :str
    
    result, actual_max_iter = gradient_descent(x0,
                                               A,
                                               c,
                                               b,
                                               alpha,
                                               max_iter,
                                               tolerance,
                                               False)
    
    if actual_max_iter < max_iter:
        info_text = "O modelo convergiu com " + str(actual_max_iter) + " iterações."
    else:
        info_text = "O modelo não convergiu."

    print(f"O resultado é: {result}\n{info_text}")

if __name__ == "__main__":
    # test()
    main()
