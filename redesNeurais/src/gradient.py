from typing import Callable, Tuple
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


def rastrigin_functional(x: NDArray[np.float64]) -> np.float64:
    """Calcula o valor da função de Rastrigin.
    x -> uma matriz (m x n).
    """

    if x.shape[0] != 2:
        raise ValueError("O vetor de entrada deve ser uma matriz (array de dimensão 2).")

    x_value :np.float64 = x[0]
    y_value :np.float64 = x[1]

    return (np.float64(20.0) +
            (x_value**2 - np.float64(10.0) * np.cos(2 * np.pi * x_value)) +
            (y_value**2 - np.float64(10.0) * np.cos(2 * np.pi * y_value)))


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


def rastrigin_gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calcula do gradiente da função de Rastrigin.
    x -> uma matriz (m x n).
    """

    if x.shape[0] != 2:
        raise ValueError("O vetor de entrada deve ser uma matriz (array de dimensão 2).")

    x_val :np.float64 = x[0]
    y_val :np.float64 = x[1]
    df_dx :np.float64 = np.float64(2) * x_val + np.float64(20) * np.pi * np.sin(2 * np.pi * x_val)
    df_dy :np.float64 = np.float64(2) * y_val + np.float64(20) * np.pi * np.sin(2 * np.pi * y_val)

    return np.array([df_dx, df_dy], dtype=np.float64)


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


def gradient_descent_multimodal(x0 :NDArray[np.float64],
                                learing_rate :np.float64,
                                functional_fn :Callable, # funcional do problema
                                gradient_fn :Callable,   # função que calcula do gradiente
                                max_iter :int = 500,
                                tolerance :np.float64 = np.float64(0.0001)
                                ) -> Tuple[NDArray[np.float64], int]:
    total_iter :int = 0
    x_optimal :NDArray[np.float64] = x0.copy()
    previous_cost :np.float64 = functional_fn(x0)

    for i in range(max_iter):
        gradient :NDArray[np.float64] = gradient_fn(x_optimal)

        x_optimal = np.subtract(x_optimal, np.multiply(learing_rate, gradient))

        current_cost :np.float64 = functional_fn(x_optimal)

        if np.abs(current_cost - previous_cost) < tolerance:
            total_iter = i
            break

        previous_cost = current_cost
        total_iter = i + 1

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


def exec_multimodal_gradiente_descent(functional_func :Callable,
                                      descent_gradiente_func :Callable) -> None:
    x :NDArray[np.float64] = np.array([[0.5, -0.5],   # próximo do mínimo global
                                       [5.0, 5.0],    # próximo de um mínimo local
                                       [-5.0, -5.0]], # longe de todos os mínimos
                                      dtype=np.float64)
    alphas :NDArray[np.float64] = np.array([0.001, 0.01, 0.1],
                                           dtype=np.float64)
    max_iter :int = 1000
    tolerance :np.float64 = np.float64(0.0001)
    
    for alpha in alphas:
        result :NDArray[np.float64]
        total_iter :int

        for i in range(x.shape[0]):
            result, total_iter = gradient_descent_multimodal(x[i,:],
                                                             alpha,
                                                             functional_func,
                                                             descent_gradiente_func,
                                                             max_iter,
                                                             tolerance)
            informational_text :str = f"Convergiu em {total_iter} iterações." \
                if total_iter < max_iter \
                else f"Não convergiu em {max_iter} interações."

            print(f"""Teste alpha (learning rate) = {alpha}
Ponto inicial (x0) = {x[i, :]}
{informational_text}
Resultado = {result}
Valor mínimo = {functional_func(result)}
\n\n""")


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

    exec_multimodal_gradiente_descent(rastrigin_functional, rastrigin_gradient)


if __name__ == "__main__":
    # test()
    main()
    
