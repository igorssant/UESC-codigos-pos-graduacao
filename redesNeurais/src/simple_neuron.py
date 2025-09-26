from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Tuple, Union
import pandas as pd
import numpy as np


def convert_rows_to_float(df :pd.DataFrame) -> pd.DataFrame:
    """função convert_rows_to_float
    Recebe o dataframe e converte cada coluna para np.float64
    :param: df (pd.DataFrame).
    :return: operated_df (pd.DataFrame).
    """

    for column in df.columns:
        df[column] = df[column].astype(np.float64)
    
    return df


def cost(y_true :NDArray[np.float64],
         y_predicted :NDArray[np.float64],
         method :str = "mse") -> np.float64:
    """função cost
    Realiza o calculo do custo.
    :param: y_true      (np.float64).
    :param: y_predicted (np.float64).
    :param: method      (String).
    """

    match method:
        case "mse":
            return np.mean((y_true - y_predicted)**2)
        case "rmse":
            return np.srqt(np.mean((y_true - y_predicted)**2))
        case "mae":
            return np.mean(np.abs(y_true - y_predicted))
        case "sse":
            return np.float64(0.5) * np.sum(np.abs(y_true - y_predicted)**2)
        case _:
            raise ValueError("Método de custo incorreto.\n" +
                             "Use 'mse', 'rmse', 'sse' ou 'mae'.")


def neuron_model(x :NDArray[np.float64],
                 weights :NDArray[np.float64]) -> NDArray[np.float64]:
    """função neuron_model
    Modelo do neurônio: y = tanh(a0 + a1 * x1 + a2 * x2)
    :param: x (NDArray[np.float64]);
    :param: weights (NDArray[np.float64]).
    :return: y_predicted (NDArray[np.float64]).
    """

    a0 :np.float64
    a1 :np.float64
    a2 :np.float64

    a0, a1, a2 = weights

    x1 :NDArray[np.float64] = x[:, 0]
    x2 :NDArray[np.float64] = x[:, 1]
    v :NDArray[np.float64] = a0 + a1 * x1 + a2 * x2

    return np.tanh(v)


def calculate_residuals(coeficients :NDArray[np.float64],
                        x :NDArray[np.float64],
                        y_true :NDArray[np.float64]) -> Tuple[NDArray[np.float64], np.float64]:
    """função calculate_residuals
    Calcula o vetor de resíduos (erro) e o custo (soma dos quadrados dos resíduos).
    :param: coeficients (NDArray[np.float64]);
    :param: x (NDArray[np.float64]);
    :param: y_true (NDArray[np.float64]).
    :return: erro residual & custo (Tuple).
        erro residual (NDArray[np.float64]);
        custo (np.float64).
    """

    y_predicted :NDArray[np.float64] = neuron_model(x, coeficients)
    residuals :NDArray[np.float64] = y_true - y_predicted
    cost :np.float64 = np.float64(0.5) * np.sum(residuals**2)

    return (residuals, cost)


def calculate_jacobian(x :NDArray[np.float64],
                       weights :NDArray[np.float64]) -> NDArray[np.float64]:
    """ função calculate_jacobian
    Calcula a Matriz Jacobiana J (N x 3).
    Onde N é o número de pontos de dados e 3 é o número de parâmetros (a0, a1, a2).
    :param: x (NDArray[np.float64]);
    :param: weights (NDArray[np.float64]).
    :return: Jacobian matrix (NDArray[np.float64]).
    """

    # derivada de tanh(v)
    sech_squared :NDArray[np.float64] = np.float64(1.0) - neuron_model(x, weights)**2

    # inicializando a matriz Jacobiana
    J :NDArray[np.float64] = np.zeros((x.shape[0], len(weights)))
    x1 :NDArray[np.float64] = x[:, 0]
    x2 :NDArray[np.float64] = x[:, 1]

    # coluna 0 -> derivada em relacao a a0
    J[:, 0] = -sech_squared * 1
    # coluna 1 -> derivada em relacao a a1
    J[:, 1] = -sech_squared * x1
    # coluna 2 -> derivada em relacao a a2
    J[:, 2] = -sech_squared * x2

    return J


def levenberg_marquardt(
    df: pd.DataFrame,
    initial_points: Union[NDArray[np.float64], list[np.float64], list[float]],
    damping_factor: np.float64,
    tol: np.float64 = np.float64(1e-4),
    max_iter: int = 1000,
    normalization_method: int = 0
) -> Tuple[NDArray[np.float64], int]:
    """ função levenberg_marquardt
    Executa o algoritmo de Levenberg-Marquardt para os valores de a, b, c.
    :param: df (pd.DataFrame).
    :param: initial_points (NDArray[np.float64]).
    :param: damping_factor (np.float64).
    :param: tol (np.float64).
    :param: max_iter (int).
    :param: normalization_method (int).
    :return: Tuple[NDArray[np.float64], int].
    """

    x :NDArray[np.float64] = df.iloc[:, 0:2].to_numpy()
    y_true :NDArray[np.float64] = df.iloc[:, 2].to_numpy()
    parameters :NDArray[np.float64] = np.array(initial_points, dtype=np.float64)
    number_iterations :int = 0
    previous_cost :np.float64 = np.inf
    current_cost :np.float64
    residuals :NDArray[np.float64]
    y_predicted :NDArray[np.float64]

    for _ in range(max_iter):
        number_iterations += 1
        # calcular predicoes e residuos
        y_predicted = neuron_model(x, parameters)
        residuals = y_true - y_predicted
        current_cost = cost(y_true, y_predicted, method="sse")

        # criterio de parada
        if np.abs(previous_cost - current_cost) < tol:
            break

        jacobian = calculate_jacobian(x, parameters)

        # (J^T * J + lambda * I) * delta_p = J^T * r
        J_trasposed_times_J :NDArray[np.float64] = jacobian.T @ jacobian

        jacobian = jacobian.T @ residuals

        # hessiana
        H_lm :NDArray[np.float64] = J_trasposed_times_J + damping_factor * np.eye(J_trasposed_times_J.shape[0])

        # resolver para a atualização de parametros
        try:
            delta_p = np.linalg.solve(H_lm, jacobian)
        # caso a matriz seja singular -> reduzir o amortecimento
        except np.linalg.LinAlgError:
            damping_factor *= 0.1
            continue

        # atualizar os parâmetros
        new_parameters = parameters + delta_p
        # decidir sobre o próximo passo
        new_z_predicted = neuron_model(x, parameters)
        new_cost = cost(y_true, new_z_predicted, "sse")

        if new_cost < current_cost:
            # reduz o fator de amortecimento
            parameters = new_parameters
            damping_factor *= 0.1
        else:
            # aumenta o fator de amortecimento
            damping_factor *= 10

        previous_cost = current_cost

    return (parameters, number_iterations)


def main() -> None:
    data :pd.DataFrame = pd.read_csv("src/datasets/raw/trabalho_4_dados.csv", decimal=",")
    initial_points :NDArray[np.float64] = np.array([[0.1, 0.1, 0.1],
                                                    [-0.1, 2.0, -1.0],
                                                    [0.5, -10.0, 2.0],
                                                    [5.0, -1.0, 7.0]], dtype=np.float64)

    data = convert_rows_to_float(data)

    for alpha in [0.01, 0.02, 0.1, 0.5]:
        parameters :NDArray[np.float64]
        number_iterations :int

        for i in range(0, initial_points.shape[0]):
            parameters, number_iterations = levenberg_marquardt(data,
                                                                initial_points[i, :],
                                                                alpha,
                                                                np.float64(1e-5),
                                                                1000,
                                                                1)
            print(f"Para os pontos {initial_points[i, :]} temos que:")
            print(parameters)
            print(number_iterations)
            print("=" * 80)


if __name__ == "__main__":
    main()
