from numpy.typing import NDArray
from typing import Tuple, Union
import pandas as pd
import numpy as np


def convert_rows_to_float(df :pd.DataFrame) -> pd.DataFrame:
    """Função convert_rows_to_float
    Recebe o dataframe e converte cada coluna para np.float64
    :param: df (pd.DataFrame).
    :return: operated_df (pd.DataFrame).
    """

    for column in df.columns:
        df[column] = df[column].astype(np.float64)
    
    return df


def normalization(data :pd.DataFrame, method :int = 0) -> pd.DataFrame:
    """Função normalization
    Essa função normaliza os dados de acordo com o método escolhido.
    Os métodos são:
        [0] -> nenhuma;
        [1] -> min/max;
        [2] -> z-score.
    :param: data                (pd.DataFrame).
    :param: method              (int).
        0 -> nenhuma;
        1 -> min/max;
        2 -> z-score.
        Por padrão o valor é '0'.
    :return: normalized_data    (pd.DataFrame)
    """

    normalized_data :np.NDArray[np.float64]

    match method:
        case 0:
            normalized_data = data.copy()
        case 1:
            min_value :NDArray[np.float64] = np.min(data, axis=0)
            max_value :NDArray[np.float64] = np.max(data, axis=0)

            normalized_data = (data - min_value) / (max_value - min_value + np.float64(1e-8))
        case 2:
            mean :np.float64 = np.mean(data, axis=0)
            std :np.float64 = np.std(data, axis=0)
            
            normalized_data = (data - mean) / (std + np.float64(1e-8))
        case _:
            raise ValueError("Método de normalização inválido.")
    
    return pd.DataFrame(data=normalized_data,
                        columns=["x", "y", "z"],
                        dtype=np.float64)


def predict(x :NDArray[np.float64],
            y :NDArray[np.float64],
            a :np.float64,
            b :np.float64,
            c :np.float64) -> NDArray[np.float64]:
    """Função predict
    Usa o modelo: 'z = a*x^3 + b*y^2 + c' e retorna seu resultado.
    :param: x   (NDArray[np.float64]).
    :param: y   (NDArray[np.float64]).
    :param: a   (np.float64).
    :param: b   (np.float64).
    :param: c   (np.float64).
    :return: z  (NDArray[np.float64]).
    """

    return a * x**3 + b * y**2 + c


def cost(y_true :NDArray[np.float64], y_pred :NDArray[np.float64], method :str = "mse") -> np.float64:
    """Função cost
    Realiza o calculo do custo.
    :param: y_true      (np.float64).
    :param: y_predicted (np.float64).
    :param: method      (String).
    """

    match method:
        case "mse":
            return np.mean((y_true - y_pred)**2)
        case "rmse":
            return np.srqt(np.mean((y_true - y_pred)**2))
        case "mae":
            return np.mean(np.abs(y_true - y_pred))
        case _:
            raise ValueError("Método de custo incorreto.\n" +
                             "Use 'mse', 'rmse' ou 'mae'.")


def calculate_jacobian(x :NDArray[np.float64],
                       y :NDArray[np.float64]) -> NDArray[np.float64]:
    """ Função calculate_jacobian
    Calcula a matriz Jacobiana dos resíduos em relação aos parâmetros a, b, c.
    J = [ dr/da, dr/db, dr/dc ]
    Onde r = z_predito - z_real
    :param: x (NDArray[np.float64]).
    :param: y (NDArray[np.float64]).
    :return: J (NDArray[np.float64]).
    """

    # d(r)/da = d(a*x^3 + b*y^2 + c - z_true)/da = x^3
    da :NDArray[np.float64] = x**3
    # d(r)/db = d(a*x^3 + b*y^2 + c - z_true)/db = y^2
    db :NDArray[np.float64] = y**2
    # d(r)/dc = d(a*x^3 + b*y^2 + c - z_true)/dc = 1
    dc :NDArray[np.float64] = np.ones_like(x)

    # jacobiana é uma matriz de N x 3
    # numero de amostras x numero de parâmetros
    return np.stack([da, db, dc], axis=1)


def levenberg_marquardt(
    df: pd.DataFrame,
    initial_points: Union[NDArray[np.float64], list[np.float64], list[float]],
    damping_factor: np.float64,
    tol: np.float64 = np.float64(1e-4),
    max_iter: int = 1000,
    normalization_method: int = 0
) -> Tuple[NDArray[np.float64], int]:
    """ Função levenberg_marquardt
    Executa o algoritmo de Levenberg-Marquardt para os valores de a, b, c.
    :param: df (pd.DataFrame).
    :param: initial_points (NDArray[np.float64]).
    :param: damping_factor (np.float64).
    :param: tol (np.float64).
    :param: max_iter (int).
    :param: normalization_method (int).
    :return: Tuple[NDArray[np.float64], int].
    """

    # normalizando o dataset
    df_normalized :pd.DataFrame = normalization(df, method=normalization_method)
    x_normalized :NDArray[np.float64] = df_normalized["x"].to_numpy()
    y_normalized :NDArray[np.float64] = df_normalized["y"].to_numpy()
    z_normalized :NDArray[np.float64] = df_normalized["z"].to_numpy()
    parameters :NDArray[np.float64] = np.array(initial_points, dtype=np.float64)
    number_iterations :int = 0
    previous_cost :np.float64 = np.inf
    current_cost :np.float64
    residuals :NDArray[np.float64]
    z_predicted :NDArray[np.float64]

    for _ in range(max_iter):
        number_iterations += 1
        # calcular predicoes e residuos
        z_predicted = predict(x_normalized, y_normalized, parameters[0], parameters[1], parameters[2])
        residuals = z_normalized - z_predicted
        current_cost = cost(z_normalized, z_predicted, method="mse")

        # criterio de parada
        if np.abs(previous_cost - current_cost) < tol:
            break

        jacobian = calculate_jacobian(x_normalized, y_normalized)

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
        new_z_predicted = predict(x_normalized,
                                  y_normalized,
                                  new_parameters[0],
                                  new_parameters[1],
                                  new_parameters[2])
        new_cost = cost(z_normalized, new_z_predicted, method="mse")

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
    data :pd.DataFrame = pd.read_csv("src/datasets/raw/trabalho_2_dados.csv", decimal=",")
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
