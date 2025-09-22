from numpy.typing import NDArray
from typing import Tuple, Union
import pandas as pd
import numpy as np


def __convert_rows_to_float(df :pd.DataFrame) -> pd.DataFrame:
    """função __convert_rows_to_float
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
    """função predict
    Usa o modelo: 'z = a*x^3 + b*y^2 + c' e retorna seu resultado.
    :param: x   (NDArray[np.float64]).
    :param: y   (NDArray[np.float64]).
    :param: a   (np.float64).
    :param: b   (np.float64).
    :param: c   (np.float64).
    :return: z  (NDArray[np.float64]).
    """

    return a * x**3 + b * y**2 + c


def cost(y_true :np.float64, y_pred :np.float64, method :str = "mse") -> np.float64:
    """função cost
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


def gradients(x :NDArray[np.float64],
              y :NDArray[np.float64],
              z_true :NDArray[np.float64],
              a :np.float64,
              b :np.float64,
              c :np.float64,
              method :str = "mse") -> NDArray[np.float64]:
    """função gradients
    Calcula gradientes em relação: a, b, c.
    :param: x       (NDArray[np.float64]).
    :param: y       (NDArray[np.float64]).
    :param: z_true  (NDArray[np.float64]).
    :param: a       (np.float64).
    :param: b       (np.float64).
    :param: c       (np.float64).
    :param: method  (String).
    """

    z_predicted :NDArray[np.float64] = predict(x, y, a, b, c)
    n :int = len(z_true)
    da :np.float64
    db :np.float64
    dc :np.float64

    if method == "mse":
        error :NDArray[np.float64] = z_predicted - z_true

        da = (2/n) * np.sum(error * (x**3))
        db = (2/n) * np.sum(error * (y**2))
        dc = (2/n) * np.sum(error)
    elif method == "mae":
        error :NDArray[np.float64] = np.sign(z_predicted - z_true)

        da = (1/n) * np.sum(error * (x**3))
        db = (1/n) * np.sum(error * (y**2))
        dc = (1/n) * np.sum(error)
    else:
        raise ValueError("Método de custo incorreto.\n" +
                         "Use 'mse', 'rmse' ou 'mae'.")

    return np.array([da, db, dc], dtype=np.float64)


def gradient_descent(df :pd.DataFrame,
                     initial_points : Union[NDArray[np.float64], list[np.float64], list[float]],
                     alpha :np.float64,
                     tol :np.float64 = np.float64(1e-4),
                     max_iter :int = 1000,
                     cost_method :str = "mse",
                     normalization_method :int = 0) -> Tuple[NDArray[np.float64], list[np.float64]]:
    """função gradient_descent
    Executa descida de gradiente para os valores de a, b, c passados.
    :param: df                      (pd.DataFrame)
    :param: initial_points          (NDArray[np.float64]).
    :param: alpha                   (np.float64).
    :param: max_iter                (int).
        valor padrão: 1000.
    :param: cost_method             (String).
        valor padrão: "mse".
    :param: normalization_method    (int).
        valor padrão: 0.
    :return: Tuple[NDArray[np.float64], list[np.float64]].
    """

    # normalizando o dataset
    df_normalized :pd.DataFrame = normalization(df, method=normalization_method)
    x_normalized :NDArray[np.float64] = df_normalized["x"].to_numpy()
    y_normalized :NDArray[np.float64] = df_normalized["y"].to_numpy()
    z_normalized :NDArray[np.float64] = df_normalized["z"].to_numpy()
    # gradients_history: list[np.float64] = list()
    parameters :NDArray[np.float64] = np.array(initial_points, dtype=np.float64)
    number_iterations :int = 0
    previous_J :np.float64 = np.float64(0.0)

    for _ in range(0, max_iter):
        # funcional
        z_predicted :NDArray[np.float64] = predict(x_normalized,
                                                   y_normalized,
                                                   parameters[0],
                                                   parameters[1],
                                                   parameters[2])
        # funcao de custo
        J :np.float64 = cost(z_normalized, z_predicted, method=cost_method)
        
        # gradients_history.append(J)
        number_iterations += 1

        # atualizando os gradientes
        new_gradients :NDArray[np.float64] = gradients(x_normalized,
                                                       y_normalized,
                                                       z_normalized,
                                                       parameters[0],
                                                       parameters[1],
                                                       parameters[2],
                                                       method=cost_method)
        
        # atualizando os parametros
        parameters -= alpha * new_gradients

        if abs(J - previous_J) < tol:
            break
        
        previous_J = J

    # return (parameters, gradients_history)
    return (parameters, number_iterations)


def main():
    data :pd.DataFrame = pd.read_csv("src/datasets/raw/trabalho_2_dados.csv", decimal=",")
    initial_points :NDArray[np.float64] = np.array([[0.1, 0.1, 0.1],
                                                    [-0.1, 2.0, -1.0],
                                                    [0.5, -10.0, 2.0],
                                                    [5.0, -1.0, 7.0]], dtype=np.float64)

    data = __convert_rows_to_float(data)

    for alpha in [0.01, 0.02, 0.1, 0.5]:
        parameters :NDArray[np.float64]
        number_iterations :int

        for i in range(0, initial_points.shape[0]):
            parameters, number_iterations = gradient_descent(data,
                                                             initial_points[i, :],
                                                             alpha,
                                                             np.float64(1e-5),
                                                             1000,
                                                             "mse",
                                                             1)
            print(f"Para os pontos {initial_points[i, :]} temos que:")
            print(parameters)
            print(number_iterations)
            print("=" * 80)


if __name__ == "__main__":
    main()
