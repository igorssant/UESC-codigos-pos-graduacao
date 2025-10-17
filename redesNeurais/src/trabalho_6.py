from utils.activation_functions import *
from models.NeuronNetwork import NeuronNetwork
from matplotlib import pyplot as plt
from utils.utils import *
from typing import Tuple
import pandas as pd
import numpy as np


def dataset_to_float(df :pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        df[column] = df[column].astype(np.float64)
    
    return df


def normalize_data(data :pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_value :np.ndarray = np.min(data, axis=0)
    max_value :np.ndarray = np.max(data, axis=0)
    range_value :np.ndarray = max_value - min_value

    range_value[range_value == 0] = 1 
    
    normalized_data :np.ndarray = (data - min_value) / range_value

    return (normalized_data, min_value, max_value)


def denormalize_output(normalized_output :np.ndarray,
                       min_value :np.float64|float,
                       max_value :np.float64|float) -> np.ndarray:
    return normalized_output * (max_value - min_value) + min_value


def split_data(X :np.ndarray,
               Y :np.ndarray,
               train_ratio :float=0.6,
               val_ratio :float=0.2) -> Tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray,
                                              np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
    N :int = X.shape[0]
    train_size :int = int(train_ratio * N)
    val_size :int = int(val_ratio * N)
    indices :np.ndarray = np.random.permutation(N)

    X_train :np.ndarray = X[indices[:train_size]]
    Y_train :np.ndarray = Y[indices[:train_size]]
    X_val :np.ndarray = X[indices[train_size:train_size + val_size]]
    Y_val :np.ndarray = Y[indices[train_size:train_size + val_size]]
    X_test :np.ndarray = X[indices[train_size + val_size:]]
    Y_test :np.ndarray = Y[indices[train_size + val_size:]]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def predict_batch(network :NeuronNetwork, X_data: np.ndarray) -> np.ndarray:
    """
    Realiza o feedForward para um conjunto completo de dados (X_data),
    iterando amostra por amostra, pois a rede só aceita uma entrada 1D.
    Retorna Y_pred no formato (N_amostras, N_outputs).
    """

    N = X_data.shape[0]

    if network._NeuronNetwork__layers and network._NeuronNetwork__layers[-1]:
        N_outputs = len(network._NeuronNetwork__layers[-1])
    else:
        N_outputs = 1 

    Y_pred = np.zeros((N, N_outputs), dtype=np.float64)

    for i in range(N):
        input_sample = X_data[i, :].reshape(-1, 1)
        output = network.feedForward(input_sample).flatten()
        Y_pred[i, :] = output

    return Y_pred


def train_lm_full(network :NeuronNetwork, 
                  X_train :np.ndarray,
                  Y_train :np.ndarray,
                  X_val :np.ndarray,
                  Y_val :np.ndarray,
                  max_epochs :int=1000, 
                  initial_lambda :np.float64|float=0.001, 
                  patience :int=20) -> Tuple[NeuronNetwork, dict]:
    history = {
        "train_mse": [],
        "val_mse": [],
        "epochs": []
    }
    best_val_mse = np.inf
    epochs_no_improvement = 0
    a = network.getWeightsArray()
    lambda_val = initial_lambda
    v = np.float64(2.0)
    LIMIT_V = np.float64(1e10)
    LIMIT_LAMBDA = np.float64(1e30)
    # custo inicial
    Y_pred_train = predict_batch(network, X_train)
    cost = np.float64(0.5) * np.sum((Y_train - Y_pred_train)**2)
    # validacao inicial
    Y_pred_val = predict_batch(network, X_val)
    val_mse = mean_squared_error(Y_val, Y_pred_val)

    for epoch in range(max_epochs):
        # backpropagation - 1 ordem
        J = network.calculateJacobian(X_train)
        residuals = Y_train - predict_batch(network, X_train)
        # hessiana
        A = J.T @ J
        g = J.T @ residuals

        # teste de convergencia
        if np.linalg.norm(g, ord=np.inf) < 1e-6:
            break

        # atualização de lambda
        # busca por melhor passo
        while True:
            A_lm = A + lambda_val * np.identity(len(a))

            try:
                # resolver para delta_a
                delta_a = np.linalg.solve(A_lm, g)
            except np.linalg.LinAlgError:
                lambda_val *= v
                v *= 2
                continue
            
            # Novo vetor de pesos (tentativa)
            delta_a = delta_a.flatten()
            a_new = a + delta_a
            network.setWeightsArray(a_new)

            # custo & residuos
            Y_pred_train_new = predict_batch(network, X_train)
            cost_new = np.float64(0.5) * np.sum((Y_train - Y_pred_train_new)**2)

            # avalia a qualidade do passo
            reduction_real = cost - cost_new
            reduction_predicted = delta_a.T @ (lambda_val * delta_a - g) / np.float64(2.0)
            reduction_predicted = reduction_predicted[0]

            if reduction_predicted == 0:
                rho = -1.0
            else:
                rho = reduction_real / reduction_predicted

            if rho.item() > 0.0:
                # aceita a_new & reduz lambda
                a = a_new
                cost = cost_new
                lambda_val *= max(1/3, 1 - (2 * rho - 1)**3)
                v = 2
                break
            else:
                # Rejeita a_new & aumenta lambda
                lambda_val *= v
                v *= 2
                network.setWeightsArray(a)

                if lambda_val > LIMIT_LAMBDA or v > LIMIT_V:
                    network.setWeightsArray(a)
                    lambda_val = LIMIT_LAMBDA
                    v = np.float64(2.0)
                    break

        # avaliacao
        Y_pred_val = predict_batch(network, X_val)
        val_mse = mean_squared_error(Y_val, Y_pred_val)
        train_mse = mean_squared_error(Y_train, Y_pred_train_new)

        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['epochs'].append(epoch)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_weights = a.copy()
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

            if epochs_no_improvement >= patience:
                print(f"Early Stopping na época {epoch}!")
                break

    # melhor modelo encontrado antes do overfitting
    network.setWeightsArray(best_weights)
    history["best_val_mse"] = best_val_mse

    return (network, history) 


def main() -> None:
    input_size :int = 2
    layer_sizes :list[int] = [2, 1]
    epochs :int = 100
    tolerance :np.float64 = np.float64(0.0001)
    raw_df :pd.DataFrame = pd.read_csv("datasets/raw/trabalho_5_dados.csv", decimal=",")
    converted_df :pd.DataFrame = dataset_to_float(raw_df)
    X_normalized :np.ndarray
    Y_normalized :np.ndarray
    x_min :np.ndarray
    x_max :np.ndarray
    y_min :np.ndarray
    y_max :np.ndarray
    X_train :np.ndarray
    Y_train :np.ndarray
    X_val :np.ndarray
    Y_val :np.ndarray
    X_test :np.ndarray
    Y_test :np.ndarray

    X :np.ndarray = converted_df[["Entrada 01", "Entrada 02"]].values
    Y :np.ndarray = converted_df["Saída"].values.reshape(-1, 1)

    X_normalized, x_min, x_max = normalize_data(X)
    Y_normalized, y_min, y_max = normalize_data(Y)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X_normalized, Y_normalized)

    input_size :int = X_train.shape[1]
    output_size :int = Y_train.shape[1]
    best_val_mse :np.float64 = np.inf
    best_network :NeuronNetwork = None
    all_results :dict = {}

    # encontrar o melhor N de neuronios
    # & testar generalizacao
    for n_hidden in [2, 3, 4, 5, 6, 7, 8]:
        layer_sizes = [n_hidden, output_size]
        network = NeuronNetwork(input_size,
                                layer_sizes,
                                tanh,
                                linear,
                                np.float64(0.0))
        # treinamento por levenberg-marquardt
        trained_net, history = train_lm_full(network, X_train, Y_train, X_val, Y_val)

        # avaliacao no teste
        Y_test_pred = predict_batch(trained_net, X_test)
        test_mse = mean_squared_error(Y_test, Y_test_pred)
        test_r2 = r_squared(Y_test, Y_test_pred)

        all_results[n_hidden] = {
            "network" : trained_net,
            "history" : history,
            "test_mse" : test_mse,
            "test_r2" : test_r2
        }

        if history["best_val_mse"] < best_val_mse:
            best_val_mse = history["best_val_mse"]
            best_network = trained_net

    print("Melhor rede:", best_network)
    print("Melhor valor de MSE:", best_val_mse)
    print("Histórico:\n", all_results)


if __name__ == "__main__":
    main()

