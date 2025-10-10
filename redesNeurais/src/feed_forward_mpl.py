from utils.activation_functions import *
from models.NeuronNetwork import NeuronNetwork
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
    X :np.ndarray = converted_df[["Entrada 01", "Entrada 02"]].values
    Y :np.ndarray = converted_df["Saída"].values.reshape(-1, 1)

    X_normalized, x_min, x_max = normalize_data(X)
    Y_normalized, y_min, y_max = normalize_data(Y)

    network :NeuronNetwork = NeuronNetwork(input_size,
                                           layer_sizes,
                                           tanh,
                                           linear,
                                           np.float64(0.001))
    inputs :np.ndarray = X_normalized
    targets :np.ndarray = Y_normalized
    number_of_samples :int = inputs.shape[0]

    total_MSE : np.float64

    for epoch in range(0, epochs):
        total_MSE = np.float64(0.0)

        for i in range(number_of_samples):
            input_sample = inputs[i, :].reshape(-1, 1)
            target_sample = targets[i, :].reshape(-1, 1)
            network.train(input_sample, target_sample, epochs=1)

            output :np.ndarray = network.feedForward(input_sample)

            total_MSE += np.mean((target_sample - output)**2)
        

        average_MSE :np.float64 = total_MSE / number_of_samples

        if average_MSE <= tolerance:
            break
    
    # previsao
    final_predictions_normalized :np.ndarray = np.array(
        [network.feedForward(X_normalized[i, :].reshape(-1, 1)) for i in range(number_of_samples)],
        dtype=np.float64
    )
    # desnormalizar a saída
    final_predictions :np.ndarray = denormalize_output(final_predictions_normalized,
                                                       y_min,
                                                       y_max)

    print(final_predictions)


if __name__ == "__main__":
    main()

