from wavelon import Wavelon
from typing import Callable
import numpy as np


class WaveletNetwork:
    def __init__(self,
                 inputDimension :int,
                 waveletFunction :Callable[[np.ndarray], np.ndarray]) -> None:
        if inputDimension < 1:
            raise ValueError("A quantidade de entradas deve ser um número inteiro positivo")

        if waveletFunction is None:
            raise ValueError("A função wavelet deve ser um *Callable* e deve ser passada como parâmetro para essa classe.")

        self.__wavelon = Wavelon(inputDimension,
                                 2,
                                 waveletFunction)
        quantityOfInputs = 2
        limit :np.float64 = np.float64(1.0) / np.sqrt(quantityOfInputs)
        self.__outputPerceptronWeights :np.ndarray = np.random.uniform(
            low = -limit * np.float64(0.01),
            high = limit * np.float64(0.01),
            size = (quantityOfInputs, 1) # Shape (2, 1)
        ).astype(np.float64)
        self.__outputPerceptronBias :np.float64 = np.float64(1.0)
        self.__activationFunction :Callable[[np.ndarray], np.ndarray] = lambda x: x


    def forward(self, input: np.ndarray) -> np.ndarray:
        wavelon_output: np.ndarray = self.wavelon_layer.forward(input)
        weightedSum: np.ndarray = (wavelon_output @ self.__outputPerceptronWeights) + self.__outputPerceptronBias
        final_output: np.ndarray = self.__activationFunction(weightedSum)
        
        return final_output
    
    
    def getWeightsArray(self) -> np.ndarray:
        W_flat = self.wavelon_layer.W_linear.flatten()
        B_flat = self.wavelon_layer.B_translation.flatten()
        C_flat = self.wavelon_layer.C_dilation.flatten()
        A_flat = self.wavelon_layer.A_output.flatten()
        Wout_flat = self.__outputPerceptronWeights.flatten()
        Bias_out_flat = np.array([self.__outputPerceptronBias])
        
        return np.concatenate([W_flat, B_flat, C_flat, A_flat, Wout_flat, Bias_out_flat])

