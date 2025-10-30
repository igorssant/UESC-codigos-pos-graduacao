from typing import Callable
import pandas as pd
import numpy as np


class Neuron:
    def __init__(self,
                 activationFunction :Callable[[np.ndarray], np.float64],
                 quantityOfInputs :int,
                 bias :np.float64 = np.float64(1.0)) -> None:
        if quantityOfInputs < 1:
            raise ValueError("A quantidade de entradas deve ser um número inteiro positivo.")

        self.__activationFunction :Callable[[np.ndarray], np.float64] = activationFunction
        self.__inputs :np.ndarray = np.zeros((quantityOfInputs, 1), dtype=np.float64)
        self.__bias :np.float64 = bias

        limit :np.float64 = np.float64(1.0) / np.sqrt(quantityOfInputs)

        self.__weights :np.ndarray = np.random.uniform(
            low = -limit * np.float64(0.01),
            high = limit * np.float64(0.01),
            size = (quantityOfInputs, 1)
        ).astype(np.float64)
        self.__weightedSum :np.float64 = np.float64(0.0)
        self.__output :np.float64 = np.float64(0.0)


    def getWeights(self) -> np.ndarray:
        return self.__weights


    def setWeights(self, newWeights :np.ndarray) -> None:
        if newWeights.shape != self.__weights.shape:
             raise ValueError("O novo vetor de pesos deve ter a mesma dimensão que o vetor de entradas.")

        self.__weights = newWeights


    def getOutput(self) -> np.float64:
        return self.__output


    def getBias(self) -> np.float64:
        return self.__bias


    def setBias(self, newBias :np.float64) -> None:
        self.__bias = newBias


    def feedForward(self, inputs :np.ndarray|list[float]|np.float64|float) -> np.ndarray:
        if isinstance(inputs, (list, float)):
            inputs = np.array(inputs, dtype=np.float64).reshape(-1, 1)

        if inputs.shape != self.__inputs.shape:
            raise ValueError(f"O array de inputs deve ter dimensão {self.__inputs.shape[0]}x1.")

        self.__inputs = inputs

        weightedSum :np.ndarray = np.dot(self.__weights.T, self.__inputs) + self.__bias

        self.__weightedSum = weightedSum[0, 0]
        self.__output = self.__activationFunction(self.__weightedSum)

        return self.__output


    def backPropagation(self, localGradient :np.float64, learningRate :np.float64) -> np.ndarray:
        # se o gradiente local for NaN -> retornar 0
        # para acitar futuras complicacoes
        if np.isnan(localGradient):
            return np.zeros_like(self.__weights) * 0.0 

        clipLimit :np.float64 = np.float64(1.0)

        localGradient = np.clip(localGradient, -clipLimit, clipLimit)

        gradientBias :np.float64 = localGradient
        gradientWeights :np.ndarray = localGradient * self.__inputs
        
        self.__weights -= learningRate * gradientWeights
        self.__bias -= learningRate * gradientBias
        
        return self.__weights * localGradient



class SimpleNeuron:
    def __init__(self,
                 quantityOfInputs :int,
                 quantityOfOutputs :int,
                 activationFunction :Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        if quantityOfInputs < 1 or quantityOfOutputs < 1:
            raise ValueError("A quantidade de entradas e saídas deve ser um número inteiro positivo.")

        limit :np.float64 = np.float64(1.0) / np.sqrt(quantityOfInputs)
        self.__weights :np.ndarray = np.random.uniform(
            -limit * np.float64(0.01),
            limit * np.float64(0.01),
            size=(quantityOfInputs, quantityOfOutputs)
        )
        self.__bias :np.float64 = np.float64(1.0)
        self.__activationFunction :Callable[[np.ndarray], np.ndarray] = activationFunction\
            if activationFunction is not None\
            else lambda x: x


    def setWeights(self, newWeights :np.ndarray) -> None:
        self.__weights = newWeights


    def getWeights(self) -> np.ndarray:
        return self.__weights


    def setBias(self, newBias :np.float64) -> None:
        self.__bias = newBias


    def getBias(self) -> np.float64:
        return self.__bias


    def setActivationFunction(self,
                              newActivationFunction :Callable[[np.ndarray], np.ndarray]) -> None:
        self.__activationFunction = newActivationFunction 


    def getActivationFunction(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.__activationFunction


    def forward(self, input :np.ndarray) -> np.ndarray:
        return self.__activationFunction(input)

