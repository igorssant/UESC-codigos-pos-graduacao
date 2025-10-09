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
            low=-limit,
            high=limit,
            size=(quantityOfInputs, 1)
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
        gradientBias :np.float64 = localGradient
        gradientWeights :np.ndarray = localGradient * self.__inputs
        
        self.__weights -= learningRate * gradientWeights
        self.__bias -= learningRate * gradientBias
        
        return self.__weights * localGradient

