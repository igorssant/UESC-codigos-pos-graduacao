from typing import Callable
import numpy as np


class Wavelon:
    def __init__(self,
                 inputDimension :int,
                 outputDimension :int,
                 waveletFunction :Callable[[np.ndarray], np.ndarray]) -> None:
        if inputDimension < 1 or outputDimension < 1:
            raise ValueError("A quantidade de entradas e a quantidade de saída devem ser números inteiros positivos.")

        if waveletFunction is None:
            raise ValueError("A função wavelet deve ser um *Callable* e deve ser passada como parâmetro para essa classe.")

        self.__linearWeights :np.ndarray = np.random.randn(inputDimension, outputDimension) * np.float64(0.1)
        self.__translation :np.ndarray = np.random.randn(1, outputDimension) * np.float64(0.1)
        self.__dilation :np.ndarray = np.ones((1, outputDimension), dtype=np.float64) * np.float64(0.5)
        self.__outputWeights :np.ndarray = np.random.randn(1, outputDimension) * np.float64(0.1)
        self.__waveletFunction :Callable[[np.ndarray], np.ndarray] = waveletFunction


    def getLinearWeights(self) -> np.ndarray:
        return self.__linearWeights


    def setLinearWeights(self, newLinearWeights :np.ndarray) -> None:
        self.__linearWeights = newLinearWeights


    def getOutputWeights(self) -> np.ndarray:
        return self.__outputWeights


    def setOutputWeights(self, newOutputWeights :np.ndarray) -> None:
        self.__outputWeights = newOutputWeights


    def getTranslation(self) -> np.ndarray:
        return self.__translation


    def setTranslation(self, newTranslation :np.ndarray) -> None:
        self.__translation = newTranslation


    def getDilation(self) -> np.ndarray:
        return self.__dilation


    def setDilation(self, newDilation :np.ndarray) -> None:
        self.__dilation = newDilation


    def setWaveletFunction(self, newWaveletFunction :Callable[[np.ndarray], np.ndarray]) -> None:
        self.__waveletFunction = newWaveletFunction


    def getWaveletFunction() -> Callable[[np.ndarray], np.ndarray]:
        return self.__waveletFunction


    def forward(self, input :np.ndarray) -> np.ndarray:
        linearNormalSum :np.ndarray = input @ self.__linearWeights
        childWaveletOutput :np.ndarray = (linearNormalSum - self.__translation) / self.__dilation
        waveletOutput :np.ndarray = self.__waveletFunction(childWaveletOutput)

        return self.__outputWeights * waveletOutput

