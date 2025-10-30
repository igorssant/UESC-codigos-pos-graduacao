from models.Neuron import SimpleNeuron
from models.Wavelon import Wavelon
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
        self.__neuron = SimpleNeuron(2, 1)


    def getWeightsArray(self) -> np.ndarray:
        wavelonLinearWeights :np.ndarray = self.__wavelon.getLinearWeights().flatten()
        translation :np.ndarray = self.__wavelon.getTranslation().flatten()
        dilation :np.ndarray = self.__wavelon.getDilation().flatten()
        wavelonWeights :np.ndarray = self.__wavelon.getOutputWeights().flatten()
        neuronOutput :np.ndarray = self.__neuron.getWeights().flatten()
        neuronBias :np.ndarray = np.array([self.__neuron.getBias()], dtype=np.float64)
 
        return np.concatenate([wavelonLinearWeights,
                               translation,
                               dilation,
                               wavelonWeights,
                               neuronOutput,
                               neuronBias])


    def forward(self, input: np.ndarray) -> np.ndarray:
        wavelonOutput :np.ndarray = self.__wavelon.forward(input)
        neuronWeights :np.ndarray = self.__neuron.getWeights()
        weightedSum :np.ndarray = (wavelonOutput @ neuronWeights) + self.__neuron.getBias()
        finalOutput :np.ndarray = self.__neuron.forward(weightedSum)
        
        return finalOutput

