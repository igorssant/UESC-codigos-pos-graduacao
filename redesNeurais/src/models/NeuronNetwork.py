from utils.activation_functions import *
from typing import Callable, List, Tuple
from models.Neuron import Neuron
import pandas as pd
import numpy as np


class NeuronNetwork:
    def __init__(self,
                 inputSize :int,
                 layerSizes :list[int],
                 activationFunction :Callable[[np.ndarray], np.float64],
                 outputActivation :Callable[[np.ndarray], np.float64],
                 learningRate :np.float64 = np.float64(0.0001)) -> None:
        
        if inputSize < 1:
            raise ValueError("O tamanho da entrada deve ser positivo.")
        if not layerSizes:
            raise ValueError("A rede deve ter pelo menos uma camada oculta/saída.")

        self.__inputSize :int = inputSize
        self.__learningRate :np.float64 = learningRate
        self.__layers : List[List[Neuron]] = [] 
        self.__initializeNetwork(inputSize, layerSizes, activationFunction, outputActivation)


    def __initializeNetwork(self,
                            inputSize :int,
                            layerSizes :List[Neuron],
                            activationFunction :Callable[[np.ndarray], np.float64],
                            outputActivation :Callable[[np.ndarray], np.float64]) -> None:
        # o numero de entradas para a primeira camada é inputSize
        previousInputs :int = inputSize
        
        for i, quantityOfNeurons in enumerate(layerSizes):
            currentLayer :List[Neuron] = []
            # flag verificando se esta é a ultima camada
            # a famosa camada de saida
            isOutputLayer :bool = (i == len(layerSizes) - 1)
            # escolhe a função de ativação
            activationFunc :Callable[[np.ndarray], np.float64] = outputActivation\
                if isOutputLayer\
                else activationFunction

            # criando neuronios para a camada atual
            for _ in range(quantityOfNeurons):
                neuron = Neuron(
                    activationFunction=activationFunc,
                    quantityOfInputs=previousInputs
                )
                currentLayer.append(neuron)

            self.__layers.append(currentLayer)

            # o numero de entradas da proxima camada
            # é o numero de neuronios na camada atual
            previousInputs = quantityOfNeurons


    def __calculateOutputLayerDeltas(self, targets :np.ndarray) -> None:
        outputLayer :List[Neuron] = self.__layers[-1]
        
        for i, neuron in enumerate(outputLayer):
            error :np.ndarray = targets[i, 0] - neuron.getOutput()
            
            # Aqui, você precisaria da DERIVADA da função de ativação
            # No entanto, como não temos a derivada na classe Neuron,
            # vamos usar o erro puro como um delta inicial (simplificação comum)
            # ou assumir que o erro já foi multiplicado pela derivada.
            
            # Para uma implementação completa, a fórmula seria:
            # delta = error * d_activation_dz(neuron.__weightedSum)
            
            # Armazena o delta no neurônio (usaremos um atributo dinâmico aqui)
            setattr(neuron, '_delta', error)


    def __calculateHiddenLayerDeltas(self, layerIndex :int) -> None:
        currentLayer :List[Neuron] = self.__layers[layerIndex]
        nextLayer :List[Neuron] = self.__layers[layerIndex + 1]
        
        for i, neuron in enumerate(currentLayer):
            errorSum :np.float64 = np.float64(0.0)

            for nextNeuron in nextLayer:
                weightJI :np.ndarray = nextNeuron.getWeights()[i, 0]
                errorSum += weightJI * getattr(nextNeuron, '_delta')
            
            # Delta da neurona atual
            # delta_i = error_sum * d_activation_dz(neuron.__weightedSum)
            # Simplificação: assume que a derivada está implícita
            delta :np.float64 = errorSum 
            setattr(neuron, '_delta', delta)


    def feedForward(self, inputs :np.ndarray|list[float]|np.float64|float) -> np.ndarray:
        currentOutputs :np.ndarray|list[float]|np.float64|float = inputs

        for layer in self.__layers:
            nextInputs :List[np.float64] = []
 
            for neuron in layer:
                output = neuron.feedForward(currentOutputs)
                nextInputs.append(output)
                
            # a saida camada atual se torna a entrada para a proxima
            currentOutputs = np.array(nextInputs, dtype=np.float64).reshape(-1, 1)
            
        return currentOutputs


    def train(self,
              inputs :np.ndarray|list[float]|np.float64|float,
              targets :np.ndarray,
              epochs :int = 1) -> List[float]:
        errors :List[float] = []
        
        for epoch in range(epochs):
            # Feed Forward
            output :np.ndarray = self.feedForward(inputs)
            error :np.ndarray = targets - output
            mse :np.ndarray = np.mean(error**2)
            previousLayerOutputs :List[np.float64]
            currentInputs :np.ndarray

            errors.append(mse)

            # Backpropagation
            self.__calculateOutputLayerDeltas(targets)

            for i in range(len(self.__layers) - 2, -1, -1):
                self.__calculateHiddenLayerDeltas(i)
                
            for i, layer in enumerate(self.__layers):
                # para a primeira camada -> a entrada é o input original
                # para as demais -> a entrada é a saída da camada anterior
                if i == 0:
                    currentInputs = inputs
                else:
                    previousLayerOutputs = [neuron.getOutput() for neuron in self.__layers[i - 1]]
                    currentInputs = np.array(previousLayerOutputs,
                                             dtype=np.float64).reshape(-1, 1)

                for neuron in layer:
                    localGradient :np.ndarray = getattr(neuron, '_delta')
                    neuron.backPropagation(localGradient, self.__learningRate)
                    
        return errors


    def getWeightsArray(self) -> np.ndarray:
        weights :List[np.ndarray] = []

        for layer in self.__layers:
            for neuron in layer:
                weights.extend(neuron.getWeights().flatten())
                weights.append(neuron.getBias())

        return np.array(weights, dtype=np.float64)


    def setWeightsArray(self, weightsArray :np.ndarray) -> None:
        expectedSize :int = self.getWeightsArray().size

        if weightsArray.size != expectedSize:
            print(f"ERRO CRÍTICO: O vetor de pesos tem tamanho {weightsArray.size}, mas deveria ter {expectedSize}.\nA Jacobiana ou o LM falhou!")
            raise ValueError("Tamanho do vetor de pesos do LM incorreto.")

        cursor :int = 0
        
        for layer in self.__layers:
            for neuron in layer:
                numberOfWeights :int = neuron.getWeights().size

                if cursor + numberOfWeights > weightsArray.size:
                    print(f"ERRO DE ALINHAMENTO: Cursor {cursor} + {numberOfWeights} excede {weightsArray.size}")
                    raise ValueError("Desalinhamento do vetor LM.")

                newWeights :np.ndarray = weightsArray[cursor : cursor + numberOfWeights]
                newWeights = newWeights.reshape(-1, 1)
                neuron.setWeights(newWeights)
                cursor += numberOfWeights

                newBias :np.float64 = weightsArray[cursor]
                neuron.setBias(newBias)
                cursor += 1


    def calculateJacobian(self, X :np.ndarray) -> np.ndarray:
        N :int = X.shape[0]
        numberOfParams :int = len(self.getWeightsArray())
        J :np.ndarray = np.zeros((N, numberOfParams))

        hiddenLayer :List[Neurons] = self.__layers[0]
        outputNeuron :Neuron = self.__layers[1][0]
    
        for k in range(N):
            inputSample :np.ndarray = X[k, :].reshape(-1, 1)
            # obter as ativacoes
            self.feedForward(inputSample) 

            # delta da saida
            outputWeightedSum :np.float64 = outputNeuron._Neuron__weightedSum
            outputActivation : np.float64 = outputNeuron.getOutput()

            # derivada da ativacao de saida
            derivOutput = linear_derivative(outputActivation) 
            # preenchendo colunas da jacobiana
            cursor :int = 0

            # pesos da camada escondida
            for j, hiddenNeuron in enumerate(hiddenLayer):
                hiddenOutput :np.float64 = hiddenNeuron.getOutput()
                hiddenWeightedSum :np.float64 = hiddenNeuron._Neuron__weightedSum

                # backpropagation - 1 ordem
                weightOutputJ :np.float64 = outputNeuron.getWeights()[j, 0]
                deltaHidden :np.float64 = derivOutput * weightOutputJ * tanh_derivative(hiddenOutput) 
                inputValues :np.ndarray = hiddenNeuron._Neuron__inputs.flatten()

                # pesos
                for i in range(inputValues.size):
                    J[k, cursor] = deltaHidden * inputValues[i]
                    cursor += 1

                # bias
                J[k, cursor] = deltaHidden * np.float64(1.0)
                cursor += 1

            # pesos da camada de saída
            hiddenOutputsNoBias = [n.getOutput() for n in hiddenLayer]
        
            for ah in hiddenOutputsNoBias:
                J[k, cursor] = derivOutput * ah
                cursor += 1
            
            # bias
            J[k, cursor] = derivOutput * np.float64(1.0)
            cursor += 1
        
        return J

