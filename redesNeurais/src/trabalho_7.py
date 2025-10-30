from models.WaveletNetwork import WaveletNetwork
import numpy as np


def morlet_wavelet(x):
    return np.cos(1.75 * x) * np.exp(-x**2 / 2)


def main() -> None:
    X_input = np.random.randn(10, 5)
    wavelon_net = WaveletNetwork(5, morlet_wavelet)
    output = wavelon_net.forward(X_input)
    
    print(f"Entrada (shape): {X_input.shape}")
    print(f"Saída Wavelon (shape): {output.shape}")
    print(output)
    # print("\nParâmetros Ajustáveis:")
    # print(f"Pesos W (shape): {wavelon_layer.getLinearWeights().shape}")
    # print(f"Translação B (shape): {wavelon_layer.getTranslation().shape}")
    # print(f"Dilatação C (shape): {wavelon_layer.getDilation().shape}")
    # print(f"Peso de Saída A (shape): {wavelon_layer.getOutputWeights().shape}")


if __name__ == "__main__":
    main()
 
