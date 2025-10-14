from matplotlib import pyplot as plt
from typing import List
import numpy as np


def cost_function(x :np.ndarray|np.float64,
                  A :np.ndarray,
                  c :np.ndarray,
                  b :np.float64=np.float64(0.0)) -> np.float64:
    """
    J(x) = (x-c)^T A (x-c) + C 
    ignorando o termo constante C = b*e^T*J(x*)
    """

    x_minus_c :np.ndarray = x - c

    # (x-c)^T A (x-c)
    return x_minus_c.T @ A @ x_minus_c


def gradient_function(x :np.ndarray|np.float64,
                      A :np.ndarray,
                      c :np.ndarray) -> np.ndarray:
    """
    J(x) = 2 * A * (x-c)
    """

    return np.float64(2.0) * A @ (x - c)


def gradient_descent(x_initial :np.ndarray,
                     A :np.ndarray,
                     c :np.ndarray,
                     alpha :np.float64,
                     max_iter :int=100,
                     tolerance :np.float64=np.float64(0.0001)) -> np.ndarray:
    """
    Implementação manual da Descida de Gradiente
    Passo Constante
    """

    x :np.ndarray = x_initial
    history :List[np.ndarray] = [x.copy()]
    
    for i in range(max_iter):
        grad :np.ndarray = gradient_function(x, A, c)
        
        # verifica se o gradiente for muito próximo da tolerancia
        if np.linalg.norm(grad) < tolerance:
            break
            
        # x_k+1 = x_k - alpha * grad(J(x_k))
        x = x - alpha * grad
        history.append(x.copy())
        
    return np.array(history)


def plot_paraboloid_and_steps(A :np.ndarray,
                              c :np.float64,
                              results :dict,
                              x_star :np.ndarray) -> None:
    """
    Gera o plot do paraboloide e dos passos do Descida de Gradiente
    """

    # paraboloide
    x1 = np.linspace(-6, 6, 100)
    x2 = np.linspace(-6, 6, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # calcula J(x) em 3D
    Z = np.zeros(X1.shape)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = cost_function(x, A, c)

    plt.figure(figsize=(10, 8))

    # contornos
    CS = plt.contour(X1, X2, Z, levels=np.logspace(0, 3, 15), cmap="viridis")
    plt.colorbar(CS, label="$J(x)$")

    # passos de otimização
    colors = plt.cm.get_cmap("hsv", len(results))

    for k, (alpha, history) in enumerate(results.items()):
        # os passos
        plt.plot(history[:, 0],
                 history[:, 1], 
                 marker="o",
                 linestyle="--",
                 markersize=3, 
                 color=colors(k),
                 label=f"$\\alpha={alpha}$")
        
        # ponto Inicial
        plt.plot(history[0, 0],
                 history[0, 1], 
                 marker="s",
                 markersize=6, 
                 color=colors(k),
                 label="Início" if k == 0 else "")
    
    # ponto de minimo analitico
    plt.plot(x_star[0],
             x_star[1], 
             marker="*",
             markersize=12,
             color="red",
             label="$x^*$ Analítico")
    plt.title("Descida de Gradiente em Parabolóide (Análise de $\\alpha$)")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.show()


def main() -> None:
    A :np.ndarray = np.array([
        [2.0, 0.5],
        [0.5, 3.0]
    ])

    # O ponto de minimo analitico é x* = [1.0, 2.0]
    c :np.ndarray = np.array([1.0, 2.0], dtype=np.float64)
    b :np.float64 = np.float64(5.0)
    e :np.ndarray = np.ones((2, 1), dtype=np.float64)
    max_iter :int = 200
    tol :np.float64 = np.float64(0.00001)

    # otimizacao analitica
    x_star_analytical :np.ndarray = c
    J_min_analytical :np.float64 = cost_function(x_star_analytical, A, c)

    print(f"Ponto de mínimo analítico x*: {x_star_analytical}")
    print(f"Custo mínimo J(x*): {J_min_analytical:.6f}\n")


    # otimizacao numerica
    x_initial :np.ndarray = np.array([5.0, -5.0], dtype=np.float64)
    alphas :np.ndarray = np.array([0.01, 0.1, 0.3, 0.5], dtype=np.float64)

    results :dict = {}

    for alpha in alphas:
        history :np.ndarray = gradient_descent(x_initial, A, c, alpha, max_iter, tol)
        final_x :np.float64 = history[-1]
        final_J :np.float64 = cost_function(final_x, A, c)
        num_iter :int = len(history) - 1
    
        results[alpha] = history

        print(f"Alpha = {alpha}:")
        print(f"> Iterações: {num_iter}")
        print(f"> x* numérico: {final_x}")
        print(f"> Custo J(x*): {final_J:.6f}")

    plot_paraboloid_and_steps(A, c, results, x_star_analytical)


if __name__ == "__main__":
    main()

