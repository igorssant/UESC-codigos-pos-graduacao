import ctypes
import os


def calcular_gradiente(x :list[float],
                       y :list[float],
                       theta :list[float],
                       data_size :int,
                       learning_rate :float,
                       max_iter :int,
                       tolerance :float) -> list[float]:
    # fazendo a ligacao entre python e C
    library_name :str = "src/core/libgradient.so"
    library :ctypes.CDLL = ctypes.CDLL(os.path.join(os.getcwd(), library_name))
    
    # definindo a funcao
    library.calculate_gradient.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double
    ]
    library.calculate_gradient.restype = None
    
    # convertendo os tipos
    x_array :ctypes.ARRAY[ctypes.c_double] = (ctypes.c_double * data_size)(*x)
    y_array :ctypes.ARRAY[ctypes.c_double] = (ctypes.c_double * data_size)(*y)
    theta_array :ctypes.ARRAY[ctypes.c_double] = (ctypes.c_double * data_size)(*theta)
    
    # executando a funcao
    library.calculate_gradient(x_array, y_array, theta_array, data_size, learning_rate, max_iter, tolerance)
    
    return list(theta_array)


def main() -> None:
    x :list[float] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y :list[float] = [0.0, 1.0, 20.0, 30.0, 40.0, 50.0]
    theta :list[float] = [0.0, 0.0]
    data_size :int = len(x)
    learning_rate :float = 0.01
    max_iter :int = 500
    tolerance :float = 0.001
    result :float[float] = calcular_gradiente(x, y, theta, data_size, learning_rate, max_iter, tolerance)
    
    print("Viés:", result[0], "\nPeso:", result[1])


if __name__ == "__main__":
    # print(os.getcwd())
    main()
