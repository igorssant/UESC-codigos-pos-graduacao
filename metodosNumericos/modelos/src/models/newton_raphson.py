import numpy as np
from utils.parser import evaluate_one_variable

class NewtonRaphson:
    def __init__(self, function:str, derivative:str,
                 x0:np.float64, tol:np.float64, max_iter:int) -> None:
        self._function:str = function
        self._derivative:str = derivative
        self._x0:np.float64 = x0
        self._tol_:np.float64 = tol
        self._max_iter_:int = max_iter
        self._result = np.float64(0.0)

    def executeMethod(self) -> None:
        iter:int = 0
        absoluteError:np.float64 = np.float64(100.0)
        x:np.float64 = self._x0
        
        while (absoluteError > self._tol_) and (iter < self._max_iter_):
            self._x0  = x
            iter = iter + 1
            x = self._x0 - (evaluate_one_variable(self._function, self._x0) /
                            evaluate_one_variable(self._derivative, self._x0))
            
            if x != np.float64(0.0):
                absoluteError = abs((x - self._x0) / x) * 100
                
        self._result = x

    def getResult(self) -> np.float64:
        return self._result
