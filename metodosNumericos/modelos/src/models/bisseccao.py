import numpy as np
from utils.parser import evaluate_one_variable

class Bissection:
    def __init__(self, function:str, xL:np.float64, xU:np.float64,
                 tol:np.float64, max_iter:int) -> None:
        self._function:str = function
        self._xL:np.float64 = xL
        self._xU:np.float64 = xU
        self._tol_:np.float64 = tol
        self._max_iter_:int = max_iter
        self._result = np.float64(0.0)
        
    def executeMethod(self) -> None:
        iter:int = 0
        absoluteError:np.float64 = np.float64(100.0)
        x:np.float64 = self._xL
        fL:np.float64 = evaluate_one_variable(self._function, self._xL)
        
        while (absoluteError > self._tol_) and (iter < self._max_iter_):
            xOld:np.float64 = x
            x = (self._xL + self._xU) / 2
            fX:np.float64 = evaluate_one_variable(self._function, x)
            iter = iter + 1
            
            if x != np.float64(0.0):
                absoluteError = abs((x - xOld) / x) * 100
            
            if (fL * fX) < 0:
                self._xU = x
            else:
                self._xL = x
                fL = fX
                
        self._result = x

    def getResult(self) -> np.float64:
        return self._result
