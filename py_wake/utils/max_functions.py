import numpy as np
from numpy import newaxis as na


class StrictMax():

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, loads_siilk):
        return loads_siilk.max(1)


class SoftMax():
    def __init__(self, base=.1):
        """Overpredict the maximum of similar values

        Parameters
        ----------
        soft_max_base : float
            smoothing factor ]0;1[. higher number gives more smooth transition
        """
        self.alpha = 1 / base

    def __str__(self):
        return f"{self.__class__.__name__}({1/self.alpha})"

    def __call__(self, loads_siilk):
        # factor used to reduce numerical errors in power
        f = (loads_siilk.mean((1, 2, 3, 4)) / 10)[:, na, na, na, na]
        loads_silk = np.log((self.alpha**(loads_siilk / f)).sum(1)) / np.log(self.alpha) * f[:, 0]
        return loads_silk


class SmoothMax(SoftMax):
    def __init__(self, base=.5):
        """Underpredict the maximum of similar values

        Parameters
        ----------
        smooth_max_base : float
            smoothing factor ]0;1[. higher number gives more smooth transition
        """
        self.alpha = 1 / base

    def __call__(self, loads_siilk):
        # factor used to reduce numerical errors in power
        f = (loads_siilk.mean((1, 2, 3, 4)) / 10)[:, na, na, na, na]
        loads_siilk = loads_siilk / f
        return np.sum(loads_siilk * np.exp(self.alpha * loads_siilk), 1) / \
            np.sum(np.exp(self.alpha * loads_siilk), 1) * f[:, 0]
