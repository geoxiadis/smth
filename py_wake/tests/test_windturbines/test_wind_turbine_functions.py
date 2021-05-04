
from numpy import newaxis as na

import matplotlib.pyplot as plt
import numpy as np
from py_wake.tests import npt
from py_wake.utils.max_functions import StrictMax, SoftMax, SmoothMax


def test_max():
    f = 1000
    a = np.arange(1, 2.01, .01) * f
    b = a[::-1]
    A = np.array([a, b])[na, :, :, na, na]

    plt.plot(A[0, 0, :, 0, 0], '-', label='A')
    plt.plot(A[0, 1, :, 0, 0], '-', label='B')
    for max_function, label, ref in [(StrictMax(), 'max', [1.6, 1.58, 1.56, 1.54, 1.52, 1.5]),
                                     (SoftMax(.1), 'SoftMax(.1)', [1.6, 1.59, 1.57, 1.56, 1.55, 1.55]),
                                     (SoftMax(0.4), 'SoftMax(.4)', [1.64, 1.63, 1.62, 1.62, 1.61, 1.61]),
                                     (SmoothMax(.5), 'SmoothMax(.5)', [1.59, 1.56, 1.54, 1.52, 1.51, 1.5]),
                                     (SmoothMax(1), 'SmoothMax(1)', [1.56, 1.54, 1.52, 1.51, 1.5, 1.5])]:

        plt.plot(max_function(A)[0, :, 0, 0], '-', label=label)

        # print(np.round(max_function(A)[0, 40:51:2, 0, 0] / f, 2).tolist())
        npt.assert_array_almost_equal(max_function(A)[0, 40:51:2, 0, 0] / f, ref, 2)
    if 0:
        plt.legend()
        plt.show()
    plt.close('all')
