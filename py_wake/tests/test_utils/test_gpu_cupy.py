import os

import numpy
import pytest

from py_wake import np
from py_wake.deficit_models import BastankhahGaussianDeficit
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.tests import npt
from py_wake.utils import gpu_utils
from py_wake.utils.layouts import rectangle
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind

if gpu_utils.cupy_found:
    import cupy as cp

    def test_gpu_name():
        import socket
        if socket.gethostname() == 'DTU-MM156971':
            assert gpu_utils.gpu_name == 'NVIDIA GeForce RTX 5070'

    def test_print_gpu_mem():
        gpu_utils.print_gpu_mem()

    def test_free_gpu_mem():
        initial = gpu_utils.mempool.total_bytes()
        cp.zeros(128 * 1024)
        before = gpu_utils.mempool.total_bytes()
        gpu_utils.free_gpu_mem(verbose=False)
        after = gpu_utils.mempool.total_bytes()
        assert after < before

    @pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
    def test_cupy(wfm_cls):
        kwargs = dict(site=Hornsrev1Site(), windTurbines=V80(),
                      wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True))

        def get_aep(np_backend):
            np.set_backend(np_backend)
            wfm = wfm_cls(**kwargs)
            x, y = rectangle(16, 4, 80 * 5)
            n_cpu = 1  # n_cpu = (1, 2)[np_backend == numpy]
            wfm.aep([0], [0], n_cpu=n_cpu)  # setup multiprocessing pool
            P = wfm(x, y, n_cpu=n_cpu).Power
            aep = wfm.aep(x, y, n_cpu=n_cpu)
            np.set_backend(numpy)
            return aep

        aep_cpu = get_aep(numpy)
        aep_gpu = get_aep(cp)

        with pytest.raises(AssertionError):
            npt.assert_array_equal(aep_cpu, aep_gpu.get())
        npt.assert_array_almost_equal(aep_cpu, aep_gpu.get(), 10)
        assert aep_gpu.__class__ == cp.ndarray
