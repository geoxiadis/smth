from py_wake.utils import gpu_utils
import pytest
import os
from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake import np
from py_wake.utils.layouts import rectangle
from py_wake.utils.profiling import timeit
from py_wake.tests import npt


if gpu_utils.cupy_found:
    import cupy as cp

    def test_gpu_name():
        if os.environ.get('SLURM_JOB_PARTITION', '') == 'gpuq':
            assert gpu_utils.gpu_name == 'Quadro P4000'

    def test_print_gpu_mem():
        gpu_utils.print_gpu_mem()

    def test_free_gpu_mem():
        initial = gpu_utils.mempool.total_bytes()
        cp.zeros(128 * 1024)
        before = gpu_utils.mempool.total_bytes()
        gpu_utils.free_gpu_mem(verbose=False)
        after = gpu_utils.mempool.total_bytes()
        assert after < before

    def test_pywake_gpu():
        import numpy
        np.set_backend(numpy)
        wfm = BastankhahGaussian(site=Hornsrev1Site(), windTurbines=V80())
        x, y = rectangle(100, 50, 5 * 80)

        aep_cpu, t_cpu = timeit(wfm.aep)(x, y)

        np.set_backend(cp)
        wfm.aep([0], [0])
        aep_gpu, t_gpu  = timeit(wfm.aep)(x, y)
        npt.assert_almost_equal(aep_cpu, aep_gpu)
        assert t_gpu < t_cpu
