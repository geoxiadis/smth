import pytest

import numpy as np
from py_wake.deficit_models.noj import NOJ, NOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, HornsrevV80, wt9_y, wt9_x, wt16_x, wt16_y
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.utils.profiling import timeit
from py_wake.utils.parallelization import get_pool
import multiprocessing


def test_yaw_wrong_name():
    wfm = NOJ(Hornsrev1Site(), V80())
    for k in ['yaw_ilk', 'Yaw']:
        with pytest.raises(ValueError, match=r'Custom \*yaw\*\-keyword arguments not allowed'):
            wfm([0], [0], **{k: [[[30]]]})


def test_yaw_dimensions():
    site = Hornsrev1Site()
    windTurbines = HornsrevV80()
    wake_deficitModel = NOJDeficit()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=wake_deficitModel)

    x, y = wt9_x, wt9_y

    I, L, K = 9, 360, 23
    for yaw in [45,
                np.broadcast_to(45, (I,)),
                np.broadcast_to(45, (I, L)),
                np.broadcast_to(45, (I, L, K)),
                ]:
        sim_res_all_wd = wf_model(x, y, yaw=yaw)
        if len(np.shape(yaw)) > 1:
            yaw1 = yaw[:, 1:2]
        else:
            yaw1 = yaw
        sim_res_1wd = wf_model(x, y, wd=1, yaw=yaw1)

        npt.assert_almost_equal(sim_res_all_wd.WS_eff.sel(wt=1, wd=1, ws=10), 9.70670076)
        npt.assert_almost_equal(sim_res_1wd.WS_eff.sel(wt=1, ws=10), 9.70670076)


def test_calc_wt_interaction_parallel_results():
    site = Hornsrev1Site()
    x, y = wt16_x, wt16_y
    wt = HornsrevV80()
    wfm = NOJ(site, wt)
    (WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, wt_kwargs), tn = timeit(wfm.calc_wt_interaction_parallel)(
        x, y, wd_chunks=3, ws_chunks=2, processes=2)
    (WS_ref, TI_ref, power_ref, ct_ref, lw_ref, wt_kwargs_ref), t1 = timeit(wfm.calc_wt_interaction)(x, y)
    npt.assert_array_equal(WS_eff_ilk, WS_ref)
    npt.assert_array_equal(TI_eff_ilk, TI_ref)
    npt.assert_array_equal(power_ilk, power_ref)
    npt.assert_array_equal(ct_ilk, ct_ref)


def test_calc_wt_interaction_parallel_speed():
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wt = HornsrevV80()
    wfm = NOJ(site, wt)
    _, tn = timeit(wfm.calc_wt_interaction_parallel, min_runs=2)(
        x, y, wd_chunks=3, ws_chunks=2)
    _, t1 = timeit(wfm.calc_wt_interaction)(x, y)
    if multiprocessing.cpu_count() >= 3:
        assert np.min(tn) < np.min(t1)
