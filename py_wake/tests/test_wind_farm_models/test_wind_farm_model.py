import pytest
import numpy as np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.deficit_models.noj import NOJ, NOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, HornsrevV80, wt9_y, wt9_x, wt16_x, wt16_y
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.tests import npt
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.profiling import timeit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
import matplotlib.pyplot as plt
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.flow_map import XYGrid


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
    WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, *_ = wfm.calc_wt_interaction(x, y, wd_chunks=3, ws_chunks=2, n_cpu=None)
    WS_ref, TI_ref, power_ref, ct_ref, *_ = wfm.calc_wt_interaction(x, y)
    npt.assert_array_equal(WS_eff_ilk, WS_ref)
    npt.assert_array_equal(TI_eff_ilk, TI_ref)
    npt.assert_array_equal(power_ilk, power_ref)
    npt.assert_array_equal(ct_ilk, ct_ref)


def test_aep_parallel_results():
    site = Hornsrev1Site()
    x, y = wt16_x, wt16_y
    wt = HornsrevV80()
    wfm = NOJ(site, wt)
    aep = wfm.aep(x, y, wd_chunks=3, ws_chunks=2, n_cpu=None)
    aep_ref = wfm.aep(x, y)
    npt.assert_array_equal(aep, aep_ref)


@pytest.mark.parametrize('wrt_arg', ['x', 'y',
                                     'xy', 'h',
                                     #'wd', 'ws'
                                     'yaw'
                                     ])
def test_dAEP_function(wrt_arg):
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, wt, deflectionModel=JimenezWakeDeflection())
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wt.hub_height(),
              'wd': [0], 'ws': 9.8, 'yaw': np.arange(1, 6).reshape((5, 1, 1)) * 5}
    dAEP_autograd = wfm.daep(gradient_method=autograd, wrt_arg=wrt_arg)(**kwargs)
    dAEP_cs = wfm.daep(gradient_method=cs, wrt_arg=wrt_arg)(**kwargs)
    dAEP_fd = wfm.daep(gradient_method=fd, wrt_arg=wrt_arg)(**kwargs)

    if 0:
        ax1, ax2 = plt.subplots(1, 2)[1]
        wfm(**kwargs).flow_map(XYGrid(resolution=100)).plot_wake_map(ax=ax1)
        ax2.set_title(wrt_arg)
        ax2.plot(dAEP_autograd.flatten(), '.')
        plt.show()

    npt.assert_array_almost_equal(dAEP_autograd, dAEP_cs, 15)
    npt.assert_array_almost_equal(dAEP_autograd, dAEP_fd, 6)
    npt.assert_array_equal(dAEP_autograd, wfm.daep(gradient_method=autograd, wrt_arg=wrt_arg, **kwargs))


@pytest.mark.parametrize('wrt_arg', ['x',
                                     #'y', 'xy', 'h',
                                     #'wd', 'ws'
                                     #'yaw'
                                     ])
def test_dAEP_parallel(wrt_arg):
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, wt, deflectionModel=JimenezWakeDeflection())
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wt.hub_height(), 'wd': None, 'ws': [10]}
    dAEP_ref, t_seq = timeit(lambda: wfm.daep(gradient_method=autograd, wrt_arg=wrt_arg, **kwargs))()
    # import pickle
    # import io
    #
    # def get_pickle_fail(obj, ignore_lst=[]):
    #     try:
    #         ignore_lst.append(obj)
    #         pickle.dump(obj, io.BytesIO())
    #     except Exception:
    #         print("failed", obj)
    #
    #         if hasattr(obj, '__dict__'):
    #             if hasattr(obj, '__getstate__'):
    #                 items = obj.__getstate__().items()
    #             else:
    #                 items = obj.__dict__.items()
    #             for k, v in items:
    #                 if not isinstance(v, (np.ndarray, list, tuple)) and v not in ignore_lst:
    #                     print("check", k)
    #                     get_pickle_fail(v, ignore_lst)
    #                     print("ok", k)
    #         print("Return", obj)
    # get_pickle_fail(wfm)

    dAEP_autograd, t_par = timeit(lambda: wfm.daep(gradient_method=fd, wrt_arg=wrt_arg, n_cpu=None, **kwargs),
                                  min_runs=2)()

    npt.assert_array_almost_equal(dAEP_ref, dAEP_autograd, 8)

    print(t_par, t_seq)
    # assert min(t_par)<min(t_seq) # not faster on 4 CPU laptop
