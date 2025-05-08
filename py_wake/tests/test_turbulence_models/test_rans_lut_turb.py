from py_wake import np
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.examples.data import hornsrev1
import matplotlib.pyplot as plt
from py_wake.deficit_models.rans_lut import RANSLUTDeficit
from py_wake.turbulence_models.rans_lut_turb import RANSLUTTurbulence
from py_wake.flow_map import HorizontalGrid
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.wind_farm_models.engineering_models import All2AllIterative
import pytest
from py_wake.utils.profiling import timeit
import xarray as xr


def test_rans_lut_turb():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    ti_upstream = 0.075
    wake_added_ti = 0.05
    site = UniformSite([1, 0, 0, 0], ti=ti_upstream)
    deficit = RANSLUTDeficit()
    turb = RANSLUTTurbulence()
    wfm = All2AllIterative(site,
                           wts,
                           wake_deficitModel=deficit,
                           blockage_deficitModel=deficit,
                           turbulenceModel=turb,
                           )

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.98181609, 9.95358693, 8.45426639, 9.98181609, 9.35814773,
                                                                8.45426639, 7.97008201], 8)
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.79325457, 0.79364978, 0.80645427, 0.79325457, 0.80198593,
                                                            0.80645427, 0.80597008], 8)
    npt.assert_array_almost_equal(simres.TI_eff_ilk.flatten(), [0.075, 0.075, 0.05951451, 0.075, 0.06872367,
                                                                0.05951451, 0.0536696], 8)

    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)
    flow_map70 = simres.flow_map(HorizontalGrid(x_j, y_j, h=70))
    flow_map73 = simres.flow_map(HorizontalGrid(x_j, y_j, h=73))

    X, Y = flow_map70.XY
    Z70 = flow_map70.WS_eff_xylk[:, :, 0, 0]
    Z73 = flow_map73.WS_eff_xylk[:, :, 0, 0]

    if 0:
        flow_map70.plot_wake_map(levels=np.arange(6, 10.5, .1))
        plt.plot(X[0], Y[140])
        plt.figure()
        plt.plot(X[0], Z70[140, :], label="Z=70m")
        plt.plot(X[0], Z73[140, :], label="Z=73m")
        plt.plot(X[0, 100:400:10], Z70[140, 100:400:10], '.')
        print(list(np.round(Z70.data[140, 100:400:10], 4)))
        print(list(np.round(Z73.data[140, 100:400:10], 4)))
        plt.legend()
        plt.show()
