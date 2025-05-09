from py_wake import np
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.examples.data import hornsrev1
import matplotlib.pyplot as plt
from py_wake.deficit_models.rans_lut import RANSLUTDeficit, RANSLUT
from py_wake.flow_map import HorizontalGrid
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.wind_farm_models.engineering_models import All2AllIterative
import pytest
from py_wake.utils.profiling import timeit
import xarray as xr
from py_wake.utils.rans_lut_utils import fugalut2ranslut


def test_rans_lut_deficit():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    deficit = RANSLUTDeficit()
    wfm = All2AllIterative(site,
                           wts,
                           wake_deficitModel=deficit,
                           blockage_deficitModel=deficit,
                           )

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.98181609, 9.95358693, 8.45426639, 9.98181609, 9.35814773,
                                                                8.45426639, 7.97008201], 8)
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.79325457, 0.79364978, 0.80645427, 0.79325457, 0.80198593,
                                                            0.80645427, 0.80597008], 8)

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


def test_rans_lut():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    zRef = wts.hub_height()
    Dref = wts.diameter()
    lut, da_ti = fugalut2ranslut(Dref=Dref, zRef=zRef, rename=False, nondim=True)
    dataset = xr.Dataset(data_vars={'deficits': lut, 'added_ti': da_ti})
    lutname = 'rans_lut.nc'
    dataset.to_netcdf(lutname)
    wfm = RANSLUT(lutname, site, wts, zRef, Dref, power_as_RANS=False)

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.98181609, 9.95358693, 8.45426639, 9.98181609, 9.35814773,
                                                                8.45426639, 7.97008201], 8)
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.79325457, 0.79364978, 0.80645427, 0.79325457, 0.80198593,
                                                            0.80645427, 0.80597008], 8)

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

    # Test Power as RANS post step
    lut, da_ti = fugalut2ranslut(Dref=Dref, zRef=zRef, rename=False, nondim=True, cover_rotor_area=True)
    dataset = xr.Dataset(data_vars={'deficits': lut, 'added_ti': da_ti})
    dataset.to_netcdf(lutname)
    wfm = RANSLUT(lutname, site, wts, zRef, Dref, power_as_RANS=True, ws_cutin=[4.0], ws_cutout=[25.0], dws=1.0, Prated=[2e6])
    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    # print(simres.WS_eff_ilk.flatten())
    # print(simres.ct_ilk.flatten())
    # print(simres.power_ilk.flatten() * 1e-6)
    # print(simres.TI_eff_ilk.flatten())
    npt.assert_array_almost_equal(simres.power_ilk.flatten() * 1e-6, [1.34972737, 1.34155641, 0.88731729, 1.34972737, 1.09624206,
                                                                      0.88731729, 0.73970888], 8)
    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [8.49496922, 8.4703886, 7.31576317, 8.49496922, 7.86743257,
                                                                7.31576317, 6.89186358], 8)
    # CTstar
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [1.10297574, 1.10552849, 1.13160315, 1.10297574, 1.12403335,
                                                            1.13160315, 1.13064446], 8)
    npt.assert_array_almost_equal(simres.TI_eff_ilk.flatten(), [0.075, 0.075, 0.05970602, 0.075, 0.06875403,
                                                                0.05970602, 0.05390452], 8)
