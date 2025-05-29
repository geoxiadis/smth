from py_wake import np
from py_wake.wind_turbines import WindTurbine
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
from py_wake.utils.rans_lut_utils import convert_ranslut, merge_luts


def test_rans_lut_deficit():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite([1, 0, 0, 0], ti=0.075 * 0.8)
    deficit = RANSLUTDeficit()
    wfm = All2AllIterative(site,
                           wts,
                           wake_deficitModel=deficit,
                           blockage_deficitModel=deficit,
                           )

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])
    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.97278258, 9.93703762, 7.66620186, 9.9943525, 9.49814004,
                                                                7.67200834, 6.95684497])
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.79338104, 0.79388147, 0.8056662, 0.79307906, 0.80002604,
                                                            0.80567201, 0.80495685])
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
    site = UniformSite([1, 0, 0, 0], ti=0.075 * 0.8)
    zRef = wts.hub_height()
    Dref = wts.diameter()
    dataset = convert_ranslut(Dref=Dref, zRef=zRef, rename=False, nondim=True)
    lutname = 'rans_lut.nc'
    dataset.to_netcdf(lutname)
    wfm = RANSLUT(lutname, site, wts, zRef, Dref, power_as_RANS=False)

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])
    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.97278258, 9.93703762, 7.66620186, 9.9943525, 9.49814004,
                                                                7.67200834, 6.95684497])
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.79338104, 0.79388147, 0.8056662, 0.79307906, 0.80002604,
                                                            0.80567201, 0.80495685])
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
    dataset = convert_ranslut(Dref=Dref, zRef=zRef, rename=False, nondim=True, cover_rotor_area=True)
    dataset.to_netcdf(lutname)
    wfm = RANSLUT(lutname, site, wts, zRef, Dref, power_as_RANS=True, ADcontrolfiles_write=True, ws_cutin=[4.0], ws_cutout=[25.0], dws=1.0, Prated=[2e6])
    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    # print(simres.WS_eff_star_ilk.flatten())
    # print(simres.ct_star_ilk.flatten())
    # print(simres.power_ilk.flatten() * 1e-6)
    # print(simres.TI_eff_ilk.flatten())
    UAD_expected = [7.67779686, 7.648784, 6.01979316, 7.69558068, 6.84114756, 6.02452841, 5.47620317]
    CTstar_expected = [1.34750391, 1.35007474, 1.38495081, 1.34480971, 1.38797481, 1.38496821, 1.38295341]
    power_expected = [1.34492562, 1.33278619, 0.66709526, 1.3510846, 0.98660253, 0.66873156, 0.49696202]
    TI_eff_expected = [0.06002419, 0.06004405, 0.14076398, 0.06000223, 0.08990774, 0.14064143, 0.1891118]
    npt.assert_array_almost_equal(simres.power_ilk.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(simres.WS_eff_star_ilk.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(simres.ct_star_ilk.flatten(), CTstar_expected, 6)
    npt.assert_array_almost_equal(simres.TI_eff_ilk.flatten(), TI_eff_expected, 6)

    # Test with provided control file
    ADcontrolfile = 'lutcal_WT80_Ti0.06_0.dat'
    wfm = RANSLUT(lutname, site, wts, zRef, Dref, power_as_RANS=True, ADcontrolfiles=[ADcontrolfile], ws_cutin=[4.0], ws_cutout=[25.0], dws=1.0, Prated=[2e6], flowmap_maxpoints=100)
    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])
    npt.assert_array_almost_equal(simres.power_ilk.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(simres.WS_eff_star_ilk.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(simres.ct_star_ilk.flatten(), CTstar_expected, 6)
    npt.assert_array_almost_equal(simres.TI_eff_ilk.flatten(), TI_eff_expected, 6)

    # Test multi lut (using the same file for now)
    lut2 = xr.load_dataset(lutname)
    lut2.attrs['name'] = 'V80b'
    lutname2 = 'rans_lut2.nc'
    lut2.to_netcdf(lutname2)
    lutname_merged = 'rans_lut_merged.nc'
    merge_luts([lutname, lutname2], outfile=lutname_merged)
    wts = WindTurbine.from_WindTurbines([wts, wts])
    wfm = RANSLUT(lutname_merged, site, wts, zRef, None, power_as_RANS=True, ws_cutin=[4.0] * 2, ws_cutout=[25.0] * 2, dws=1.0, Prated=[2e6] * 2, flowmap_maxpoints=None)
    type_i = np.array([0, 1, 0, 0, 1, 1, 0])
    simres = wfm(wt_x, wt_y, type=type_i, wd=[30.0], ws=[10.0])
    npt.assert_array_almost_equal(simres.power_ilk.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(simres.WS_eff_star_ilk.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(simres.ct_star_ilk.flatten(), CTstar_expected, 6)
    npt.assert_array_almost_equal(simres.TI_eff_ilk.flatten(), TI_eff_expected, 6)
