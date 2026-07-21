import matplotlib.pyplot as plt
import xarray as xr

from py_wake import np
from py_wake.deficit_models.rans_lut import RANSLUT, RANSLUTDemoDeficit
from py_wake.examples.data import hornsrev1
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.flow_map import HorizontalGrid
from py_wake.site._site import UniformSite
from py_wake.tests import npt, ptf
from py_wake.turbulence_models.rans_lut_turb import RANSLUTDemoTurbulence
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils.profiling import timeit
from py_wake.utils.rans_lut_utils import ADControl, get_Ellipsys_equivalent_output, fit_gauss
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines._wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.rotor_avg_models.rotor_avg_model import GQGridRotorAvg


def test_rans_lut_deficit():
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite([1, 0, 0, 0], ti=0.075 * 0.8)
    deficit = RANSLUTDemoDeficit()
    wfm = All2AllIterative(site, wts, wake_deficitModel=deficit, blockage_deficitModel=deficit,
                           turbulenceModel=RANSLUTDemoTurbulence())

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])
    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.97278258, 9.93703762, 7.66620186, 9.9943525, 9.49814004,
                                                                7.67200834, 6.95684497])
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.79338104, 0.79388147, 0.8056662, 0.79307906, 0.80002604,
                                                            0.80567201, 0.80495685])
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    if 0:
        flow_map70 = simres.flow_map(HorizontalGrid(x_j, y_j, h=70))
        flow_map73 = simres.flow_map(HorizontalGrid(x_j, y_j, h=73))

        X, Y = flow_map70.XY
        Z70 = flow_map70.WS_eff_xylk[:, :, 0, 0]
        Z73 = flow_map73.WS_eff_xylk[:, :, 0, 0]

        flow_map70.plot_wake_map(levels=np.arange(6, 10.5, .1))
        plt.plot(X[0], Y[140])
        plt.figure()
        plt.plot(X[0], Z70[140, :], label="Z=70m")
        plt.plot(X[0], Z73[140, :], label="Z=73m")
        plt.plot(X[0, 100:400:10], Z70[140, 100:400:10], '.')
        print(np.round(Z70.data[140, 100:400:10], 4).tolist())
        print(np.round(Z73.data[140, 100:400:10], 4).tolist())
        plt.legend()
        plt.show()

    flow_map70 = simres.flow_map(HorizontalGrid(x_j[100:400:10], y_j[[140]], h=70))
    flow_map73 = simres.flow_map(HorizontalGrid(x_j[100:400:10], y_j[[140]], h=73))

    X, Y = flow_map70.XY
    Z70 = flow_map70.WS_eff_xylk[0, :, 0, 0]
    Z73 = flow_map73.WS_eff_xylk[0, :, 0, 0]

    npt.assert_array_almost_equal(
        Z70,
        [10.0199, 10.0235, 10.0293, 10.0358, 9.8078, 5.5094, 4.718, 9.4776, 10.034, 10.0113, 10.0116, 10.027, 10.0617,
         9.4429, 5.6, 8.8807, 10.0984, 10.0073, 9.9856, 9.96, 9.0589, 7.4311, 3.7803, 6.5824, 10.1399, 10.0271,
         9.9978, 9.9919, 9.9907, 9.9906], 4)

    npt.assert_array_almost_equal(
        Z73,
        [10.0198, 10.0234, 10.0291, 10.0351, 9.8118, 5.6347, 4.7181, 9.4902, 10.0332, 10.0113, 10.0116, 10.0267,
         10.0607, 9.4732, 5.5569, 8.9562, 10.0969, 10.0073, 9.9858, 9.9601, 9.0821, 7.4662, 3.8426, 6.6404, 10.1375,
         10.0268, 9.9979, 9.992, 9.9908, 9.9906], 4)


def test_rans_lut():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite([1, 0, 0, 0], ti=0.075 * 0.8)
    Dref = wts.diameter()
    demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                   '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')
    wfm = RANSLUT(demo_lut, site, wts)

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])
    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.97278258, 9.93703762, 7.66620186, 9.9943525, 9.49814004,
                                                                7.67200834, 6.95684497])
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.79338104, 0.79388147, 0.8056662, 0.79307906, 0.80002604,
                                                            0.80567201, 0.80495685])

    # Test Power as RANS post step
    dataset = xr.open_dataset(demo_lut)
    lutname = 'rans_lut.nc'
    dataset.to_netcdf(lutname)
    aDControl = ADControl.from_lut(dataset.deficits, wts, ws_cutin=4, ws_cutout=25, dws=1.0, cal_TI=0.06)
    aDControl.save()
    ADcontrolfile = 'lutcal_WT80_Ti0.06_0.dat'

    aDControl_saved = ADControl.from_files([ADcontrolfile])

    npt.assert_array_almost_equal(aDControl.U_CT_CP_AD_ws, aDControl_saved.U_CT_CP_AD_ws, 10)

    wfm = RANSLUT(dataset, site, wts)
    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    # print(simres.WS_eff_star_ilk.flatten())
    # print(simres.ct_star_ilk.flatten())
    # print(simres.power_ilk.flatten() * 1e-6)
    # print(simres.TI_eff_ilk.flatten())
    UAD_expected = [7.731823, 7.702646, 6.23615, 7.749774, 7.072877, 6.241342, 5.746075]
    CTstar_expected = [1.328699, 1.331197, 1.365904, 1.326045, 1.360755, 1.365923, 1.364156]
    power_expected = [1.344857, 1.332689, 0.727579, 1.351047, 1.060075, 0.729429, 0.564433]
    TI_eff_expected = [0.060025, 0.060044, 0.141482, 0.060005, 0.090368, 0.141364, 0.190107]
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(simres, aDControl)
    npt.assert_array_almost_equal(ellipsys_power.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(WS_eff_star.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(ct_star.flatten(), CTstar_expected, 6)
    npt.assert_array_almost_equal(simres.TI_eff.values.flatten(), TI_eff_expected, 6)

    # Test with provided control file
    aDControl.save()
    ADcontrolfile = 'lutcal_WT80_Ti0.06_0.dat'
    aDControl_saved = ADControl.from_files([ADcontrolfile])
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(
        simres, aDControl_saved, flowmap_maxpoints=100)
    npt.assert_array_almost_equal(ellipsys_power.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(WS_eff_star.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(ct_star.flatten(), CTstar_expected, 6)
    npt.assert_array_almost_equal(simres.TI_eff.values.flatten(), TI_eff_expected, 6)

    # Test multi lut (using the same file for now)
    lut2 = xr.load_dataset(lutname)
    lut2.attrs['name'] = 'V80b'
    lutname2 = 'rans_lut2.nc'
    lut2.to_netcdf(lutname2)
    wts = WindTurbines.from_WindTurbine_lst([wts, wts])
    wfm = RANSLUT([dataset, lut2], site, wts)
    type_i = np.array([0, 1, 0, 0, 1, 1, 0])
    simres = wfm(wt_x, wt_y, type=type_i, wd=[30.0], ws=[10.0])
    aDControl2 = ADControl.from_lut([dataset, lut2], wts, ws_cutin=4, ws_cutout=25, dws=1.0, cal_TI=0.06)
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(simres, aDControl2)

    npt.assert_array_almost_equal(ellipsys_power.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(WS_eff_star.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(ct_star.flatten(), CTstar_expected, 6)
    npt.assert_array_almost_equal(simres.TI_eff.values.flatten(), TI_eff_expected, 6)

    # Test MOST shear, stable.
    # WS_eff_star does not change since we not change the actual simulation
    # using Site with MOSTShear and LUTs with stability
    aDControl2 = ADControl.from_lut([dataset, lut2], wts, ws_cutin=4, ws_cutout=25, dws=1.0, cal_TI=0.06, cal_zeta=0.5)
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(simres, aDControl2)

    # print(ct_star.flatten())
    # print(ellipsys_power.flatten() * 1e-6)
    CTstar_expected = [1.32895893, 1.33159161, 1.36637959, 1.32630388, 1.36116655, 1.36639816, 1.36463016]
    power_expected = [1.34531419, 1.33329859, 0.72796236, 1.35150555, 1.060569, 0.72981311, 0.56473626]

    npt.assert_array_almost_equal(ellipsys_power.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(WS_eff_star.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(ct_star.flatten(), CTstar_expected, 6)

    # Test MOST shear, unstable
    aDControl2 = ADControl.from_lut([dataset, lut2], wts, ws_cutin=4, ws_cutout=25, dws=1.0, cal_TI=0.06, cal_zeta=-0.5)
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(simres, aDControl2)

    # print(ct_star.flatten())
    # print(ellipsys_power.flatten() * 1e-6)
    CTstar_expected = [1.3254684, 1.32682915, 1.36065434, 1.32318215, 1.35620019, 1.36067278, 1.35891729]
    power_expected = [1.33938202, 1.32595377, 0.72335257, 1.34597418, 1.05462423, 0.72519153, 0.56109148]

    npt.assert_array_almost_equal(ellipsys_power.flatten() * 1e-6, power_expected, 6)
    npt.assert_array_almost_equal(WS_eff_star.flatten(), UAD_expected, 6)
    npt.assert_array_almost_equal(ct_star.flatten(), CTstar_expected, 6)


def test_rans_lut_multi_wd_ws():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite(ti=0.075 * 0.8)
    Dref = wts.diameter()
    demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                   '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')
    wfm = RANSLUT(demo_lut, site, wts)

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30, 120, 210], ws=[8, 10])
    power_ref = [[[7.978253, 7.949314, 6.104552, 7.995857, 7.592563, 6.108074, 5.544956],
                  [9.97278258, 9.93703762, 7.66620186, 9.9943525, 9.49814004, 7.67200834, 6.95684497]],
                 [[6.86814519, 7.99753318, 6.87463393, 7.99468031, 7.97001202, 7.99709671, 7.96569652],
                  [8.60552359, 9.99679284, 8.61335717, 9.99285614, 9.96291775, 9.99607099, 9.95761711]],
                 [[6.10490992, 7.05898214, 7.98694797, 6.09979465, 6.07664782, 7.96281632, 7.9833782],
                  [7.66743939, 8.8346778, 9.98334522, 7.65915159, 7.63177922, 9.9536539, 9.97887183]]]

    ct_ref = [[[0.805978, 0.805949, 0.804105, 0.805996, 0.805593, 0.804108, 0.80491],
              [0.79338104, 0.79388147, 0.8056662, 0.79307906, 0.80002604, 0.80567201, 0.80495685]],
              [[0.80486814, 0.80599753, 0.80487463, 0.80599468, 0.80597001, 0.8059971, 0.8059657],
               [0.80660552, 0.7930449, 0.80661336, 0.79310001, 0.79351915, 0.79305501, 0.79359336]],
              [[0.80410491, 0.80505898, 0.80598695, 0.80409979, 0.80407665, 0.80596282, 0.80598338],
               [0.80566744, 0.80683468, 0.79323317, 0.80565915, 0.80563178, 0.79364885, 0.79329579]]]

    npt.assert_array_almost_equal(simres.WS_eff_ilk, np.moveaxis(power_ref, -1, 0))
    npt.assert_array_almost_equal(simres.ct_ilk, np.moveaxis(ct_ref, -1, 0))

    # Test Power as RANS post step
    dataset = xr.open_dataset(demo_lut)
    lutname = 'rans_lut.nc'
    dataset.to_netcdf(lutname)
    aDControl = ADControl.from_lut(dataset.deficits, wts, ws_cutin=4, ws_cutout=25, dws=1.0, cal_TI=0.06)

    ADcontrolfile = 'lutcal_WT80_Ti0.06_0.dat'
    aDControl_saved = ADControl.from_files([ADcontrolfile])

    npt.assert_array_almost_equal(aDControl.U_CT_CP_AD_ws, aDControl_saved.U_CT_CP_AD_ws, 10)

    wfm = RANSLUT(dataset, site, wts)
    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30, 120, 210], ws=[8, 10])

    UAD_expected = [[[6.156, 6.134, 4.973, 6.17, 5.638, 4.976, 4.583],
                     [7.732, 7.703, 6.236, 7.75, 7.073, 6.241, 5.746]],
                    [[5.4, 6.171, 5.405, 6.169, 6.15, 6.171, 6.146],
                     [6.76, 7.752, 6.766, 7.748, 7.724, 7.751, 7.719]],
                    [[4.974, 5.337, 6.163, 4.969, 4.952, 6.144, 6.16],
                     [6.237, 6.672, 7.741, 6.23, 6.21, 7.716, 7.737]]]
    CTstar_expected = [[[1.366, 1.366, 1.361, 1.366, 1.364, 1.361, 1.36],
                        [1.329, 1.331, 1.366, 1.326, 1.361, 1.366, 1.364]],
                       [[1.363, 1.366, 1.363, 1.366, 1.366, 1.366, 1.366],
                        [1.368, 1.326, 1.368, 1.326, 1.33, 1.326, 1.33]],
                       [[1.361, 1.363, 1.366, 1.361, 1.361, 1.366, 1.366],
                        [1.366, 1.367, 1.327, 1.366, 1.366, 1.331, 1.328]]]
    power_expected = [[[0.699, 0.692, 0.358, 0.704, 0.532, 0.359, 0.276],
                       [1.345, 1.333, 0.728, 1.351, 1.06, 0.729, 0.564]],
                      [[0.465, 0.705, 0.467, 0.704, 0.697, 0.705, 0.696],
                       [0.931, 1.352, 0.933, 1.351, 1.342, 1.351, 1.34]],
                      [[0.358, 0.448, 0.702, 0.357, 0.353, 0.695, 0.701],
                       [0.728, 0.894, 1.348, 0.725, 0.718, 1.339, 1.347]]]
    TI_eff_expected = [[[0.06, 0.06, 0.143, 0.06, 0.091, 0.143, 0.192],
                        [0.06, 0.06, 0.141, 0.06, 0.09, 0.141, 0.19]],
                       [[0.124, 0.06, 0.124, 0.06, 0.06, 0.06, 0.06],
                        [0.123, 0.06, 0.123, 0.06, 0.06, 0.06, 0.06]],
                       [[0.143, 0.139, 0.06, 0.143, 0.143, 0.06, 0.06],
                        [0.141, 0.139, 0.06, 0.142, 0.141, 0.06, 0.06]]]
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(simres, aDControl)
    # print(np.round(np.moveaxis(WS_eff_star, 0, -1), 3).tolist())
    # print(np.round(np.moveaxis(ct_star, 0, -1), 3).tolist())
    # print(np.round(np.moveaxis(ellipsys_power * 1e-6, 0, -1), 3).tolist())
    # print(np.round(np.moveaxis(simres.TI_eff_ilk, 0, -1), 3).tolist())
    npt.assert_array_almost_equal(np.moveaxis(ellipsys_power * 1e-6, 0, -1), power_expected, 3)
    npt.assert_array_almost_equal(np.moveaxis(WS_eff_star, 0, -1), UAD_expected, 3)
    npt.assert_array_almost_equal(np.moveaxis(ct_star, 0, -1), CTstar_expected, 3)
    npt.assert_array_almost_equal(np.moveaxis(simres.TI_eff.values, 0, -1), TI_eff_expected, 3)

    wts.powerCtFunction.power_ct_tab[0, -1] *= .5  # set power at cutout to 50%
    aDControl = ADControl.from_lut([dataset.deficits], wts, ws_cutin=4, ws_cutout=25, dws=1.0, cal_TI=0.06)
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(simres, aDControl)
    npt.assert_array_almost_equal(np.moveaxis(ellipsys_power * 1e-6, 0, -1), power_expected, 3)


def test_RANSLUT_multiturbine():
    # Test with provided control file
    demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                   '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')
    ds = xr.load_dataset(demo_lut)

    # Test multi lut (using the same file for now with 50% wake deficit)
    lut_V120 = ds.copy(deep=True)
    lut_V120.deficits[:] *= .5

    v80 = HornsrevV80()
    v120 = WindTurbine('V120', 120, 70, powerCtFunction=PowerCtTabular(
        hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1], 'w', hornsrev1.ct_curve[:, 1]))
    wts = WindTurbines.from_WindTurbine_lst([v80, v120])

    wfm = RANSLUT([ds, lut_V120], UniformSite(ti=0.075 * 0.8), wts)
    type_i = np.array([0, 0, 1, 1])
    sim_res = wfm([0, 500, 1000, 1500], [0, 0, 0, 0], type=type_i, wd=[90, 270], ws=10.0)
    if 0:
        axes = plt.subplots(2, 1)[1]
        sim_res.flow_map(wd=90).plot_wake_map(ax=axes[0], levels=np.linspace(3, 10.5))
        sim_res.flow_map(wd=270).plot_wake_map(ax=axes[1], levels=np.linspace(3, 10.5))
        plt.show()

    aDControl = ADControl.from_lut([ds, lut_V120], wts, ws_cutin=[3, 4], ws_cutout=[26, 25], dws=1.0, cal_TI=0.06)
    ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(sim_res, aDControl)
    WS_eff_ref = [[5.565174, 6.064697, 7.628937, 8.856583], [7.737094, 6.21574, 6.83784, 6.59628]]
    ct_star_ref = [[1.363511, 1.365292, 1.049124, 1.020519], [1.32792, 1.365832, 1.047633, 1.047178]]
    power_ref = [[0.511008, 0.667681, 0.898337, 1.363178], [1.346674, 0.72034, 0.642516, 0.574577]]
    TI_eff_ref = [[0.27437, 0.215329, 0.141343, 0.060031], [0.060023, 0.141508, 0.200574, 0.244656]]
    npt.assert_array_almost_equal(WS_eff_star.squeeze().T, WS_eff_ref, 6)
    npt.assert_array_almost_equal(ct_star.squeeze().T, ct_star_ref, 6)
    npt.assert_array_almost_equal(ellipsys_power.squeeze().T * 1e-6, power_ref, 6)
    npt.assert_array_almost_equal(sim_res.TI_eff.values.squeeze().T, TI_eff_ref, 6)

    for type_i in [(0, 0, 0, 0), (1, 1, 1, 1)]:
        sim_res = wfm([0, 500, 1000, 1500], [0, 0, 0, 0], type=type_i, wd=[90, 270], ws=10.0)
        ellipsys_power, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(sim_res, aDControl)


def test_rans_conv_lut():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite([1, 0, 0, 0], ti=0.075 * 0.8)
    Dref = wts.diameter()
    demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                   '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')
    # Fit Gaussian deficits to hub height profiles for WeightedSum superposition
    demo_convlut = 'V80_gaussian.nc'
    fit_gauss(demo_lut, 'V80_gaussian.nc', ymin=-10, ymax=10)
    wfm = RANSLUT(demo_lut, site, wts, convlut=demo_convlut, rotorAvgModel=GQGridRotorAvg(4, 3))

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])
    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.97150681, 9.93711854, 7.97991956, 9.99272109, 9.20975831,
                                                                7.98196409, 7.46986422])
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.7933989, 0.79388034, 0.80597992, 0.7931019, 0.80406338,
                                                            0.80598196, 0.80546986])
    ds = xr.load_dataset(demo_lut)
    aDControl = ADControl.from_lut(ds.deficits, wts, ws_cutin=4, ws_cutout=25, dws=1.0, cal_TI=0.06)
    _, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(simres, aDControl, rotorAvgModel=None)
    npt.assert_array_almost_equal(WS_eff_star.flatten(), [7.69759095, 7.66977332, 6.1306985, 7.71475178, 7.08139117, 6.13226504, 5.7398877])
    npt.assert_array_almost_equal(ct_star.flatten(), [1.33143388, 1.33273949, 1.36552774, 1.33062844, 1.36035502, 1.36553333, 1.36413403])


def test_rans_conv_lut_multiturbine():
    # Test with provided control file
    demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                   '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')
    ds = xr.load_dataset(demo_lut)

    # Test multi lut (using the same file for now with 50% wake deficit)
    lut_V120 = ds.copy(deep=True)
    lut_V120.deficits[:] *= .5

    # Fit Gaussian deficits to hub height profiles for WeightedSum superposition
    demo_convlut = 'V80_gaussian.nc'
    fit_gauss(demo_lut, 'V80_gaussian.nc', ymin=-10, ymax=10, xfits=np.linspace(0.0, 100.0, 11))
    lut_V120.to_netcdf('V120.nc')
    fit_gauss('V120.nc', 'V120_gaussian.nc', ymin=-10, ymax=10, xfits=np.linspace(0.0, 100.0, 11))

    v80 = HornsrevV80()
    v120 = WindTurbine('V120', 120, 70, powerCtFunction=PowerCtTabular(
        hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1], 'w', hornsrev1.ct_curve[:, 1]))
    wts = WindTurbine.from_WindTurbines([v80, v120])

    wfm = RANSLUT([ds, lut_V120], UniformSite(ti=0.075 * 0.8), wts, convlut=['V80_gaussian.nc', 'V120_gaussian.nc'], rotorAvgModel=GQGridRotorAvg(4, 3))
    type_i = np.array([0, 0, 1, 1])
    sim_res = wfm([0, 500, 1000, 1500], [0, 0, 0, 0], type=type_i, wd=[90, 270], ws=10.0)
    if 0:
        axes = plt.subplots(2, 1)[1]
        sim_res.flow_map(wd=90).plot_wake_map(ax=axes[0], levels=np.linspace(3, 10.5))
        sim_res.flow_map(wd=270).plot_wake_map(ax=axes[1], levels=np.linspace(3, 10.5))
        plt.show()

    WS_eff_ref = [[7.3973466, 7.90998823, 8.56042535, 9.97638533], [9.97756182, 7.94966889, 7.78868653, 7.50030898]]
    ct_ref = [[0.80539735, 0.80590999, 0.80656043, 0.79333061], [0.79331413, 0.80594967, 0.80578869, 0.80550031]]
    power_ref = [[0.5537738, 0.67475722, 0.8641276, 1.33285294], [1.33325883, 0.68412186, 0.64613002, 0.57807292]]
    TI_eff_ref = [[0.26789829, 0.21534878, 0.14177617, 0.06003113], [0.0600239, 0.13786305, 0.18521198, 0.23600163]]
    npt.assert_array_almost_equal(sim_res.WS_eff_ilk.squeeze().T, WS_eff_ref, 6)
    npt.assert_array_almost_equal(sim_res.ct_ilk.squeeze().T, ct_ref, 6)
    npt.assert_array_almost_equal(sim_res.power_ilk.squeeze().T * 1e-6, power_ref, 6)
    npt.assert_array_almost_equal(sim_res.TI_eff.values.squeeze().T, TI_eff_ref, 6)

    aDControl = ADControl.from_lut([ds, lut_V120], wts, ws_cutin=[3, 4], ws_cutout=[26, 25], dws=1.0, cal_TI=0.06)
    _, WS_eff_star, ct_star = get_Ellipsys_equivalent_output(sim_res, aDControl, rotorAvgModel=None)
    WS_eff_ref = [[5.68432379, 6.07711624, 7.50664894, 8.76770315], [7.70248902, 6.10752006, 6.83079781, 6.57818957]]
    ct_star_ref = [[1.36393587, 1.36533666, 1.0488932, 1.0271518], [1.33120399, 1.36544508, 1.04761978, 1.04714416]]
    npt.assert_array_almost_equal(WS_eff_star.squeeze().T, WS_eff_ref, 6)
    npt.assert_array_almost_equal(ct_star.squeeze().T, ct_star_ref, 6)
