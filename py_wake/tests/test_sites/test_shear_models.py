from py_wake.site.shear import PowerShear, MOSTShear, LogShear, LLJRiemer
from py_wake.utils.most import psi
from py_wake import np
from py_wake.tests import npt
from py_wake.site._site import UniformSite
import matplotlib.pyplot as plt
from numpy import newaxis as na
import xarray as xr


def test_power_shear():
    h_lst = np.arange(10, 100, 10)
    site = UniformSite([1], .1, shear=PowerShear(70, alpha=[.1, .2]))
    WS = site.local_wind(x=h_lst * 0, y=h_lst * 0, h=h_lst, wd=[0, 180], ws=[10, 12, 13]).WS

    if 0:
        plt.plot(WS.sel(wd=0, ws=10), h_lst, label='alpha=0.1')
        plt.plot((h_lst / 70)**0.1 * 10, h_lst, ':')
        plt.plot(WS.sel(wd=180, ws=12), h_lst, label='alpha=0.2')
        plt.plot((h_lst / 70)**0.2 * 12, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_equal(WS.sel(wd=0, ws=10), (h_lst / 70)**0.1 * 10)
    npt.assert_array_equal(WS.sel(wd=180, ws=12), (h_lst / 70)**0.2 * 12)


def test_most_shear():
    h_lst = np.arange(10, 100, 10)
    h_zetas = [-0.5, 0.0, 0.5]
    for h_zeta in h_zetas:
        L_inv = h_zeta / 70.0
        site = UniformSite([1], .1, shear=MOSTShear(70, h_zeta=h_zeta, z0=[.02, 2], Cm1=5.0, Cm2=-16.0))
        WS = site.local_wind(x=h_lst * 0, y=h_lst * 0, h=h_lst, wd=[0, 180], ws=[10, 12]).WS

        if 0:
            plt.plot(WS.sel(wd=0, ws=10), h_lst, label='z0=0.02, h_zeta=%g' % h_zeta)
            plt.plot((np.log(h_lst / 0.02) - psi(h_lst * L_inv, Cm1=5.0, Cm2=-16.0)) / (np.log(70.0 / 0.02) - psi(h_zeta, Cm1=5.0, Cm2=-16.0)) * 10.0, h_lst, ':')
            plt.plot(WS.sel(wd=180, ws=12), h_lst, label='z0=2, h_zeta=%g' % h_zeta)
            plt.plot((np.log(h_lst / 2) - psi(h_lst * L_inv, Cm1=5.0, Cm2=-16.0)) / (np.log(70.0 / 2) - psi(h_zeta, Cm1=5.0, Cm2=-16.0)) * 12.0, h_lst, ':')
            if h_zeta == h_zetas[-1]:
                plt.legend()
                plt.show()
        npt.assert_array_equal(WS.sel(wd=0, ws=10), (np.log(h_lst / 0.02) - psi(h_lst * L_inv, Cm1=5.0, Cm2=-16.0)) / (np.log(70.0 / 0.02) - psi(h_zeta, Cm1=5.0, Cm2=-16.0)) * 10)
        npt.assert_array_equal(WS.sel(wd=180, ws=12), (np.log(h_lst / 2) - psi(h_lst * L_inv, Cm1=5.0, Cm2=-16.0)) / (np.log(70.0 / 2) - psi(h_zeta, Cm1=5.0, Cm2=-16.0)) * 12)


def test_log_shear():
    h_lst = np.arange(10, 100, 10)
    site = UniformSite([1], .1, shear=LogShear(70, z0=[.02, 2]))
    WS = site.local_wind(x=h_lst * 0, y=h_lst * 0, h=h_lst, wd=[0, 180], ws=[10, 12]).WS

    if 0:
        plt.plot(WS.sel(wd=0, ws=10), h_lst, label='z0=0.02')
        plt.plot(np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10, h_lst, ':')
        plt.plot(WS.sel(wd=180, ws=12), h_lst, label='z0=2')
        plt.plot(np.log(h_lst / 2) / np.log(70 / 2) * 12, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_equal(WS.sel(wd=0, ws=10), np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10)
    npt.assert_array_equal(WS.sel(wd=180, ws=12), np.log(h_lst / 2) / np.log(70 / 2) * 12)


def test_log_shear_constant_z0():
    h_lst = np.arange(10, 100, 10)
    site = UniformSite([1], .1, shear=LogShear(70, z0=.02))
    WS = site.local_wind(x=h_lst * 0, y=h_lst * 0, h=h_lst, wd=[0, 180], ws=[10, 12, 13]).WS

    if 0:
        plt.plot(WS.sel(ws=10), WS.h, label='z0=0.02')
        plt.plot(np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_equal(WS.sel(ws=10), np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10)


def test_custom_shear():
    def my_shear(localWind, WS, h):
        return WS * (0.02 * (h[:, na, na] - 70) + 1) * (localWind.wd[na, :, na] * 0 + 1)
    h_lst = np.arange(10, 100, 10)

    site = UniformSite([1], .1, shear=my_shear)
    WS = site.local_wind(x=h_lst * 0, y=h_lst * 0, h=h_lst, wd=[0, 180], ws=[10, 12]).WS

    if 0:
        plt.plot(WS.sel(wd=0, ws=10), WS.h, label='2z-2')
        plt.plot((h_lst - 70) * 0.2 + 10, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(WS.sel(wd=0, ws=10), (h_lst - 70) * 0.2 + 10)


def test_llj_riemer_abs():
    # Test a Low Level Jet, according to Riemer model.
    # The strength parameter is absolute.
    site = UniformSite(
        [1], 0.1, shear=LLJRiemer(strength=1.5, width=100.0, h_ref=300.0)
    )
    h_lst = np.arange(0.0, 600.1, 50.0)
    WS_actual = site.local_wind(
        x=np.zeros_like(h_lst),
        y=np.zeros_like(h_lst),
        h=h_lst,
        wd=[0, 180],
        ws=[5.0, 15.0],
    ).WS
    WS_desired = xr.DataArray(
        data=np.array(
            [
                [1.85114706e-04, 1.85114706e-04],
                [2.89568120e-03, 2.89568120e-03],
                [2.74734583e-02, 2.74734583e-02],
                [1.58098837e-01, 1.58098837e-01],
                [5.51819162e-01, 5.51819162e-01],
                [1.16820117e00, 1.16820117e00],
                [1.50000000e00, 1.50000000e00],
                [1.16820117e00, 1.16820117e00],
                [5.51819162e-01, 5.51819162e-01],
                [1.58098837e-01, 1.58098837e-01],
                [2.74734583e-02, 2.74734583e-02],
                [2.89568120e-03, 2.89568120e-03],
                [1.85114706e-04, 1.85114706e-04],
            ]
        ),
        coords={"i": np.arange(h_lst.size), "ws": [5.0, 15.0]},
    )
    xr.testing.assert_allclose(WS_actual, WS_desired)

    if 0:
        fig, ax = plt.subplots()
        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("Height [m]")
        ax.grid(True)
        plt.plot(WS_actual.sel(ws=5.0), h_lst)


def test_llj_riemer_rel():
    # Test a Low Level Jet, according to Riemer model.
    # The strength parameter is relative.
    site = UniformSite(
        [1],
        0.1,
        shear=LLJRiemer(strength=1.5, width=100.0, h_ref=300.0, wsp_ref=6.0),
    )
    h_lst = np.arange(0.0, 600.1, 50.0)
    WS_actual = site.local_wind(
        x=np.zeros_like(h_lst),
        y=np.zeros_like(h_lst),
        h=h_lst,
        wd=[0, 180],
        ws=[5.0, 15.0],
    ).WS
    WS_desired = xr.DataArray(
        data=np.array(
            [
                [1.11068824e-03, 1.11068824e-03],
                [1.73740872e-02, 1.73740872e-02],
                [1.64840750e-01, 1.64840750e-01],
                [9.48593021e-01, 9.48593021e-01],
                [3.31091497e00, 3.31091497e00],
                [7.00920705e00, 7.00920705e00],
                [9.00000000e00, 9.00000000e00],
                [7.00920705e00, 7.00920705e00],
                [3.31091497e00, 3.31091497e00],
                [9.48593021e-01, 9.48593021e-01],
                [1.64840750e-01, 1.64840750e-01],
                [1.73740872e-02, 1.73740872e-02],
                [1.11068824e-03, 1.11068824e-03],
            ]
        ),
        coords={"i": np.arange(h_lst.size), "ws": [5.0, 15.0]},
    )
    xr.testing.assert_allclose(WS_actual, WS_desired)

    if 0:
        fig, ax = plt.subplots()
        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("Height [m]")
        ax.grid(True)
        plt.plot(WS_actual.sel(ws=5.0), h_lst)


def test_power_shear_and_llj_riemer():
    hub_height = 150.0
    power_shear = PowerShear(hub_height, alpha=0.2)
    llj = LLJRiemer(strength=1.5, width=50.0, h_ref=300.0)

    site_power = UniformSite([1], 0.1, shear=power_shear)
    site_llj = UniformSite([1], 0.1, shear=llj)
    site_power_llj = UniformSite([1], 0.1, shear=(power_shear, llj))

    wd_ambient = [0, 180]
    ws_ambient = [5.0, 9.0, 15.0]
    h_lst = np.arange(10, 500.1, 5)
    x = np.zeros_like(h_lst)
    y = np.zeros_like(h_lst)
    WS_power = site_power.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=wd_ambient,
        ws=ws_ambient,
    ).WS
    WS_llj = site_llj.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=wd_ambient,
        ws=ws_ambient,
    ).WS
    WS_power_llj = site_power_llj.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=wd_ambient,
        ws=ws_ambient,
    ).WS

    # The power shear and LLJ must be added to each other.
    npt.assert_allclose(WS_power_llj, WS_power + WS_llj)
    # The LLJ reference height should not be confused with the hub one.
    # We placed the LLJ rather far from the hub.
    i_hub = np.searchsorted(h_lst, hub_height)
    npt.assert_allclose(WS_power_llj[i_hub, :].data, ws_ambient, atol=5e-4)

    if 0:
        fig, ax = plt.subplots()
        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("Height [m]")
        ax.grid(True)
        plt.plot(WS_power.sel(ws=9.0), h_lst, label="Power shear")
        plt.plot(WS_llj.sel(ws=9.0), h_lst, label="LLJ")
        plt.plot(WS_power_llj.sel(ws=9.0), h_lst, label="Power shear + LLJ")
        plt.legend()


def test_llj_positive_and_negative():
    # Reproduce Fig. 1 (a) from:
    #   Hallgren, C., Aird, J. A., Ivanell, S., Körnich, H., Barthelmie, R. J., Pryor, S. C., and Sahlée, E.:
    #   Brief communication: On the definition of the low-level jet, Wind Energ. Sci., 8, 1651–1658, 2023
    #   https://doi.org/10.5194/wes-8-1651-2023
    log_shear = LogShear(h_ref=150.0, z0=.06)
    llj_positive = LLJRiemer(strength=1.0, width=50.0, h_ref=150.0)
    llj_negative = LLJRiemer(strength=-0.5, width=75.0, h_ref=300.0)
    site_log = UniformSite([1], 0.1, shear=log_shear)
    site_llj_positive = UniformSite([1], 0.1, shear=llj_positive)
    site_llj_negative = UniformSite([1], 0.1, shear=llj_negative)
    site_only_llj = UniformSite([1], 0.1, shear=(llj_positive, llj_negative))
    site_all = UniformSite([1], 0.1,
                           shear=(log_shear, llj_positive, llj_negative))
    h_lst = np.arange(10, 500.1, 5)
    x = np.zeros_like(h_lst)
    y = np.zeros_like(h_lst)
    ws_log = site_log.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=6.0,
    ).WS
    ws_llj_positive = site_llj_positive.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=6.0,
    ).WS
    ws_llj_negative = site_llj_negative.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=6.0,
    ).WS
    ws_all = site_all.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=6.0,
    ).WS
    # Check that a site with only absolute shears does not crash.
    site_only_llj.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=6.0,
    )
    # The order of the shears must not matter.
    npt.assert_allclose(ws_all,
                        ws_llj_negative + ws_log + ws_llj_positive)
    # Min and max.
    npt.assert_allclose(ws_all[h_lst > 225.0].min(), 6.01, atol=5e-3)
    npt.assert_allclose(ws_all[h_lst < 225.0].max(), 7.00, atol=5e-3)

    if 0:
        fig, ax = plt.subplots()
        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("Height [m]")
        ax.grid(True)
        plt.plot(ws_log.data, h_lst, label="Log")
        plt.plot(ws_all.data, h_lst, label="Log + LLJ")
        plt.legend()


def test_power_and_log_shears():
    hub_height = 150.0
    log_shear = LogShear(h_ref=hub_height, z0=.05)
    power_shear = PowerShear(h_ref=hub_height, alpha=-0.1)
    site_log = UniformSite([1], 0.1, shear=log_shear)
    site_power = UniformSite([1], 0.1, shear=power_shear)
    site_all = UniformSite([1], 0.1, shear=(power_shear, log_shear))
    h_lst = np.arange(10, 500.1, 5)
    x = np.zeros_like(h_lst)
    y = np.zeros_like(h_lst)
    ws_ambient = 4.0
    ws_log = site_log.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=ws_ambient,
    ).WS
    ws_power = site_power.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=ws_ambient,
    ).WS
    ws_all = site_all.local_wind(
        x=x,
        y=y,
        h=h_lst,
        wd=0.0,
        ws=ws_ambient,
    ).WS
    # Since these shears are relative, the hub height wind speed must be the specified one.
    i_hub = np.searchsorted(h_lst, hub_height)
    npt.assert_allclose(ws_all[i_hub].data, ws_ambient)

    if 0:
        fig, ax = plt.subplots()
        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("Height [m]")
        ax.grid(True)
        plt.plot(ws_log.data, h_lst, label="Log")
        plt.plot(ws_power.data, h_lst, label="Power")
        plt.plot(ws_all.data, h_lst, label="Log * Power")
        plt.legend()


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
