import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import gamma

from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.site import UniformSite
from py_wake.tests import npt
from py_wake.utils import fuga_utils
from py_wake.utils.denmark_utils import DKWindTurbines
from py_wake.utils.layouts import farm_area
from py_wake.wind_farm_models.minimalistic_wind_farm_model import (
    Larsen_etal2026,
    Sorensen_etal_2024,
    Sorensen_Larsen_2021,
)
from py_wake.wind_turbines.generic_wind_turbines import SimpleGenericWindTurbine


def test_Sorensen_Larsen_2021():
    wt = SimpleGenericWindTurbine(name='Simple', diameter=80, hub_height=70, power_norm=2000)

    CT = 0.75
    z0 = 0.0001
    kw = 2.4

    wfs = ['LG', 'R1', 'R2', 'HR1', 'HR2', 'HR3']
    Pgs = np.array([2.3, 2.3, 2.3, 2, 2.3, 8]) * 1e6
    Ds = [93.0, 82.0, 93.0, 80.0, 93.0, 164.0]
    Hs = [65.0, 69.0, 68.0, 70.0, 68.0, 105.0]
    Aws = [9.7, 10.5, 10.5, 11.0, 11.2, 11.5]
    Nturbs = [48, 72, 90, 80, 91, 49]
    Areas = np.array([4.8, 22, 35, 20, 33, 88]) * 1e6
    AEP_paper = [299, 566, 791, 598, 872, 1855]
    wfm = Sorensen_Larsen_2021(UniformSite(), wt, max_cp=0.48, ws_cutin=4, ws_cutout=25)

    for i in range(len(wfs)):
        power, wseff = wfm._calculate_power_ws(Pgs[i], CT, Ds[i], Hs[i], z0, Aws[i], kw, Nturbs[i], Areas[i])
        aep = power * 1e-9 * 24 * 365
        npt.assert_allclose(aep, AEP_paper[i], atol=[1, 3][int(i == 0)])


def test_Sorensen_etal_2024():
    # Site, Nt, n, D, Area, Aw, kw, U, S, a_fuga
    table1 = [["Lillgrund", 48, 6.9, 93, 4.7, 9.7, 2.4, 8.6, 3.9, 4.1],
              ["Rødsand 1", 72, 8.5, 82, 23.4, 10.5, 2.4, 9.3, 7.9, 5.4],
              ["Rødsand 2", 90, 9.5, 93, 34.8, 10.5, 2.4, 9.3, 7.5, 5.8],
              ["Horns Rev 1", 80, 8.9, 80, 19.6, 11, 2.4, 9.8, 7.0, 5.7],
              ["Horns Rev 2", 91, 9.5, 93, 35.8, 11.2, 2.4, 9.9, 7.5, 5.9],
              ["Horns Rev 3", 49, 7.0, 164, 100.7, 11.5, 2.4, 10.2, 10.2, 4.8]]

    # Check S and a_fuga form table1
    wfm = Sorensen_etal_2024(
        UniformSite(),
        SimpleGenericWindTurbine(
            name='Simple',
            diameter=80,
            hub_height=70,
            power_norm=2000))
    for Site, Nt, n, D, Area, Aw, kw, U, S, a_fuga in table1:
        Uh0 = Aw * gamma(1 + 1 / kw)
        S1 = np.sqrt(Area * 1e6) / ((np.sqrt(Nt) - 1) * D)
        npt.assert_allclose(S, S1, atol=0.1)
        npt.assert_allclose(wfm.correction_factor(Uh0=Uh0, sr=S1, Nturb=Nt), a_fuga, atol=0.4)

    table2 = [  # ["Lillgrund", 335, np.nan],
        ["Rødsand1", 540, 9.6],
        ["Rødsand2", 790, 6.8],
        ["Hornsrev1", 580, 17],
        ["Hornsrev2", 880, 6.7],
        ["Hornsrev3", 1750, np.nan]]
    for site, AEP, rsd in table2:
        wf = DKWindTurbines().get_wind_farm(site)
        da = wf.get_production()
        first_year = da.time[np.where(~np.isnan(np.sum(da.values, 0)))[0][0]].dt.year.item()
        # the table values are rounded and the start and stop years are not start+2..2019 as stated in the paper
        last_year = {'Hornsrev3': 2021}.get(site, 2020)
        y_lst = np.arange(first_year + 2, last_year)
        p = [da.sel(time=slice(str(y), str(y))).sum().item() for y in y_lst]
        atol = {'Hornsrev3': 65}.get(site, 15)
        npt.assert_allclose(np.mean(p) * 1e-6, AEP, atol=atol)
        if site != 'Hornsrev3':
            npt.assert_allclose(np.std(p) / np.mean(p) * 100, rsd, atol=4)


def test_MinimalisticWindFarmModel():

    wt = SimpleGenericWindTurbine(name='Simple', diameter=80, hub_height=70, power_norm=2000)

    ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]

    site = Hornsrev1Site(ti=ti)
    x, y = site.initial_position.T
    for wfm, eff, ref in [(Sorensen_Larsen_2021(site, wt), 1, 553.948821),
                          (Sorensen_etal_2024(site, wt), .91, 592.386668),
                          (Larsen_etal2026(site, wt, latitude=55), .91, 617.022948)]:
        res = wfm.aep(x, y) * eff
        npt.assert_allclose(res, ref, rtol=0.001)


def test_MinimalisticWindFarmModel_specify_args():
    wt = V80()

    ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]

    site = Hornsrev1Site(ti=ti)
    x, y = site.initial_position.T
    with pytest.raises(AttributeError, match="'V80' object has no attribute 'max_cp'"):
        Larsen_etal2026(site, wt, 55)

    aep1 = Larsen_etal2026(site, wt, 55, max_cp=.48).aep(x, y)
    aep2 = Larsen_etal2026(site, wt, 55, max_cp=.48, ws_cutin=4, ws_cutout=25).aep(x, y)
    npt.assert_allclose(aep1, aep2)


def test_models():
    wt = SimpleGenericWindTurbine(name='Simple', diameter=80, hub_height=70, power_norm=2000)
    ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]
    z0 = fuga_utils.z0(np.mean(ti), zref=np.mean(wt.hub_height()), zeta0=0)[0]
    site = Hornsrev1Site(ti=ti)
    x, y = site.initial_position.T

    Pg = wt.power([8, 12, 16]).max()
    ct_max = wt.ct([6, 8, 12, 16]).max()
    area = farm_area(wt_x=x, wt_y=y)

    for wfm, eff, ref in [
        (Sorensen_Larsen_2021(site, wt, correction_factor=7.8), 1, 671.8239668312056),
        (Sorensen_etal_2024(site, wt), .91, 591.656773),
        (Larsen_etal2026(site, wt, latitude=55), .91, 617.237787),
    ]:

        aep = wfm._calculate_power_ws(Pg=Pg,
                                      CT=ct_max,
                                      D=wt.diameter(),
                                      H=wt.hub_height(),
                                      z0=z0,
                                      Aw=10,
                                      kw=2.4,
                                      Nturb=len(x),
                                      Area=area)[0] * 24 * 365 * 1e-9

        npt.assert_allclose(aep, ref, rtol=0.001)


def test_a():
    wt = SimpleGenericWindTurbine(name='Simple', diameter=80, hub_height=70, power_norm=2000)
    ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]
    z0 = fuga_utils.z0(np.mean(ti), zref=np.mean(wt.hub_height()), zeta0=0)[0]
    site = Hornsrev1Site(ti=ti)
    x, y = site.initial_position.T
    wfm = Larsen_etal2026(site, wt, 55)
    N = len(x)
    Area = farm_area(wt_x=x, wt_y=y)
    D = wt.diameter()
    S = np.sqrt(Area) / (D * (np.sqrt(N) - 1))

    a = wfm.get_a_func(np.sqrt(N), S, U_rated=11)
    # print(np.round(a(np.array([5., 10, 15])), 3).tolist())
    ref = [0.700318, 0.724581, 1.007877]
    if 0:
        ws = np.arange(4., 25)
        plt.plot(ws, a(ws))
        plt.plot([5, 10, 15], ref, 'xr')
        plt.show()
    npt.assert_allclose(a(np.array([5., 10, 15])), ref, rtol=0.001)


if __name__ == '__main__':
    test_models()
