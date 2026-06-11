import tempfile
import warnings

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from py_wake.deficit_models.noj import NOJ
from py_wake.examples.data.hornsrev1 import (
    V80,
    Hornsrev1Site,
    wt16_x,
    wt16_y,
    wt_x,
    wt_y,
)
from py_wake.site._site import UniformSite
from py_wake.tests import ptf
from py_wake.utils.newa import NEWAPointTimeseries, NEWAGridTimeseries, to_newa_crs
from py_wake.utils.plotting import setup_plot
import urllib.error


@pytest.fixture()
def zarr_file():
    ds_tmp = xr.load_dataset(ptf('wrf/Hornsrev1_2020_04.nc',
                                 known_hash='2df8efff6ed71d3c84639ba3d669b4d65e1ec45170b37f0305f3f0f7b54df048'))
    x, y, h = np.median(wt_x), np.median(wt_y), 70
    with tempfile.TemporaryDirectory() as folder:
        ds_tmp.to_zarr(folder, consolidated=False)
        yield folder


def test_to_newa_crs():
    x, y = to_newa_crs(wt_x[0], wt_y[0], crs='epsg:25832')
    npt.assert_allclose([x[0], y[0]], [-444357.78223744, 183555.988855])

    x, y = to_newa_crs(7.76336669921875, 55.44114685058594, crs='epsg:4326')

    npt.assert_allclose([x[0], y[0]], [-447000, 177000], atol=2)


def test_NEWAGridTimeseries_compare_zarr_web(zarr_file):
    h = [70, 80, 110]

    newa_grid1 = NEWAGridTimeseries.from_zarr(
        wt_x, wt_y, h, start='2020-04-01', stop='2020-04-01T23:30', zarr_path=zarr_file)

    ds1 = newa_grid1.to_pywake()

    x, y = to_newa_crs(wt_x, wt_y, crs='epsg:25832')
    if 0:
        ds1.WS[:, :, 0, 0].plot()
        plt.plot(x, y, '2r')
        plt.show()
    assert ds1.x.min() < x.min() and ds1.x.max() > x.max() and ds1.y.min() < y.min() and ds1.y.max() > y.max()
    try:
        newa_grid2 = NEWAGridTimeseries.from_web(x, y, h, start='2020-04-01', stop='2020-04-01T23:30')
    except urllib.error.HTTPError as e:
        warnings.warn(f"Web request failed with error {e}. Skipping test_NEWAGridTimeseries_compare_zarr_web.")
        return
    ds2 = newa_grid2.to_pywake()

    npt.assert_allclose(ds1.WS.sel(h=75), ds2.WS.sel(h=75))


def test_NEWAPointTimeseries_compare_zarr_web(zarr_file):

    x, y, h = np.median(wt_x), np.median(wt_y), [70, 80, 110]

    newa_pts1 = NEWAPointTimeseries.from_zarr(
        x, y, h, start='2020-04-01', stop='2020-04-01T23:30', zarr_path=zarr_file)

    ds1 = newa_pts1.to_pywake()
    try:
        newa_pts2 = NEWAPointTimeseries.from_web(x, y, h, start='2020-04-01', stop='2020-04-01T23:30')
    except urllib.error.HTTPError as e:
        warnings.warn(f"Web request failed with error {e}. Skipping test_NEWAGridTimeseries_compare_zarr_web.")
        return
    ds2 = newa_pts2.to_pywake()

    npt.assert_allclose(ds1.WS.sel(h=75), ds2.WS.sel(h=75))


def test_NEWAPointTimeseries_interp_height(zarr_file):

    x, y, h = np.median(wt_x), np.median(wt_y), [70, 80, 110]

    newa_pts1 = NEWAPointTimeseries.from_zarr(
        x, y, h, start='2020-04-01', stop='2020-04-01T23:30', zarr_path=zarr_file)

    profile = newa_pts1.ds.WS.mean('time')
    ws75 = newa_pts1.ds.WS.sel(height=75)

    newa_pts1.ds = newa_pts1.ds.sel(height=[50, 100])

    nearest = newa_pts1.ds.WS.interp(height=75, method='nearest')
    linear = newa_pts1.ds.WS.interp(height=75)
    log = newa_pts1.ds.WS.interp_log(height=[75])

    if 0:
        profile.plot(y='height')
        plt.plot(nearest.mean(), nearest.height, '.', label='nearest')
        plt.plot(linear.mean(), linear.height, '.', label='linear')
        plt.plot(log.mean(), log.height, '.', label='log')
        setup_plot()
        plt.figure()
        (nearest - ws75).plot(label='nearest')
        (linear - ws75).plot(label='linear')
        (log - ws75).plot(label='log')

        setup_plot()

        plt.show()

    npt.assert_array_less(np.abs(linear - ws75), np.abs(nearest - ws75))
    npt.assert_array_less(np.abs(log.mean() - ws75.mean()), np.abs(linear.mean() - ws75.mean()))
    npt.assert_allclose(log.mean(), ws75.mean(), .005)


def test_NEWAPointTimeseries_P_weibull():

    ds = xr.load_dataset(ptf('wrf/Hornsrev1_2020_04.nc',
                             known_hash='2df8efff6ed71d3c84639ba3d669b4d65e1ec45170b37f0305f3f0f7b54df048'))

    newa_pts = NEWAPointTimeseries.from_grid_dataset(ds, x=np.median(wt_x), y=np.median(wt_y), h=70)
    newa_pts.add_P(time_step='month', height=70, ws=np.arange(1, 26))
    P = newa_pts.ds.P
    npt.assert_allclose(P.sum(('wd', 'ws')), 1, atol=0.07)

    wfm = NOJ(Hornsrev1Site(), V80())
    sim_res = wfm(wt16_x, wt16_y, ws=P.ws)
    aep_from_site_mean = sim_res.aep().sum(('wd', 'ws'))

    # april_mean_ws = (P / len(P.month) * P.ws).sum()  # 8.27m/s
    # site_mean_ws = (sim_res.WS * sim_res.P).sum()  # 9.37 m/s

    aep_from_april_P = (P.sum('month') * sim_res.Power).sum(('wd', 'ws')) * 1e-9 * 365 * 24

    newa_pts.add_weibull(time_step='month', height=70, n_sectors=12)
    site_lst, da_hours = newa_pts.get_weibullSite_list()
    wfm = NOJ(site_lst[0], V80())
    aep_from_april_weibull = wfm(wt16_x, wt16_y, ws=P.ws).aep().sum(('wd', 'ws'))

    if 0:
        (aep_from_april_P + 1.6).plot(label='AEP from april + 1.6 GWh')
        (aep_from_april_weibull + 1.6).plot(label='AEP from april weibull + 1.6 GWh')
        aep_from_site_mean.plot(label='AEP from Hornsrev1Site')
        plt.legend()
        plt.show()

    # the mean wind speed in april is lower than the mean wind speed at the
    # site, so the AEP pr wt is around 1.6 GWh lower
    npt.assert_allclose(aep_from_site_mean - aep_from_april_P, 1.6, rtol=0.2)
    npt.assert_allclose(aep_from_site_mean - aep_from_april_weibull, 1.6, rtol=0.2)


@pytest.mark.parametrize("time_step,start, stop, ref", [
    ('year', '2010-01-01', '2011-12-31T23:00', ['2010', '2011', '2012']),
    ('year', '2010-01-01', '2011-12-31T22:00', ['2010', '2011']),
    ('year', '2010-01-01T01:00', '2011-12-31T23:00', ['2011', '2012']),
    ('month', '2010-03-01', '2010-04-30T23:00', ['2010-03-01', '2010-04-01', '2010-05-01']),
    ('month', '2010-03-01', '2010-04-30T22:00', ['2010-03-01', '2010-04-01']),
    ('month', '2010-03-01T01:00', '2010-04-30T23:00', ['2010-04-01', '2010-05-01']),
    ('month', '2010-11-01T00:00', '2010-12-31T23:00', ['2010-11-01', '2010-12-01', '2011-01-01']),
    ('month', '2010-12-01T01:00', '2011-01-31T23:00', ['2011-01-01', '2011-02-01']),
    ('day', '2010-03-15', '2010-03-16T23:00', ['2010-03-15', '2010-03-16', '2010-03-17']),
    ('day', '2010-03-15', '2010-03-16T22:00', ['2010-03-15', '2010-03-16']),
    ('day', '2010-03-15T01:00', '2010-03-16T23:00', ['2010-03-16', '2010-03-17']),
    ('day', '2010-12-30', '2010-12-31T23:00', ['2010-12-30', '2010-12-31', '2011-01-01']),
    ('day', '2010-12-31T01:00', '2011-01-01T23:00', ['2011-01-01', '2011-01-02']),
])
def test_NEWAPointTimeseries_get_time_list(time_step, start, stop, ref):
    t = np.r_[np.arange(start, stop, dtype='datetime64[h]').astype('datetime64[ns]'), np.datetime64(stop)]
    da = xr.DataArray(t, dims='time', coords={'time': t})
    npt.assert_array_equal(NEWAPointTimeseries(da).get_time_list(time_step), [np.datetime64(r) for r in ref])


def test_get_production(zarr_file):

    x, y, h = np.median(wt_x), np.median(wt_y), [70, 80, 110]

    newa_pts = NEWAPointTimeseries.from_zarr(
        x, y, h, start='2020-04-01', stop='2020-04-30T23:30', zarr_path=zarr_file)
    wd = np.arange(360)
    ws = np.arange(3, 26)
    power = NOJ(UniformSite(), V80())(wt16_x, wt16_y, wd=wd, ws=ws).Power
    with pytest.raises(AssertionError, match=r"Power distribution P must be added to the dataset using add_P\(\) before"):
        newa_pts.get_production(power)
    newa_pts.add_P('day', height=70, wd=wd, ws=ws)
    dayly_production = newa_pts.get_production(power)
    newa_pts = NEWAPointTimeseries.from_zarr(
        x, y, h, start='2020-04-01', stop='2020-05-01T23:30', zarr_path=zarr_file)
    newa_pts.add_P('month', height=70, wd=wd, ws=ws)
    monthly_production = newa_pts.get_production(power)

    npt.assert_array_almost_equal(dayly_production.sum(), monthly_production, 10)
