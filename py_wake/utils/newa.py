import datetime
import io
import urllib.request

import numpy as np
import windkit as wk
import xarray as xr
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm

from py_wake.site.xrsite import XRSite
from _datetime import timezone

newa_crs = "+proj=lcc +lat_0=54 +lon_0=15 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs +type=crs"


def get_newa_point(x, y, h, crs):
    p = wk.spatial.create_dataset(x, y, 1, crs=crs)
    newa_heights = np.array([50., 75., 100., 150., 200., 250., 500.])
    h_lst = newa_heights[np.arange(*np.searchsorted(newa_heights, [np.min(h), np.max(h)]) + [-1, 1])]
    return p, h_lst


def reproject(x, y, from_crs, to_crs):
    """Converts position to new crs

    Often used crs:
    - 'EPSG:4326' for WGS 84 longitude/latitude (degrees)
    - 'EPSG:25832' for UTM zone 32N (m), Europe between 6°E and 12°E and approximately 36°30'N to 84°N.
    - '+proj=lcc +lat_0=54 +lon_0=15 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs +type=crs' for NEWA coordinates (m)

    Parameters
    ----------
    x : number or array_like
        x position / longitude
    y : number or array_like
        y position / latitude
    from_crs : str
        Coordinate reference system of the input coordinates
    to_crs : str
        Coordinate reference system of the output coordinates

    """

    p = wk.spatial.create_dataset(x, y, 1, crs=from_crs)
    p = wk.spatial.reproject(p, to_crs)
    return p.west_east.values, p.south_north.values


def to_newa_crs(x, y, crs):
    """Converts position to coordinates compatible with NEWA

    Parameters
    ----------
    x : number or array_like
        x position / longitude
    y : number or array_like
        y position / latitude
    crs : str
        Coordinate reference system of the input coordinates, e.g.
        - 'EPSG:4326' for WGS 84 longitude/latitude (degrees)
        - 'EPSG:25832' for UTM zone 32N (m), Europe between 6°E and 12°E and approximately 36°30'N to 84°N.
    """
    return reproject(x, y, from_crs=crs, to_crs=newa_crs)


def wk2pywake(ds, TI=None, variable_lst=['WD', 'WS', 'TI']):
    if 'west_east' in ds:
        ds = ds.rename(west_east='x', south_north='y', height='h')
    if 'TI' not in ds:
        if TI is None:
            TI = np.sqrt(2 / 3 * ds['TKE']) / ds['WS']
        ds['TI'] = TI
    ds = ds[variable_lst]
    dims = [d for d in ['x', 'y', 'h', 'time'] if d in ds.dims] + [...]
    return ds.transpose(*dims)


class NEWAGridTimeseries():
    def __init__(self, ds):
        self.ds = ds

    def to_pywake(self, TI=None, variable_lst=['WD', 'WS', 'TI']):
        return wk2pywake(self.ds, TI=TI, variable_lst=variable_lst)

    def interp_height(self, height):

        ds = self.ds.interp(height=np.atleast_1d(height))
        for h in np.atleast_1d(height):
            ih2 = np.minimum(np.searchsorted(self.ds.height, h), 1)
            ih1 = ih2 - 1

            h1, h2 = self.ds.height[[ih1, ih2]].values
            u1, u2 = [self.ds.WS.isel(height=ih).values for ih in [ih1, ih2]]
            u = u1 + (u2 - u1) * (np.log(h) - np.log(h1)) / (np.log(h2) - np.log(h1))

            ds.WS.sel(height=h)[:] = u
        self.ds = ds.sel(height=height)
        return self.ds

    def get_ds_list(self, time_step, height):
        t_lst = self.get_time_list(time_step)
        if 'height' in self.ds.dims and len(self.ds.height) > 1:
            ds_height = self.interp_height(height=height)
        else:
            ds_height = self.ds
        if 'hours' not in self.ds or self.ds['hours'].dims[0] != time_step:
            h_lst = [t1.astype('<M8[h]') - t0.astype('<M8[h]') for t0, t1 in zip(t_lst[:-1], t_lst[1:])]
            self.ds['hours'] = (time_step, h_lst)
            self.ds = self.ds.assign_coords({time_step: t_lst[:-1]})
        return t_lst, [ds_height.isel(time=(ds_height.time.values >= t0) & (
            ds_height.time.values < t1)) for t0, t1 in zip(t_lst[:-1], t_lst[1:])]

    @classmethod
    def from_web(cls, x, y, h, start='2002-01-01', stop='2002-01-01T23:30',
                 crs='EPSG:25832'):  # pragma: no cover
        p, h_lst = get_newa_point(x, y, h, crs)
        p = wk.spatial.reproject(p, "EPSG:4326")
        lon, lat = p['west_east'].values, p['south_north'].values

        height = "&".join([f"height={int(h)}" for h in h_lst])
        if stop is not None:
            stop = f'&dt_stop={stop.replace(":", "%3A")}'
        else:
            stop = ''
        start = start.replace(":", "%3A")
        var_lst = ['WS', 'WD', 'TI', 'RMOL', 'RHO']
        var_str = "&".join([f'variable={v.replace("TI", "TKE")}' for v in var_lst])
        url = 'https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/v1/'
        if len(lon) == len(lat) == 1:
            url += f'get-data-point?latitude={lat[0]:.8f}&longitude={lon[0]:.8f}'
        else:
            url += f'get-data-bbox?southBoundLatitude={lat.min():.8f}&northBoundLatitude{lat.min():.8f}&westBoundLongitude={lon.min():.8f}eastBoundLongitude={lon.max():.8f}'
        url += f'&{height}&{var_str}&dt_start={start}{stop}'
        f, msg = urllib.request.urlretrieve(url)
        ds = xr.open_dataset(f)
        return cls(ds)

    @staticmethod
    def from_grid_dataset(ds, x, y, h, start=None, stop=None, crs='EPSG:25832'):
        p, h_lst = get_newa_point(x, y, h, crs)
        p = wk.spatial.reproject(p, newa_crs)
        ds = ds.sel(height=h_lst, time=slice(start, stop))
        we_slice = slice(*np.searchsorted(ds.west_east, [p.west_east.min(), p.west_east.max()]) + [-1, 1])
        sn_slice = slice(*np.searchsorted(ds.south_north, [p.south_north.min(), p.south_north.max()]) + [-1, 1])
        ds = ds.isel(west_east=we_slice, south_north=sn_slice)
        return NEWAGridTimeseries(ds)

    @classmethod
    def from_zarr(cls, x, y, h, start='2002-01-01', stop='2002-01-01T23:30',
                  crs='EPSG:25832', zarr_path=None):
        ds = xr.open_zarr(zarr_path, consolidated=False)
        return cls.from_grid_dataset(ds, x, y, h, start=start, stop=stop, crs=crs)


class NEWAPointTimeseries(NEWAGridTimeseries):
    def add_P(self, time_step, height, wd=np.arange(360), ws=np.arange(3, 26)):
        dwd = np.diff(wd[:2])[0]
        assert np.all(dwd == np.diff(wd))
        dws = np.diff(ws[:2])[0]
        assert np.all(dws == np.diff(ws))

        def tswc2P(tswc):
            dwd = np.diff(wd[:2])[0]
            assert np.all(dwd == np.diff(wd))
            dws = np.diff(ws[:2])[0]
            assert np.all(dws == np.diff(ws))
            return np.histogram2d((tswc.WD.values + dwd / 2) % 360 - dwd / 2, tswc.WS.values,
                                  bins=[len(wd), len(ws)],
                                  range=[[wd[0] - dwd / 2, wd[-1] + dwd / 2], [ws[0] - dws / 2, ws[-1] + dws / 2]])[0] / tswc.time.size
        t_lst, ds_lst = self.get_ds_list(time_step, height)
        P = [tswc2P(ds) for ds in ds_lst]
        self.ds['P'] = ((time_step, 'wd', 'ws'), np.array(P))
        self.ds = self.ds.assign_coords({'wd': wd, 'ws': ws})

    def add_weibull(self, time_step, height, n_sectors=12):
        t_lst, ds_lst = self.get_ds_list(time_step, height)
        wwc = xr.concat([tswc2weibull(ds) for ds in tqdm(ds_lst)], dim=time_step)

        self.ds['Sector_frequency'] = wwc.wdfreq
        self.ds['Weibull_A'] = wwc.A
        self.ds['Weibull_k'] = wwc.k

    def get_weibullSite_list(self):
        assert 'Weibull_A' in self.ds and 'Weibull_k' in self.ds and 'Sector_frequency' in self.ds, "Weibull parameters and sector frequencies must be added to the dataset using add_weibull() before calling get_weibullSite_list()"
        dim = self.ds.Weibull_A.dims[0]
        ds = wk2pywake(self.ds, variable_lst=['Weibull_A', 'Weibull_k', 'Sector_frequency', 'TI'])

        def get_ds(i):
            ds_i = ds.isel(**{dim: i})
            ds_i['TI'] = ds_i.TI.mean()
            return ds_i.drop_dims('time').rename(sector='wd')
        return [XRSite(get_ds(i)) for i in range(ds[dim].size)], self.ds['hours'] / np.timedelta64(1, 'h')

    def get_time_list(self, time_step):
        assert time_step in ['year', 'month', 'day']
        scaling = {'<M8[s]': 1, '<M8[ns]': 1e9}[self.ds.time.dtype.str]
        dt = relativedelta(seconds=int((np.diff(self.ds.time[-2:])[0] / scaling).astype(int)))
        delta = relativedelta(**{f'{time_step}s': 1})
        start = datetime.datetime.fromtimestamp(self.ds.time[0].item() / scaling, timezone.utc) - dt + delta
        stop = datetime.datetime.fromtimestamp(self.ds.time[-1].item() / scaling, timezone.utc) + dt + delta

        if time_step == 'year':
            start_ymd = start.year, 1, 1
            stop_ymd = stop.year, 1, 1

        elif time_step == 'month':
            start_ymd = start.year, start.month, 1
            stop_ymd = stop.year, stop.month, 1
        elif time_step == 'day':
            start_ymd = start.year, start.month, start.day
            stop_ymd = stop.year, stop.month, stop.day
        n = time_step[0].upper()

        return np.arange(np.datetime64("%d-%02d-%02d" % start_ymd), np.datetime64("%d-%02d-%02d" %
                         stop_ymd), np.timedelta64(1, n), dtype=f'datetime64[{n}]')

    def get_production(self, power):
        """Calculate power production time series by multiplying the power curve with the wind speed distribution.

        paprameters:
        ------------
        power: xarray.DataArray
            Power curve as a function of wind speed, wind direction and/or wind turbine

        returns:
        --------
        xarray.DataArray                Yearly/monthly power production time series [GWh]

        """
        assert 'P' in self.ds, "Power distribution P must be added to the dataset using add_P() before calling get_production()"
        dims = [d for d in ('wt', 'wd', 'ws') if d in power.dims or d in self.ds.P.dims]
        hours = self.ds.hours / np.timedelta64(1, 'h')
        res = ((power * self.ds.P).sum(dims) * 1e-9 * hours).values
        time_step = self.ds.hours.dims[0]
        return xr.DataArray(res, dims=time_step, coords={time_step: self.ds[time_step].values})

    @staticmethod
    def from_grid_dataset(ds, x, y, h, start=None, stop=None, crs='EPSG:25832'):
        p, h_lst = get_newa_point(x, y, h, crs)
        wrf_crs = "+proj=lcc +lat_0=54 +lon_0=15 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs +type=crs"
        p = wk.spatial.reproject(p, wrf_crs)
        ds = ds.sel(height=h_lst, time=slice(start, stop))
        ds = ds.interp(west_east=p.west_east.item(), south_north=p.south_north.item(), method='nearest')
        return NEWAPointTimeseries(ds)


def tswc2weibull(tswc, n_sectors=12):
    bwc = wk.bwc_from_tswc(tswc[['WD', 'WS']].rename({'WS': 'wind_speed', 'WD': 'wind_direction'}), n_sectors=1)
    return wk.weibull_fit(bwc).isel(point=0)
