import numpy as np
from scipy.interpolate._rgi import RegularGridInterpolator as RGI

from py_wake.site.streamline_distance import StreamlineDistance
from py_wake.site.xrsite import XRSite
from py_wake.utils.streamline import VectorField3D


def wrf2pywake(ds, TI=None, variable_lst=['WD', 'WS', 'TI']):
    if 'west_east' in ds:
        ds = ds.rename(west_east='x', south_north='y', height='h')
    if 'TI' not in ds:
        if TI is None:
            TI = np.sqrt(2 / 3 * ds['TKE']) / ds['WS']
        ds['TI'] = TI
    ds = ds[variable_lst]
    dims = [d for d in ['x', 'y', 'h', 'time'] if d in ds.dims]
    return ds.transpose(*dims)


class WRFVectorField(VectorField3D):
    def __init__(self, ds):
        ds = wrf2pywake(ds)
        self.ds = ds
        grid = [ds.x.values, ds.y.values, ds.h.values, np.arange(len(ds.time))]
        theta = np.deg2rad(270 - ds.WD.values)
        Vx, Vy = np.cos(theta) * ds.WS.values, np.sin(theta) * ds.WS.values
        values = np.moveaxis([Vx, Vy], 0, -1)
        self.mean_values = np.mean(values, (1, 2, 3))
        self.interp = RGI(grid, values, bounds_error=False)

    def __call__(self, wd, time, x, y, h):
        time = np.broadcast_to(time, x.shape)
        Vx, Vy = np.transpose(self.interp(np.array([x, y, h, time]).T))
        return np.array([Vx, Vy, Vx * 0]).T


class WRFSite(XRSite):
    def __init__(self, ds, TI=None, streamlines=True):
        if streamlines:
            distance = StreamlineDistance(WRFVectorField(ds))
        else:
            distance = None
        ds = wrf2pywake(ds, TI)
        ds['P'] = 1
        ds['datetime'] = ds.time
        ds['time'] = np.arange(len(ds.time))
        ds['ws'] = [0]
        XRSite.__init__(self, ds, distance=distance)
