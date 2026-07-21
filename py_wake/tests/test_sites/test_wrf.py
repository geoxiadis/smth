from py_wake.examples.data.hornsrev1 import Hornsrev1WRFSite, V80
from py_wake.tests import npt
import numpy as np
import matplotlib.pyplot as plt
import pytest
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.deficit_models.noj import NOJLocalDeficit
from py_wake.flow_map import XYGrid
from py_wake.site.wrf import WRFSite


@pytest.mark.parametrize('streamlines,ref', [(False, [548.22095, 26.745916]),
                                             (True, [586.881971, -182.716646])])
def test_Hornsrev1WRFSite(streamlines, ref):
    site = Hornsrev1WRFSite(time_slice=slice(f"2020-04-27 12:00", f"2020-04-27"), streamlines=streamlines)
    wts = V80()
    x, y = site.initial_position[:4].T
    h = wts.hub_height()
    t = 10
    p = site.ds.interp(x=x[0], y=y[0], h=70).isel(time=t)
    npt.assert_array_almost_equal([p.WS.item(), p.WD.item(), p.TI.item()], [5.566621, 348.0761, 0.04780247], 4)

    t = 9
    dw, cw = [v[0, 1, 0, 0] for v in site.distance(x[:2], y[:2], x[:2] * 0 + h, wd_l=[0], time=[t])[:2]]

    if 0:
        X, Y = np.meshgrid(site.ds.x.values, site.ds.y.values)
        theta = np.deg2rad(270 - site.ds.WD)
        WS = site.ds.WS
        Vx, Vy = np.cos(theta) * WS, np.sin(theta) * WS
        site.ds.WS.isel(time=t).interp(h=h).plot.contourf(cmap='Blues_r', x='x')
        wts.plot(x[:2], y[:2], wt_number=0)
        plt.quiver(X, Y, Vx.isel(time=t).interp(h=h).values.T, Vy.isel(time=t).interp(h=h).values.T)
        if streamlines:
            streamlines = site.distance.vectorField.stream_lines(np.array([0]), time=[t],
                                                                 start_points=np.array([x[:1], y[:1], [h]]).T,
                                                                 dw_stop=np.array([4000]))
            plt.plot(streamlines[0][:, 0], streamlines[0][:, 1])

        plt.plot([x[0], x[0] + cw], [y[0], y[0] - dw])
        plt.xlim([x[0] - 4000, x[0] + 4000])
        plt.ylim([y[0] - 4000, y[0] + 4000])
        plt.show()

    npt.assert_array_almost_equal([dw, cw], ref)

    for cls in [PropagateDownwind, All2AllIterative]:
        wfm = cls(site, wts, NOJLocalDeficit(use_effective_ti=False))
        sim_res = wfm(x, y, wd=[0, 10, 20], ws=[10, 10, 10], time=True)
        fm = sim_res.flow_map(XYGrid(resolution=20))
        npt.assert_array_equal(fm.wd, [0, 10, 20])
        fm = sim_res.flow_map(XYGrid(resolution=20), time=1)
        npt.assert_array_equal(fm.wd, [10])


@pytest.mark.parametrize('wt_idx,wd,t,ref_shape', [
    (0, 0, 9, (1, 12, 3)),  # 1 wt, 1 wd, 1 time
    ([0], 0, 9, (1, 12, 3)),  # 1 wt, 1 wd, 1 time
    ([0, 1], [0, 0], [9, 9], (2, 12, 3)),  # 2 wt, same wd and time
    ([0, 1], 0, 9, (2, 12, 3)),  # 2 wt, same wd and time
    ([0, 0], [0, 10], [9, 10], (2, 12, 3)),  # 1 wt, two wd and time
    ([0, 0], [0, 10], 0, (2, 21, 3)),  # 1 wt, two wd and time=0
    ([0, 0], [0, 10], False, (2, 21, 3)),  # 1 wt, two wd and time=False (uses time=0)

])
def test_streamlines(wt_idx, wd, t, ref_shape):
    site = Hornsrev1WRFSite(time_slice=slice(f"2020-04-27 12:00", f"2020-04-27"), streamlines=True)

    x, y = site.initial_position[wt_idx].T
    h = x * 0 + 70

    streamlines = site.distance.vectorField.stream_lines(wd, time=t,
                                                         start_points=np.array([x, y, h]).T,
                                                         dw_stop=200)
    npt.assert_array_equal(streamlines.shape, ref_shape)
