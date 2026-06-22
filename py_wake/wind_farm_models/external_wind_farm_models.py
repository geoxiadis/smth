from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import newaxis as na
from scipy.interpolate import RegularGridInterpolator as RGI

from py_wake.flow_map import Points
from py_wake.site.distance import StraightDistance
from py_wake.utils.gradients import item_assign
from py_wake.utils.model_utils import Model
from py_wake.utils.xarray_utils import sel_interp_all


class ExternalWindFarm(Model, ABC):

    def __init__(self, name, windTurbines, wt_x, wt_y, wt_h=None):
        self.name = name
        self.wt_x = wt_x
        self.wt_y = wt_y
        self.wf_x = np.mean([np.min(wt_x), np.max(wt_x)])
        self.wf_y = np.mean([np.min(wt_y), np.max(wt_y)])
        if wt_h is None:
            self.wt_h = windTurbines.hub_height(windTurbines.types())
        self.wf_h = np.mean(self.wt_h)
        self.windTurbines = windTurbines

    def set_include_wd(self, include_wd):
        if callable(include_wd):
            self.include_wd_func = include_wd
        else:
            assert isinstance(include_wd, (int, float, list, tuple, np.ndarray))

            def include_wd_func(wd):
                return np.round(wd) in set(np.asarray(include_wd).tolist())
            self.include_wd_func = include_wd_func

    def rel2abs(self, dw, hcw, dh, WD):
        theta = np.deg2rad(270 - WD)
        co, si = np.cos(theta), np.sin(theta)
        x = co * dw - hcw * si + self.wf_x
        y = si * dw + hcw * co + self.wf_y
        h = dh + self.wf_h
        return x, y, h

    def plot(self, ax=None):
        ax = ax or plt.gca()
        ax.plot(self.wt_x, self.wt_y, '1', label=self.name)

    @abstractmethod
    def __call__(self):
        ""


class ExternalWFMWindFarm(ExternalWindFarm):
    def __init__(self, name, windFarmModel, wt_x, wt_y, wt_h=None, type=0, include_wd=np.arange(360), **kwargs):
        self.wfm = windFarmModel
        self.wfm_kwargs = {**kwargs, 'x': wt_x, 'y': wt_y, 'h': wt_h, 'type': type}
        self.set_include_wd(include_wd)
        ExternalWindFarm.__init__(self, name, windFarmModel.windTurbines, wt_x, wt_y, wt_h)

    def get_relevant_wd(self, target_xyh, ws=10, tol=1e-6):
        x, y, h = target_xyh
        h = np.zeros_like(x) + h
        sim_res = self.get_sim_res(wd=np.arange(360), ws=[ws])
        fm = sim_res.flow_map(Points(x, y, h), wd=sim_res.wd)
        return fm.wd[((fm.WS - fm.WS_eff).max('i') > tol)].values

    def get_sim_res(self, wd, ws):
        return self.wfm(**{**self.wfm_kwargs, 'ws': ws, 'wd': wd})

    def __call__(self, i, l, deficit_jlk, WS_jlk, WS_eff_ilk,
                 WD_ilk, dst_xyh_jlk, IJLK, dw_ijlk, hcw_ijlk, dh_ijlk, time=False, **_):
        I, J, L, K = IJLK
        WD_l = np.round(WD_ilk[np.minimum(i, len(WD_ilk) - 1), :, 0])
        m_lst = [m for m, wd in enumerate(WD_l) if l[m] and self.include_wd_func(wd)]
        if deficit_jlk is None:
            deficit_jlk = np.zeros((J, L, K))
        if m_lst:
            M = len(m_lst)
            ws_lst = np.sort(np.unique(np.r_[WS_eff_ilk.min(), np.round(WS_eff_ilk.flatten()), WS_eff_ilk.max()]))
            wd_lst = WD_l[m_lst]
            sim_res = self.wfm(**{**self.wfm_kwargs, 'ws': ws_lst, 'wd': np.sort(np.unique(wd_lst)), 'time': time})
            from py_wake.site.streamline_distance import StreamlineDistance
            if isinstance(self.wfm.site.distance, StreamlineDistance):
                # Simulation and flow map of external WF is computed with stream line distances
                # the WF wake must therefore be calculated for fixed abs positions
                dst_xyh_jlk = dst_xyh_jlk
            else:
                # Simulation and flow map of external WF is computed with straight distance.
                # The WF wakes must therefore be calcualted using the relative distances (including streamlines)
                _dst_xyh_jlk = self.rel2abs(dw_ijlk[i], hcw_ijlk[i], dh_ijlk[i], WD_l[na, :, na])
                if M == 1 or np.abs(dst_xyh_jlk[0][:, [0]] - dst_xyh_jlk[0]).max() > 1e-10:
                    # destinations differ for differnet wd, use destinations calculated from
                    # distances to include streamlines
                    dst_xyh_jlk = _dst_xyh_jlk
            if dst_xyh_jlk[0].shape[1:] == (1, 1) and M > J:
                # same destination for all wd
                x_j, y_j, h_j = [v[:, 0, 0] for v in dst_xyh_jlk]
                lw_j, WS_eff_jlk, TI_eff_jlk = self.wfm._flow_map(x_j[:, na], y_j[:, na], h_j[:, na], sim_res.localWind,
                                                                  wd_lst, ws_lst, sim_res)
                _deficit_jlk = lw_j.WS_ilk - WS_eff_jlk

                deficit_jmk = np.moveaxis([RGI([ws_lst], np.moveaxis(_deficit_jlk[:, l], 0, -1))(WS_eff_ilk[i, m])
                                          for l, m in enumerate(m_lst)], -1, 0)
            else:
                assert dst_xyh_jlk[0].shape[2] == 1, "ExternalWFMWindFarm does not support inflow dependent positions"

                def get_deficit_l(m):
                    x_j, y_j, h_j = [np.broadcast_to(v_jlk, (J, L, K))[:, m] for v_jlk in dst_xyh_jlk]
                    WD = WD_l[m]
                    WS_eff = WS_eff_ilk[i, m]
                    sr = sel_interp_all(sim_res)(dict(wd=WD, ws=WS_eff))
                    lw_j, WS_eff_jlk, TI_eff_jlk = self.wfm._flow_map(x_j, y_j, h_j, sim_res.localWind, WD, WS_eff, sr)
                    return lw_j.WS_ilk[:, 0] - WS_eff_jlk[:, 0]

                deficit_jmk = np.moveaxis([get_deficit_l(m) for m in m_lst], 0, 1)
            deficit_jlk = item_assign(deficit_jlk, m_lst, deficit_jmk, axis=1)
        return deficit_jlk


class ExternalXRAbsWindFarm(ExternalWindFarm):
    def __init__(self, name, ds, windTurbines, wt_x, wt_y, relative_distance=True, include_wd=None):
        ExternalWindFarm.__init__(self, name, windTurbines, wt_x, wt_y)
        x = [ds.x.values, ds.y.values, ds.h.values, ds.wd.values, ds.ws.values]
        self.deficit_interp = RGI(x, ds.deficit.values, bounds_error=False)
        if include_wd is None:
            include_wd = ds.wd.values
        self.set_include_wd(include_wd)
        self.relative_distance = relative_distance

    def to_xarray(self):
        di = self.deficit_interp
        return xr.Dataset({'deficit': (['x', 'y', 'h', 'wd', 'ws'], di.values)},
                          coords=dict(x=di.grid[0], y=di.grid[1], h=di.grid[2], wd=di.grid[3], ws=di.grid[4]))

    @classmethod
    def generate(cls, name, grid_xyh, windFarmModel, wt_x, wt_y, type=0, **kwargs):
        sim_res = windFarmModel(wt_x, wt_y, type=type, **kwargs)
        X, Y, H = np.meshgrid(*grid_xyh, indexing='ij')
        x_j, y_j, h_j = X.flatten(), Y.flatten(), H.flatten()

        lw_j, WS_eff_jlk, TI_eff_jlk = windFarmModel._flow_map(x_j[:, na], y_j[:, na], h_j[:, na], sim_res.localWind,
                                                               sim_res.wd, sim_res.ws, sim_res)
        deficit = lw_j.WS_ilk - WS_eff_jlk.reshape(X.shape + (WS_eff_jlk.shape[1:]))
        ds = xr.Dataset({'deficit': (['x', 'y', 'h', 'wd', 'ws'], deficit)},
                        coords=dict(x=grid_xyh[0], y=grid_xyh[1], h=np.atleast_1d(grid_xyh[2]), wd=sim_res.wd, ws=sim_res.ws))

        from py_wake.site.streamline_distance import StreamlineDistance
        return cls(name, ds.sortby('wd'), windFarmModel.windTurbines, wt_x, wt_y,
                   relative_distance=not isinstance(windFarmModel.site.distance, StreamlineDistance))

    def __call__(self, i, l, deficit_jlk, WS_eff_ilk, WS_ilk,
                 dw_ijlk, hcw_ijlk, dh_ijlk,
                 WD_ilk, IJLK, dst_xyh_jlk, **_):
        I, J, L, K = IJLK
        WD_l = WD_ilk[np.minimum(i, len(WD_ilk) - 1), :, 0]
        m_lst = [m for m, wd in enumerate(WD_l) if l[m] and self.include_wd_func(wd)]
        if deficit_jlk is None:
            deficit_jlk = np.zeros((J, L, K))
        if m_lst:
            M = len(m_lst)
            WS_eff_ilk[i]

            if self.relative_distance:
                # Simulation and flow map of external WF is computed with straight distance.
                # The WF wakes must therefore be calcualted using the relative distances (including streamlines)
                _dst_xyh_jlk = self.rel2abs(dw_ijlk[i], hcw_ijlk[i], dh_ijlk[i], WD_l[na, :, na])
                if M == 1 or np.abs(dst_xyh_jlk[0][:, [0]] - dst_xyh_jlk[0]).max() > 1e-10:
                    # destinations differ for different wd, use destinations calculated from
                    # distances to include streamlines
                    dst_xyh_jlk = _dst_xyh_jlk
            if dst_xyh_jlk[0].shape[1:] == (1, 1) and M > J:
                # same destination for all wd
                # make interpolator with current destination and interpolate ws for each wd
                di = self.deficit_interp
                fm_jlk = RGI(di.grid[:3], di.values, bounds_error=False)(np.array([v[:, 0, 0] for v in dst_xyh_jlk]).T)
                np.nan_to_num(fm_jlk, copy=False)
                rgi_dst = RGI(di.grid[3:], np.moveaxis(fm_jlk, 0, -1), bounds_error=False)
                WS_eff = WS_eff_ilk[i, m_lst]
                WD = np.broadcast_to(WD_l[m_lst, na], WS_eff.shape)
                deficit_jmk = np.moveaxis(
                    rgi_dst(np.array([WD.flatten(), WS_eff.flatten()]).T), -1, 0).reshape((J, M, K))
            else:
                x = [np.broadcast_to(v[:, m_lst], (J, M, K)).flatten()
                     for v in list(dst_xyh_jlk) + [WD_l[na, :, na], WS_eff_ilk[i, :][na]]]

                deficit_jmk = self.deficit_interp(np.array(x).T).reshape((J, M, K))
            np.nan_to_num(deficit_jmk, copy=False)
            deficit_jlk = item_assign(deficit_jlk, m_lst, deficit_jmk, axis=1)
        return deficit_jlk


class ExternalXRRelWindFarm(ExternalXRAbsWindFarm):
    @classmethod
    def generate(cls, name, grid_xyh, windFarmModel, wt_x, wt_y, type=0, **kwargs):
        sim_res = windFarmModel(wt_x, wt_y, type=type, **kwargs)
        X, Y, H = np.meshgrid(*grid_xyh, indexing='ij')
        dw, hcw, dh = X.flatten(), Y.flatten(), H.flatten()
        wf_x = np.mean([np.min(wt_x), np.max(wt_x)])
        wf_y = np.mean([np.min(wt_y), np.max(wt_y)])
        wf_h = windFarmModel.windTurbines.hub_height(type)

        theta = np.deg2rad(270 - sim_res.wd.values)
        co, si = np.cos(theta), np.sin(theta)
        x_jl = co[na] * dw[:, na] - hcw[:, na] * si[na] + wf_x
        y_jl = si[na] * dw[:, na] + hcw[:, na] * co[na] + wf_y
        h_jl = dh[:, na] + wf_h
        lw_j, WS_eff_jlk, TI_eff_jlk = windFarmModel._flow_map(x_jl, y_jl, h_jl, sim_res.localWind,
                                                               sim_res.wd.values, sim_res.ws, sim_res)
        deficit = (lw_j.WS_ilk - WS_eff_jlk).reshape(X.shape + (len(sim_res.wd), len(sim_res.ws)))

        ds = xr.Dataset({'deficit': (['x', 'y', 'h', 'wd', 'ws'], deficit)},
                        coords=dict(x=grid_xyh[0], y=grid_xyh[1], h=grid_xyh[2], wd=sim_res.wd, ws=sim_res.ws))

        from py_wake.site.streamline_distance import StreamlineDistance
        return cls(name, ds.sortby('wd'), windFarmModel.windTurbines, wt_x, wt_y,
                   relative_distance=not isinstance(windFarmModel.site.distance, StreamlineDistance))

    def __call__(self, i, l, deficit_jlk, WS_eff_ilk, WS_ilk,
                 dw_ijlk, hcw_ijlk, dh_ijlk,
                 WD_ilk, IJLK, dst_xyh_jlk, **_):
        I, J, L, K = IJLK
        WD_l = WD_ilk[np.minimum(i, len(WD_ilk) - 1), :, 0]
        m_lst = [m for m, wd in enumerate(WD_l) if l[m] and self.include_wd_func(wd)]
        if deficit_jlk is None:
            deficit_jlk = np.zeros((J, L, K))
        if not self.relative_distance:
            xyh_ilk = [np.reshape(v, (1, 1, 1)) for v in [self.wf_x, self.wf_y, self.wf_h]]
            dw_jlk, hcw_jlk, dh_jlk = [v[0] for v in StraightDistance()(*xyh_ilk, wd_l=WD_l, dst_xyh_jlk=dst_xyh_jlk)]
        else:
            dw_jlk, hcw_jlk, dh_jlk = dw_ijlk[i], hcw_ijlk[i], np.broadcast_to(dh_ijlk[i], (J, L, K))
        if m_lst:
            M = len(m_lst)
            WS_eff_ilk[i]

            x = [np.broadcast_to(v[:, m_lst], (J, M, K)).flatten()
                 for v in [dw_jlk, hcw_jlk, dh_jlk,
                           WD_l[na, :, na], WS_eff_ilk[i][na]]]

            deficit_jmk = self.deficit_interp(np.array(x).T).reshape((J, M, K))
            np.nan_to_num(deficit_jmk, copy=False)
            deficit_jlk = item_assign(deficit_jlk, m_lst, deficit_jmk, axis=1)
        return deficit_jlk
