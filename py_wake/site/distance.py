from py_wake import np
from numpy import newaxis as na
import matplotlib
from py_wake.utils.functions import mean_deg
from py_wake.utils import gradients
from py_wake.utils.gradients import rad2deg, deg2rad


class StraightDistance():

    def __init__(self, wind_direction='wd'):
        """
        Parameters
        ----------
        wind_direction : {'wd','WD_i'}
            'wd': The reference wind direction, wd, is used to calculate downwind and horizontal crooswind distances
            'WD_i': The wind direction at the current (upstream) wind turbine is used to calculate downwind and
            horizontal crossswind distances.

        """
        assert wind_direction in ['wd', 'WD_i'], "'StraightDistance.wind_direction must be 'wd' or 'WD_i'"
        self.wind_direction = wind_direction

    def _cos_sin(self, wd):
        theta = gradients.deg2rad(90 - wd)
        cos = np.cos(theta)
        sin = np.sin(theta)
        return cos, sin

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()
                if k not in {'src_x_ilk', 'src_y_ilk', 'src_h_ilk', 'dst_x_j', 'dst_y_j', 'dst_h_j',
                             'dx_iilk', 'dy_iilk', 'dh_iilk', 'dx_ijlk', 'dy_ijlk', 'dh_ij', 'src_eq_dst'}}

    def get_pos(self, src_x_ilk, src_y_ilk, src_h_ilk, wd_l=None, WD_ilk=None, dst_xyh_jlk=None):
        # ensure 3d and
        # +.0 ensures float or complex
        src_x_ilk, src_y_ilk, src_h_ilk = [np.expand_dims(v, tuple(range(len(np.shape(v)), 3))) + .0
                                           for v in [src_x_ilk, src_y_ilk, src_h_ilk]]

        if dst_xyh_jlk is None:
            dst_x_jlk, dst_y_jlk, dst_h_jlk = src_x_ilk, src_y_ilk, src_h_ilk
        else:
            assert len(dst_xyh_jlk) == 3
            dst_x_jlk, dst_y_jlk, dst_h_jlk = [np.expand_dims(v, tuple(range(len(np.shape(v)), 3))) + .0
                                               for v in dst_xyh_jlk]
        return (src_x_ilk, src_y_ilk, src_h_ilk), (dst_x_jlk, dst_y_jlk, dst_h_jlk)

    def __call__(self, src_x_ilk, src_y_ilk, src_h_ilk, wd_l=None, WD_ilk=None, time=None, dst_xyh_jlk=None):
        (src_x_ilk, src_y_ilk, src_h_ilk), (dst_x_jlk, dst_y_jlk, dst_h_jlk) = self.get_pos(
            src_x_ilk, src_y_ilk, src_h_ilk, wd_l, WD_ilk, dst_xyh_jlk)
        wd_l = np.asarray(wd_l)
        if self.wind_direction == 'wd':
            assert wd_l is not None, "wd_l must be specified when Distance.wind_direction='wd'"
            WD_ilk = np.asarray(wd_l)[na, :, na]
        else:
            assert WD_ilk is not None, "WD_ilk must be specified when Distance.wind_direction='WD_i'"

        cos_ilk, sin_ilk = self._cos_sin(WD_ilk)
        if self.wind_direction == 'wd':

            src_dw_ilk = -cos_ilk * src_x_ilk - sin_ilk * src_y_ilk
            src_hcw_ilk = sin_ilk * src_x_ilk - cos_ilk * src_y_ilk

            if dst_xyh_jlk is None:
                dst_dw_jlk, dst_hcw_jlk, dst_h_jlk = src_dw_ilk, src_hcw_ilk, src_h_ilk
            else:
                cos_jlk, sin_jlk = self._cos_sin(wd_l[na, :, na])
                dst_dw_jlk = (-cos_jlk * dst_x_jlk - sin_jlk * dst_y_jlk)
                dst_hcw_jlk = (sin_jlk * dst_x_jlk - cos_jlk * dst_y_jlk)
            dw_ijlk = dst_dw_jlk[na] - src_dw_ilk[:, na]
            hcw_ijlk = dst_hcw_jlk[na] - src_hcw_ilk[:, na]
        else:
            dx_ijlk = dst_x_jlk[na] - src_x_ilk[:, na]
            dy_ijlk = dst_y_jlk[na] - src_y_ilk[:, na]

            cos_ilk, sin_ilk = self._cos_sin(WD_ilk)
            dw_ijlk = -cos_ilk[:, na] * dx_ijlk - sin_ilk[:, na] * dy_ijlk
            hcw_ijlk = sin_ilk[:, na] * dx_ijlk - cos_ilk[:, na] * dy_ijlk

        dh_ijlk = dst_h_jlk[na, :] - src_h_ilk[:, na]

        return dw_ijlk, hcw_ijlk, np.broadcast_to(dh_ijlk, hcw_ijlk.shape)

    def dw_order_indices(self, src_x_ilk, src_y_ilk, wd_l):
        WD_ilk = np.asarray(wd_l)[na, :, na]
        src_x_ilk, src_y_ilk = [np.expand_dims(v, tuple(range(len(np.shape(v)), 3))) + .0
                                for v in [src_x_ilk, src_y_ilk]]

        cos_ilk, sin_ilk = self._cos_sin(WD_ilk)

        src_dw_ilk = -cos_ilk * src_x_ilk - sin_ilk * src_y_ilk
        return np.moveaxis(np.argsort(src_dw_ilk, 0), 0, -1)

    def plot(self, src_x_ilk, src_y_ilk, src_h_ilk, wd_l=None, WD_ilk=None, dst_xyh_jlk=None):
        import matplotlib.pyplot as plt
        (src_x_ilk, src_y_ilk, src_h_ilk), (dst_x_jlk, dst_y_jlk, dst_h_jlk) = self.get_pos(
            src_x_ilk, src_y_ilk, src_h_ilk, wd_l, WD_ilk, dst_xyh_jlk)

        dw_ijlk, hcw_ijlk, _ = self(src_x_ilk, src_y_ilk, src_h_ilk, WD_ilk=WD_ilk, wd_l=wd_l, dst_xyh_jlk=dst_xyh_jlk)
        if self.wind_direction == 'wd':
            WD_ilk = np.asarray(wd_l)[na, :, na]

        wdirs = mean_deg(WD_ilk, (0, 2))
        for l, wd in enumerate(wdirs):
            plt.figure()
            ax = plt.gca()
            theta = np.deg2rad(90 - wd)
            ax.set_title(wd)
            ax.arrow(0, 0, -np.cos(theta) * 20, -np.sin(theta) * 20, width=1)
            colors = [c['color'] for c in iter(matplotlib.rcParams['axes.prop_cycle'])]
            f = 2
            for i, x_, y_ in zip(np.arange(len(src_x_ilk)), src_x_ilk[:, 0, 0], src_y_ilk[:, 0, 0]):
                c = colors[i % len(colors)]
                ax.plot(x_, y_, '2', color=c, ms=10, mew=3)
                for j, dst_x, dst_y in zip(np.arange(len(dst_x_jlk)),
                                           dst_x_jlk[:, 0, 0], dst_y_jlk[:, 0, 0]):
                    ax.arrow(x_ - j / f, y_ - j / f, -np.cos(theta) * dw_ijlk[i, j, l, 0], -
                             np.sin(theta) * dw_ijlk[i, j, l, 0], width=.3, color=c)
                    ax.plot([dst_x - i / f, dst_x - np.sin(theta) * hcw_ijlk[i, j, l, 0] - i / f],
                            [dst_y - i / f, dst_y + np.cos(theta) * hcw_ijlk[i, j, l, 0] - i / f], '--', color=c)
            plt.plot(src_x_ilk[:, 0, 0], src_y_ilk[:, 0, 0], 'k2')
            plt.plot(dst_x_jlk[:, 0, 0], dst_y_jlk[:, 0, 0], 'k2')
            ax.axis('equal')


class TerrainFollowingDistance(StraightDistance):
    def __init__(self, distance_resolution=1000, wind_direction='wd', **kwargs):
        super().__init__(wind_direction=wind_direction, **kwargs)
        self.distance_resolution = distance_resolution

    def __call__(self, src_x_ilk, src_y_ilk, src_h_ilk,
                 WD_ilk=None, wd_l=None, time=None, dst_xyh_jlk=None):
        # StraightDistance.setup(self, src_x_ilk, src_y_ilk, src_h_ilk, src_z_ilk, dst_xyh_jlk=dst_xyh_jlk)
        # if len(src_x_ilk) == 0:
        #     return
        # Calculate distance between src and dst and project to the down wind direction

        (src_x_ilk, src_y_ilk, src_h_ilk), (dst_x_jlk, dst_y_jlk, dst_h_jlk) = self.get_pos(
            src_x_ilk, src_y_ilk, src_h_ilk, wd_l, WD_ilk, dst_xyh_jlk)
        dw_ijlk, hcw_ijlk, dh_ijlk = StraightDistance.__call__(self, src_x_ilk, src_y_ilk, src_h_ilk,
                                                               WD_ilk=WD_ilk, wd_l=wd_l, dst_xyh_jlk=dst_xyh_jlk)

        assert src_x_ilk.shape[2] == 1, 'TerrainFollowingDistance does not support flowcase dependent positions'

        src_x_i, src_y_i = src_x_ilk[:, 0, 0], src_y_ilk[:, 0, 0]

        dst_x_j, dst_y_j = dst_x_jlk[:, 0, 0], dst_y_jlk[:, 0, 0]

        # Generate interpolation lines
        xy = np.array([(np.linspace(src_x, dst_x, self.distance_resolution),
                        np.linspace(src_y, dst_y, self.distance_resolution))
                       for src_x, src_y in zip(src_x_i, src_y_i)
                       for dst_x, dst_y in zip(dst_x_j, dst_y_j)])
        theta_ij = gradients.arctan2(dst_y_j[na, :, ] - src_y_i[:, na],
                                     dst_x_j[na, :] - src_x_i[:, na])
        x, y = xy[:, 0], xy[:, 1]

        # find height along interpolation line
        h = self.site.elevation(x.flatten(), y.flatten()).reshape(x.shape)
        # calculate horizontal and vertical distance between interpolation points
        dxy = np.sqrt((x[:, 1] - x[:, 0])**2 + (y[:, 1] - y[:, 0])**2)
        dh = np.diff(h, 1, 1)
        # calculate distance along terrain following interpolation lines
        s = np.sum(np.sqrt(dxy[:, na]**2 + dh**2), 1)

        d_ij = s.reshape(dw_ijlk.shape[:2])

        # project terrain following distance between wts onto downwind direction
        # instead of projecting the distances onto first x,y and then onto down wind direction
        # we offset the wind direction by the direction between source and destination

        if self.wind_direction == 'wd':
            WD_ilk = np.asarray(wd_l)[na, :, na]

        WD_il = mean_deg(WD_ilk, 2)
        dir_ij = 90 - rad2deg(theta_ij)
        wdir_offset_ijl = np.asarray(WD_il)[:, na] - dir_ij[:, :, na]
        theta_ijl = deg2rad(90 - wdir_offset_ijl)
        dw_ijlk = (- np.sin(theta_ijl) * d_ij[:, :, na])[..., na]

        return dw_ijlk, hcw_ijlk, dh_ijlk
