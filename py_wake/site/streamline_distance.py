import warnings

import matplotlib.pyplot as plt
from numpy import newaxis as na

from py_wake import np
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.flow_map import XYGrid
from py_wake.site.distance import StraightDistance
from py_wake.utils.streamline import VectorField3D


class StreamlineDistance(StraightDistance):
    """Just-In-Time Streamline Distance
    Calculates downwind crosswind and vertical distance along streamlines.
    Streamlines calculated in each call
    """

    def __init__(self, vectorField, step_size=20):
        """Parameters
        ----------
        vectorField : VectorField3d
        step_size : int for float
            Size of linear streamline steps
        """
        StraightDistance.__init__(self, wind_direction='wd')
        self.vectorField = vectorField
        self.step_size = step_size

    def __call__(self, src_x_ilk, src_y_ilk, src_h_ilk, time, wd_l=None, WD_ilk=None, dst_xyh_jlk=None):
        (src_x_ilk, src_y_ilk, src_h_ilk), (dst_x_jlk, dst_y_jlk, dst_h_jlk) = self.get_pos(
            src_x_ilk, src_y_ilk, src_h_ilk, wd_l, WD_ilk, dst_xyh_jlk)
        assert src_x_ilk.shape[2] == 1, 'StreamlineDistance does not support flowcase dependent positions'

        dw_ijlk, hcw_ijlk, dh_ijlk = StraightDistance.__call__(self, src_x_ilk, src_y_ilk, src_h_ilk, wd_l=wd_l,
                                                               dst_xyh_jlk=dst_xyh_jlk)
        I, J, L, K = dw_ijlk.shape
        src_z_ilk = self.site.elevation(src_x_ilk, src_y_ilk)
        dst_z_jlk = self.site.elevation(dst_x_jlk, dst_y_jlk)
        dz_ijlk = dst_z_jlk[na, :] - src_z_ilk[:, na]

        # +0 ~ autograd safe copy (broadcast_to returns readonly array)
        dh_ijlk = np.broadcast_to(dh_ijlk, dw_ijlk.shape) + 0.
        dz_ijlk = np.broadcast_to(dz_ijlk, dw_ijlk.shape) + 0.
        dw_mj, hcw_mj, dh_mj, dz_mj = [np.moveaxis(v, 1, 2).reshape(I * L, J)
                                       for v in [dw_ijlk, hcw_ijlk, dh_ijlk, dz_ijlk]]

        start_points_m = np.moveaxis([np.broadcast_to(v[:, :, 0], (I, L)).flatten()
                                     for v in [src_x_ilk, src_y_ilk, src_h_ilk]], 0, -1)
        wd_m = np.tile(wd_l, I)
        if time is not False:
            time = np.tile(time, I)

        stream_lines = self.vectorField.stream_lines(wd_m, time=time, start_points=start_points_m, dw_stop=dw_mj.max(1),
                                                     step_size=self.step_size)

        dxyz = np.diff(np.concatenate([stream_lines[:, :1], stream_lines], 1), 1, -2)
        length_is = np.cumsum(np.sqrt(np.sum(dxyz**2, -1)), -1)
        dist_xyz = stream_lines - start_points_m[:, na]
        t = np.deg2rad(270 - wd_m)[:, na]
        dw_is = dist_xyz[:, :, 0] * np.cos(t) + dist_xyz[:, :, 1] * np.sin(t)
        hcw_is = dist_xyz[:, :, 0] * np.sin(t) - dist_xyz[:, :, 1] * np.cos(t)

        for m, (dw_j, dw_s, hcw_s, dh_s, length_s) in enumerate(
                zip(dw_mj, dw_is, hcw_is, dist_xyz[:, :, 2], length_is)):
            dw = dw_j > 0
            hcw_mj[m, dw] += np.interp(dw_j[dw], dw_s, hcw_s)
            dh_mj[m, dw] -= np.interp(dw_j[dw], dw_s, dh_s)
            dw_mj[m, dw] = np.interp(dw_j[dw], dw_s, length_s)

        # streamline dh contains absolute height different, but pywake needs differences relative to ground, so
        # we need to subtract elevation differences, dz
        dh_mj += dz_mj

        return [np.moveaxis(v.reshape((I, L, J, 1)), 2, 1) for v in [dw_mj, hcw_mj, dh_mj]]


def main():
    if __name__ == '__main__':
        from py_wake.deficit_models.noj import NOJ
        from py_wake.examples.data.hornsrev1 import V80

        wt = V80()
        vf3d = VectorField3D.from_WaspGridSite(ParqueFicticioSite())
        site = ParqueFicticioSite(distance=StreamlineDistance(vf3d))

        x, y = site.initial_position[:].T
        wfm = NOJ(site, wt)
        wd = 330
        sim_res = wfm(x, y, wd=[wd], ws=10)
        fm = sim_res.flow_map(XYGrid(x=np.linspace(site.ds.x[0].item(), site.ds.x[-1].item(), 500),
                                     y=np.linspace(site.ds.y[0].item(), site.ds.y[-1].item(), 500)))
        stream_lines = vf3d.stream_lines(wd=np.full(x.shape, wd), start_points=np.array([x, y, np.full(x.shape, 70)]).T,
                                         dw_stop=y - 6504700)
        fm.plot_wake_map()
        for sl in stream_lines:
            plt.plot(sl[:, 0], sl[:, 1])

        plt.show()


main()
