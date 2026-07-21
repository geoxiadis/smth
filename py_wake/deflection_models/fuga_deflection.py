import warnings

from numpy import newaxis as na
from numpy.exceptions import ComplexWarning
from scipy.interpolate import RegularGridInterpolator as RGI

from py_wake import np
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.tests.test_files import tfp
from py_wake.utils.fuga_utils import FugaUtils
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils import gradients


class FugaDeflection(FugaUtils, DeflectionModel):

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc', on_mismatch='raise'):
        FugaUtils.__init__(self, path=LUT_path, on_mismatch=on_mismatch)
        if len(self.z) == 1:
            assert np.allclose(self.z, self.zHub)
            tabs = self.load_luts(['VL', 'VT']).reshape(2, -1, len(self.x))
        else:
            # interpolate to hub height
            jh = np.floor(np.log(self.zHub / self.z0) / self.ds)
            zlevels = [jh, jh + 1]
            tabs = self.load_luts(['VL', 'VT'], zlevels).reshape(2, 2, len(self.y), len(self.x))
            t = np.modf(np.log(self.zHub / self.z0) / self.ds)[0]
            tabs = tabs[:, 0] * (1 - t) + t * tabs[:, 1]

        VL, VT = tabs
        VL = -VL
        self.VL, self.VT = VL, VT

        nx0 = len(self.x) // 4
        ny = len(self.y)

        fL = np.cumsum(np.concatenate([np.zeros((ny, 1)), ((VL[:, :-1] + VL[:, 1:]) / 2)], 1), 1)
        fT = np.cumsum(np.concatenate([np.zeros((ny, 1)), ((VT[:, :-1] + VT[:, 1:]) / 2)], 1), 1)

        # subtract rotor center
        fL = (fL - fL[:, nx0:nx0 + 1]) * self.dx
        fT = (fT - fT[:, nx0:nx0 + 1]) * self.dx

        self.fLtab = fL = np.concatenate([-fL[::-1], fL[1:]], 0)
        self.fTtab = fT = np.concatenate([fT[::-1], fT[1:]], 0)
        self.fLT = GridInterpolator([self.x, self.mirror(self.y, anti_symmetric=True)],
                                    np.array([fL, fT]).T, bounds='limit')

    def calc_deflection(self, dw_ijlk, hcw_ijlk, dh_ijlk, WS_ilk, WS_eff_ilk, yaw_ilk, ct_ilk, D_src_il, **_):
        I, L, K = ct_ilk.shape
        X = int(np.max(D_src_il) * 3 / self.dy + 1)
        J = dw_ijlk.shape[1]

        WS_hub_ilk = WS_ilk

        theta_ilk = gradients.deg2rad(yaw_ilk)
        cos_ilk, sin_ilk = np.cos(theta_ilk), np.sin(theta_ilk)

        F_ilk = ct_ilk * (WS_eff_ilk)**2 / (WS_ilk * WS_hub_ilk)
        theta_ilk = np.broadcast_to(theta_ilk, F_ilk.shape)

        """
        For at given cross wind position in the lookup tables, yp, the deflection is lambda2p(yp), i.e.
        the real position (corresponding to the output hcw), is yp = y - lambda2p(yp) = y - lambda2(y),
        where y is the input hcw. I.e.:
        lambda(y) = lambda(yp + lp(yp)) = lp(yp)
        and
        yp = y - lambda(y)
        """

        if J < 1000:

            def get_err(deflected_hcw_ijlk, dw_ijlk):
                dw_ijlk = np.broadcast_to(dw_ijlk, deflected_hcw_ijlk.shape)
                fL, fT = self.fLT(np.array([dw_ijlk.flatten(), deflected_hcw_ijlk.flatten()]).T).T
                lambda_ijlk = F_ilk[:, na, :, :] * (fL.reshape(dw_ijlk.shape) * cos_ilk[:, na, :, :] +
                                                    fT.reshape(dw_ijlk.shape) * sin_ilk[:, na, :, :])

                return deflected_hcw_ijlk + lambda_ijlk - hcw_ijlk

            deflected_hcw_ijlk = hcw_ijlk
            D = D_src_il.max()
            # Newton Raphson
            complex = np.iscomplexobj(hcw_ijlk) or np.iscomplexobj(dw_ijlk)
            for i in range(8):
                if complex:
                    err = get_err(deflected_hcw_ijlk, dw_ijlk)
                    derr = (get_err(deflected_hcw_ijlk + .1) - err) / .1
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ComplexWarning)
                        cs_err = get_err(deflected_hcw_ijlk + 1e-20j, dw_ijlk)
                    err, derr = np.real(cs_err), np.imag(cs_err) / 1e-20

                step = -np.clip(err / derr, -D, D)  # limit step to 100m
                if np.allclose(step, 0, atol=1e-6):
                    break
                deflected_hcw_ijlk = deflected_hcw_ijlk + step
            hcw_ijlk = deflected_hcw_ijlk

        else:
            x, y = self.fLT.grid
            hcw_ijlk = np.array([self.get_hcw_jlk(i, K, L, x, y, dw_ijlk, hcw_ijlk, F_ilk, theta_ilk)
                                 for i in range(I)])

        return dw_ijlk, hcw_ijlk, dh_ijlk

    def get_hcw_jlk(self, i, K, L, x, y, dw_ijlk, hcw_ijlk, F_ilk, theta_ilk):
        return np.moveaxis([self.get_hcw_jk(i, l, K, x, y, dw_ijlk, hcw_ijlk, F_ilk, theta_ilk)
                            for l in range(L)], 2, 0)

    def get_hcw_jk(self, i, l, K, x, y, dw_ijlk, hcw_ijlk, F_ilk, theta_ilk):
        x_idx = (np.searchsorted(x, [dw_ijlk.min(), dw_ijlk.max()]) + np.array([-1, 1], dtype=int))
        m_x = len(x) + 1
        x_slice = slice(*np.minimum([m_x, m_x], np.maximum([0, 0], x_idx, dtype=int), dtype=int))

        y_idx = (np.searchsorted(y, [hcw_ijlk.min(), hcw_ijlk.max()]) + np.array([-20, 20], dtype=int))
        m_y = len(y) + 1
        y_slice = slice(*np.minimum([m_y, m_y], np.maximum([0, 0], y_idx, dtype=int), dtype=int))

        x_ = x[x_slice]
        y_ = y[y_slice]
        VLT = self.fLT.values[x_slice, y_slice]
        return [self.get_hcw_j(i, l, k, F_ilk, VLT, theta_ilk, x_, y_, hcw_ijlk, dw_ijlk) for k in range(K)]

    def get_hcw_j(self, i, l, k, F_ilk, VLT, theta_ilk, x_, y_, hcw_ijlk, dw_ijlk):
        lambda2p = F_ilk[i, l, k] * \
            np.sum(VLT * [np.cos(theta_ilk[i, l, k]), np.sin(theta_ilk[i, l, k])], -1)
        lambda2 = RGI(
            (x_, y_), np.array([np.interp(y_, y_ + l2p_x, l2p_x) for l2p_x in lambda2p], dtype=float))
        hcw_l = min(l, hcw_ijlk.shape[2] - 1)
        hcw_k = min(k, hcw_ijlk.shape[3] - 1)
        hcw_j = hcw_ijlk[i, :, hcw_l, hcw_k].copy()
        hcw_ijlk = hcw_ijlk[i, :, hcw_l, hcw_k]
        m = (hcw_ijlk > y_[0]) & (hcw_ijlk < y_[-1])
        hcw_j[m] -= lambda2((dw_ijlk[i, :, min(k, dw_ijlk.shape[2] - 1), min(k, dw_ijlk.shape[3] - 1)][m].real,
                             hcw_ijlk[m].real))
        return hcw_j


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt

        from py_wake import Fuga
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site

        site = IEA37Site(16)
        x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T
        windTurbines = IEA37_WindTurbines()
        path = tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc'
        noj = Fuga(path, site, windTurbines, deflectionModel=FugaDeflection(path))
        yaw = [-30, 30, 0]
        noj(x, y, yaw=yaw, wd=270, ws=10).flow_map().plot_wake_map()
        plt.show()


main()
