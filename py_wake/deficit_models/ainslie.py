from py_wake.deficit_models.deficit_model import WakeDeficitModel
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake import np
from numpy import newaxis as na
try:
    from jDWM.Wake import StaticWake
except BaseException:
    raise ImportError('Please install jDWM: `pip install jdwm`')


class AinslieDeficit(WakeDeficitModel):

    def __init__(self, max_x=None, dx=0.1, max_r=5, nr=501, axial_induction_model="Joukowsky",
                 meandercompensator="Reinwardt", **kwargs):
        """Parameters
        ----------
        max_x : int or float
            Max longitudinal distance to compute [rotor radii]
        dx : int or float
            Longitudinal distance between computational points [rotor radii]
        max_r : int or float
            Max radial distance to compute [rotor radii]
        nr : int
            Radial discretisations (number of radial computation points)

        """
        WakeDeficitModel.__init__(self, rotorAvgModel=None, groundModel=None,
                                  use_effective_ws=True, use_effective_ti=False)
        self.dx = dx
        self.max_x = max_x
        self.max_r = max_r
        self.nr = nr
        self.axial_induction_model = axial_induction_model
        self.meandercompensator = meandercompensator
        self.kwargs = kwargs

        self.dwm = StaticWake(
            axial_induction_model=axial_induction_model,
            meandercompensator=meandercompensator,
            TI=None, U=None, D_rot=None,
            ** kwargs
        )

    def calc_deficit(self, dw_ijlk, cw_ijlk, ct_ilk, D_src_il, IJLK, **kwargs):
        WS_ref_ilk = kwargs[self.WS_key]
        TI_ref_ilk = kwargs[self.TI_key]
        I, J, L, K = IJLK
        WS_ref_ilk = np.broadcast_to(WS_ref_ilk, (I, L, K))
        TI_ref_ilk = np.broadcast_to(TI_ref_ilk, (I, L, K))
        J = dw_ijlk.shape[1]

        def run(dw_j, cw_j, WS, TI, ct, D):
            x_max = np.maximum(np.ceil(dw_j.max()), 2 * self.dx)
            r, x, U, V, widths, dUdr = self.dwm.solve(
                Nx=int(x_max / self.dx),  # Longitudinal discretisations
                x_max=x_max,  # Max downstream distance to compute (in rotor radii)
                Nr=self.nr,  # Radial discretisations
                r_max=self.max_r,  # Max radial distance to compute (in rotor radii)
                ct=np.maximum(ct, 1e-20),  # Thrust coefficient of wake-producing turbine
                TI=TI,  # Ambient/meandering turbulence intensity
                U=WS,  # Ambient wind speed (in m/s) (required for meandering compensation)
                D_rot=D,  # Rotor diameter of turbine (in m) (required for meandering compensation)
            )
            return GridInterpolator([x, r], U)(np.array([dw_j, cw_j]).T)
        R_src_ijlk = D_src_il[:, na, :, na] / 2

        def v2r_lim(v_ijlk, min, max):
            return np.broadcast_to(np.maximum(np.minimum(v_ijlk / R_src_ijlk, max), min), (I, J, L, K))
        r_dw_ijlk = v2r_lim(dw_ijlk, 0, self.max_x or 1e100)
        r_cw_ijlk = v2r_lim(cw_ijlk, -self.max_r, self.max_r)
        deficit_ijlk = WS_ref_ilk[:, na] * (1 - np.moveaxis(
            [[[run(r_dw_ijlk[i, :, l, k], r_cw_ijlk[i, :, l, k], WS_ref_ilk[i, l, k], TI_ref_ilk[i, l, k],
                   ct_ilk[i, l, k], np.broadcast_to(D_src_il, (I, L))[i, l])
               for k in range(K)]
              for l in range(L)]
             for i in range(I)], 3, 1))
        if self.max_x:
            # set deficit downstream of self.max_x to 0
            deficit_ijlk[np.broadcast_to(dw_ijlk, deficit_ijlk.shape) > self.max_x * (D_src_il / 2)[:, na, :, na]] = 0
        return deficit_ijlk

    def wake_radius(self, dw_ijlk, D_src_il, **_):
        return np.broadcast_to(D_src_il[:, na, :, na], dw_ijlk.shape)


class AinslieDemoDeficit(AinslieDeficit):
    def __init__(self, max_x=None, dx=0.1, max_r=5, nr=501, axial_induction_model="Joukowsky", ct=.89, tsr=7,
                 meandercompensator="Reinwardt", **kwargs):
        AinslieDeficit.__init__(self, max_x=max_x, dx=dx, max_r=max_r, nr=nr,
                                axial_induction_model=axial_induction_model, meandercompensator=meandercompensator,
                                ct=ct, tsr=tsr, **kwargs)
