
from numpy import newaxis as na

from py_wake import np
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.site._site import UniformSite
from py_wake.flow_map import XYGrid, Points
from py_wake.deficit_models.deprecated.noj import NOJDeficit


class Jensen_1983_TophatDeficit(NOJDeficit):
    """
    Jensen, N. O. (1983). A note on wind generator interaction. Risø National Laboratory. Risø-M No. 2411
    https://backend.orbit.dtu.dk/ws/portalfiles/portal/55857682/ris_m_2411.pdf

    Jensen uses:
    - 2/3 as 2*a
    - ambient wind speed to be be conservative
    """

    def __init__(self, k=.1):
        NOJDeficit.__init__(self, k=k, ct2a=lambda ct_ilk: np.full(ct_ilk.shape, 1 / 3))

    def center_wake_factor(self, D_src_il, wake_radius_ijlk, **_):
        # 2/3*(r0 / (r0 + alpha*x)**2
        return np.where(wake_radius_ijlk > 0, 2 / 3 * ((D_src_il[:, na, :, na] / 2) / wake_radius_ijlk)**2, 0)


class Jensen_1983_BellshapeDeficit(Jensen_1983_TophatDeficit):
    def shape_factor(self, dw_ijlk, cw_ijlk, wake_radius_ijlk, **_):
        theta = np.arctan(cw_ijlk.copy() / dw_ijlk)

        return np.where(theta < np.deg2rad(20), (1 + np.cos(9 * theta)) / 2, 0)


def NibeA():
    return WindTurbine(name='Nibe-A', diameter=40,
                       hub_height=50,
                       # power and ct undefined
                       powerCtFunction=PowerCtFunction(['ws'], lambda ws, run_only: ws * 0, 'w'))


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        wt = NibeA()
        wfm = PropagateDownwind(site=UniformSite(ws=8.1), windTurbines=wt,
                                wake_deficitModel=Jensen_1983_TophatDeficit())
        sim_res = wfm([0], [0], wd=270)
        fm = sim_res.flow_map(XYGrid(x=[40, 100], y=0))
        print(fm)

        sim_res = wfm([0], [0], wd=270, ws=1)
        wfm_bell = PropagateDownwind(site=UniformSite(ws=8.1), windTurbines=wt,
                                     wake_deficitModel=Jensen_1983_BellshapeDeficit())
        sim_res_bell = wfm_bell([0], [0], wd=270, ws=1)
        angle = np.arange(-30, 30)
        theta_ = np.deg2rad(angle)
        R = wt.diameter() / 2
        axes = plt.subplots(3, 1)[1]
        for ax, r in zip(axes, [16, 10, 6]):
            y = np.sin(theta_) * r * R
            x = y * 0 + r * R
            fm = sim_res.flow_map(Points(x=x, y=y, h=x * 0 + wt.hub_height()))
            ax.plot(angle, fm.WS_eff)
            fm = sim_res_bell.flow_map(Points(x=x, y=y, h=x * 0 + wt.hub_height()))
            ax.plot(angle, fm.WS_eff)
            ax.set_title(r)
        plt.show()


main()
