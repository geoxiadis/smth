"""
Created on 24/03/2023

Description: Implementation of the Minimalistic Prediction Model developped by
    Jens N. Sørensen and Gunner C. Larsen

@author: David Fournely and Ariadna Garcia Montes

simplified and generalized by Mads M Pedersen
"""
import os
import warnings

from scipy.special import gamma, gammainc

from py_wake import np
from py_wake.tests import npt
from py_wake.utils import fuga_utils, weibull
from py_wake.utils.layouts import farm_area
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines._wind_turbines import WindTurbines


class Larsen_etal2026(WindFarmModel):
    def __init__(self, site, windTurbines, latitude, max_cp=None, ws_cutin=None, ws_cutout=None, rho=1.225,
                 Astar=2, Bstar=.85, kappa=0.41):
        """Minimalistic wind farm model

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        correction_factor : int, float or function
            Finite-size wind farm corrrection which multiplied with sqrt(Nturb) gives
            the number of wind turbines exposed to the free wind
        latitude : int or float
            latitude [deg] used to calculate the coriolis parameter
        max_cp : float, optional
            Wind turbine power coefficient. Must be specified or exist in the windTurbine
        ws_cutin : int or float, optional
            Wind turbine cut-in wind speed. Defaults to 4 if not specified and not present in the windTurbine
        ws_cutout : int or float, optional
            Wind turbine cut-out wind speed. Defaults to 25 if not specified and not present in the windTurbine
        """
        WindFarmModel.__init__(self, site, windTurbines)
        self.externalWindFarms = []
        self.rho = rho
        self.CP = max_cp or windTurbines.max_cp
        self.Uin = ws_cutin or getattr(windTurbines, 'ws_cutin', 4)
        self.Uout = ws_cutout or getattr(windTurbines, 'ws_cutout', 25)
        omega = 2 * np.pi / (24 * 60 * 60)  # earth rotation speed
        self.f = 2 * omega * np.sin(np.deg2rad(latitude))
        self.Astar = Astar
        self.Bstar = Bstar
        self.kappa = kappa

    def get_a_func(self, N, S, U_rated=11):
        xi_lst_below, xi_lst_above = self.get_xi_lst(N, S)

        def a_func(ws):
            a = np.ones_like(ws, dtype=float)
            m = ws < U_rated
            if np.any(m):
                a[m] = np.sum([(ws[m] / U_rated)**pi * exp for pi, exp in xi_lst_below], 0)

            m = (ws >= U_rated) & (ws <= 20)
            if np.any(m):
                a[m] = np.sum([(ws[m] / U_rated)**exp * xi for exp, xi in xi_lst_above], 0)
            return a
        return a_func

    def calc_wt_interaction(self, x_ilk, y_ilk, h_i=None, type_i=0,
                            wd=None, ws=None, time=False,
                            n_cpu=1, wd_chunks=None, ws_chunks=None, **kwargs):

        rated_power = self.windTurbines.power(np.array([8, 12, 16])).max()
        ct_max = self.windTurbines.ct(np.array([6, 8, 12, 16])).max()
        area = farm_area(wt_x=x_ilk.mean((1, 2)), wt_y=y_ilk.mean((1, 2)))

        # Create LocalWind_omni with only one wind speed and wind direction
        localWind_omni = self.site.local_wind(x_ilk, y_ilk, h_i, wd=0, ws=10)
        localWind_omni['P_ilk'][:] = 1

        TI_ilk = kwargs.get('TI_ilk', localWind_omni.get('TI_ilk'))
        z0 = fuga_utils.z0(np.mean(TI_ilk), zref=np.mean(self.windTurbines.hub_height()), zeta0=0)[0]

        power_sector, ws_eff_sector = np.array([
            self._calculate_power_ws(Pg=rated_power,
                                     CT=ct_max,
                                     D=self.windTurbines.diameter(),
                                     H=self.windTurbines.hub_height(),
                                     z0=z0,
                                     Aw=A_w,
                                     kw=k_w,
                                     Nturb=len(x_ilk),
                                     Area=area)
            for A_w, k_w in zip(self.site.ds.Weibull_A.values[:-1], self.site.ds.Weibull_k.values[:-1])]).T

        f = self.site.ds.Sector_frequency.values[:-1]
        power = np.sum(power_sector * f)
        ws_eff = np.sum(f * ws_eff_sector)

        I, L, K = len(x_ilk), 1, 1
        WS_eff_ilk = np.full((I, L, K), ws_eff)
        power_ilk = np.full((I, L, K), power / len(x_ilk))
        TI_eff_ilk = localWind_omni['TI_ilk']
        ct_ilk = np.full((I, L, K), ct_max)
        kwargs_ilk = {'type_i': type_i, **kwargs}

        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind_omni, kwargs_ilk

    def get_args(self, Area, Nturb, D, Pg):
        # Mean spacing between wt in diameters, eq 8
        S = np.sqrt(Area) / (D * (np.sqrt(Nturb) - 1))

        # Rated wind speed [m/s], eq 4
        Ur = (8 * Pg / (self.rho * np.pi * D**2 * self.CP))**(1 / 3)

        # Power modeled as P = alpha * U^3 + beta, eq 1
        alpha = Pg / (Ur**3 - self.Uin**3)  # [(m/s)^-3] eq 2
        beta = -Pg * self.Uin**3 / (Ur**3 - self.Uin**3)  # [-], eq 2

        return S, Ur, alpha, beta

    def get_gam_delta_G(self, H, z0, Aw, kw):
        # factor defined by Frandsen, should be used instead of f in eq 13 and 19 (typos in paper)
        fm = self.f * np.exp(self.Astar)
        delta = np.log(H / z0)  # eq 19
        Uh0 = gamma(1 + 1 / kw) * Aw

        # Geostrophic wind speed
        G_last = Uh0
        for n in range(10):
            G = Uh0 * (1 + np.log(G_last / (fm * H)) / delta)
            dG = abs(G - G_last)
            if dG < 1e-5:
                break
            G_last = G

        gam = np.log(G / (fm * H))  # eq 19
        return gam, delta, G

    def get_eps_function(self, S, CT_rated, H, z0, Ur, Aw, kw):

        gam, delta, G = self.get_gam_delta_G(H, z0, Aw, kw)
        kappa = self.kappa

        def eps(Uh):
            # Uh = wind speed inside infinite farm
            # eq 18. The paper states 3/2 instead of 3.2 which is a typo
            # eq 18 should be a function epsilon2(Uh)= ..., as Uh is either Ur (=eps1) or Uout
            # here Uh is replaced with Uf to avoid confusion with Uh(mean ws at hub height)

            # CT = CT_rated                , Uin<U<Ur
            #      CT_rated * (Ur/U)^3.2   , Ur<U<Uout
            CT = CT_rated * (Ur / np.maximum(Uh, Ur))**3.2
            Ctau = np.pi * CT / (8 * S * S)  # [-] Wake parameter, rotor ct smeared on WT area
            return (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau + (kappa / delta)**2))
        return eps

    def get_eps_from_Uh0(self, S, CT_rated, H, z0, Ur, Uh0, Aw, kw):
        gam, delta, G = self.get_gam_delta_G(H, z0, Aw, kw)
        kappa = self.kappa
        eps = 1
        Uh = 0
        while abs(Uh - Uh0 * eps) > 1e-6:
            Uh = Uh0 * eps
            CT = CT_rated * (Ur / np.maximum(Uh, Ur))**3.2
            Ctau = np.pi * CT / (8 * S * S)
            eps = (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau + (kappa / delta)**2))
        return eps

    def get_z0(self, H, Aw, kw):
        ws_lst = np.arange(5, 26)
        TI_lst = np.array([0.0776, 0.07334, 0.07109, 0.07011, 0.07001, 0.07051, 0.07146, 0.07274, 0.07425, 0.07593, 0.0777,
                           0.07952, 0.08134, 0.08311, 0.08479, 0.08635, 0.08775, 0.08896, 0.08995, 0.09069, 0.09117])
        w = weibull.cdf(ws_lst + .5, Aw, kw) - weibull.cdf(ws_lst - .5, Aw, kw)
        TI_mean = np.sum(w * TI_lst) / w.sum()
        kappa = 0.41
        return H / np.exp(2.39 * kappa / TI_mean)

    def get_Py(self, Aw, kw, eps_func, alpha, beta, Pg, Ur,
               xi_lst_below_rated=[(0, 1)], xi_lst_above_rated=[(0, 1)]):
        """General version

        Free-stream power when eps_func = lambda _ : 1
        xi_lst_below_rated and xi_lst_above_rated used in Larsen et al. 2026. Has no effect when = [(0,1)]
        """

        Uin = self.Uin
        Uout = self.Uout
        eps1 = eps_func(Ur)

        def gamma_int(s, x1, x2):
            # calculates int_x1^x2 t^(s-1) * exp(-t)
            # for s=1, and x = (U/A)^k, this corresponds to the weibull CDF
            return (gamma(s) *  # cancel out normalization in scipy's gammainc
                    (gammainc(s, x2) - gammainc(s, x1)))

        def weibull_int(pi, U1, U2):
            return gamma_int(s=(pi + kw) / kw, x1=(U1 / Aw)**kw, x2=(U2 / Aw)**kw)

        # integration based on Uh (wind speed inside infinite farm)
        # Uh = [Uh=Uin ... Uh0=Ur]
        e1 = alpha * np.sum([xi * eps1**3 * Aw**(pi + 3) * weibull_int(pi + 3, Uin /
                            eps1, Ur * eps1 / eps1) for pi, xi in xi_lst_below_rated])
        e2 = beta * np.sum([xi * Aw**pi * weibull_int(pi, Uin / eps1, Ur * eps1 / eps1)
                           for pi, xi in xi_lst_below_rated])
        # Uh = [Uh0=Ur ... Uh=Ur]
        e3 = alpha * np.sum([xi * eps1**3 * Aw**(pi + 3) * weibull_int(pi + 3, Ur *
                            eps1 / eps1, Ur / eps1) for pi, xi in xi_lst_above_rated])
        e4 = beta * np.sum([xi * Aw**pi * weibull_int(pi, Ur * eps1 / eps1, Ur / eps1)
                           for pi, xi in xi_lst_above_rated])
        # Uh = [Uh=Ur ... Uh0=20]
        e5 = Pg * np.sum([xi * Aw**pi * weibull_int(pi, Ur / eps1, 20 * eps_func(20) / eps_func(20))
                         for pi, xi in xi_lst_above_rated])
        # Uh = [Uh0=20 ... Uh=Uout]
        e6 = Pg * weibull_int(0, 20 * eps_func(20) / eps_func(20), Uout / eps_func(Uout))

        return np.array([e1, e2, e3, e4, e5, e6])

    def get_xi_lst(self, N, S):
        assert 5 <= N <= 50, f'Model should not be used outside calibration interval of N=[5,50], but is used with N={N}'
        assert 3 <= S <= 13, f'Model should not be used outside calibration interval of S=[3,13], but is used with S={S}'
        sqrtNT = N
        N_range = 45.0
        N_offset = 55.0
        S_range = 10.0
        S_offset = 16.0
        n_T = (sqrtNT * 2 - N_offset) / N_range
        s = (2 * S - S_offset) / S_range

        def P_2(x):
            return 1 / 2 * (3 * x**2 - 1)

        xi_lst_below = [(0, ((+0.500 - 0.140 * n_T + 0.100 * P_2(n_T)) +
                             (+0.130 - 0.100 * n_T - 0.053 * P_2(n_T)) * s +
                             (-0.005 + 0.139 * n_T + 0.024 * P_2(n_T)) * P_2(s))),
                        (1, ((+0.034 - 0.051 * n_T - 0.023 * P_2(n_T)) +
                             (+0.044 + 0.157 * n_T + 0.028 * P_2(n_T)) * s +
                             (-0.055 - 0.146 * n_T - 0.014 * P_2(n_T)) * P_2(s))),
                        ]
        xi_lst_above = [(0, ((-7.363 - 1.495 * n_T) +
                             (-4.151 - 6.027 * n_T) * s +
                             (+4.422 + 3.572 * n_T) * P_2(s))),
                        (1, ((+15.767 + 2.126 * n_T) +
                             (+10.313 + 13.335 * n_T) * s +
                             (-9.789 - 7.224 * n_T) * P_2(s))),
                        (2, ((-9.909 - 0.950 * n_T) +
                             (-7.863 - 9.413 * n_T) * s +
                             (+6.891 + 4.668 * n_T) * P_2(s))),
                        (3, ((+2.074 + 0.128 * n_T) +
                             (+1.900 + 2.149 * n_T) * s +
                             (-1.566 - 0.975 * n_T) * P_2(s))),
                        ]
        return xi_lst_below, xi_lst_above

    def _calculate_power_ws(self, Pg, CT, D, H, z0, Aw, kw, Nturb, Area):
        """
        Inputs:
            Pg    - [W] Nameplate capacity (generator power)
            CT    - [-] Thrust coefficient
            D     - [m] Rotor diameter
            H     - [m] Tower height
            z0    - [m] roughness length
            Aw    - [m/s] Weibull scale parameter
            kw    - [-] Weibull shape parameter
            Nturb - [-] Number of turbines
            Area  - [m2] Area of wind farm

        Outputs:
            power - [Wh] Annual energy production of the wind farm
            ws_eff - [m/s] Effective mean wind speed including wakes - not accessible from this model set to 0
        """
        S, Ur, alpha, beta = self.get_args(Area, Nturb, D, Pg)
        z0 = self.get_z0(H, Aw, kw)
        eps_func = self.get_eps_function(S, CT, H, z0, Ur, Aw, kw)

        # eq 21
        # a: fraction of free (solitair) wt
        # P = N_T (a*P_s + (1-a)*P_if) = N_T (a*P_s - a*P_if + P_if)
        xi_lst_below_rated, xi_lst_above_rated = [[(pi, xi / Ur**pi) for pi, xi in xi_lst]
                                                  for xi_lst in self.get_xi_lst(N=np.sqrt(Nturb), S=S)]
        P_if = self.get_Py(Aw, kw, eps_func, alpha, beta, Pg, Ur).sum()
        aP_if = self.get_Py(Aw, kw, eps_func, alpha, beta, Pg, Ur, xi_lst_below_rated, xi_lst_above_rated).sum()
        aP_s = self.get_Py(Aw, kw, lambda _: 1, alpha, beta, Pg, Ur, xi_lst_below_rated, xi_lst_above_rated).sum()

        power = Nturb * (aP_s - aP_if + P_if)
        ws_eff = power * 0
        return power, ws_eff


class Sorensen_Larsen_2021(Larsen_etal2026):
    """Sørensen, J.N.; Larsen, G.C.
    A Minimalistic Prediction Model to Determine Energy Production and Costs of Offshore Wind Farms.
    Energies 2021, 14, 448. https://doi.org/10.3390/en14020448"""

    def __init__(self, site, windTurbines, max_cp=None,
                 ws_cutin=None, ws_cutout=None, correction_factor=3, rho=1.25):
        Larsen_etal2026.__init__(self, site, windTurbines, latitude=55, max_cp=max_cp,
                                 ws_cutin=ws_cutin, ws_cutout=ws_cutout, rho=rho)
        self.Astar = 4
        self.Bstar = 1
        self.kappa = 0.4
        self.correction_factor = correction_factor
        self.f = 1.2 * 10**(-4)

    def get_eps_function(self, S, CT_rated, H, z0, Aw, kw, Ur):
        return Larsen_etal2026.get_eps_function(self, S, CT_rated, H, z0, Ur, Aw, kw)

    def _calculate_power_ws(self, Pg, CT, D, H, z0, Aw, kw, Nturb, Area):
        """
        Inputs:
            Pg    - [W] Nameplate capacity (generator power)
            CT    - [-] Thrust coefficient
            D     - [m] Rotor diameter
            H     - [m] Tower height
            z0    - [m] roughness length
            Aw    - [m/s] Weibull scale parameter
            kw    - [-] Weibull shape parameter
            Nturb - [-] Number of turbines
            Area  - [m2] Area of wind farm

        Outputs:
            power - [Wh] Annual energy production of the wind farm
            ws_eff - [m/s] Effective mean wind speed including wakes
        """

        S, Ur, alpha, beta = self.get_args(Area, Nturb, D, Pg)

        Uh0 = Aw * gamma(1 + 1 / kw)  # [m/s] Mean velocity at hub height

        # Finite-size wind farm corrrection, section 2.5
        correction_factor = self.correction_factor
        if hasattr(correction_factor, '__call__'):
            correction_factor = correction_factor(Uh0, S, Nturb)
        Nfree = correction_factor * np.sqrt(Nturb)  # Number of wt exposed to the free wind

        P_y = self.get_Py(Aw, kw, lambda _: 1, alpha, beta, Pg, Ur).sum()

        eps_func = self.get_eps_function(S, CT, H, z0, Aw, kw, Ur)

        gam, delta, G = self.get_gam_delta_G(H, z0, Aw, kw)
        # Mean velocity at hub height without wake effects from geostrophic wind
        Uh0 = G / (1 + gam / self.kappa * np.sqrt((self.kappa / delta)**2))  # eq 13, ct=0
        Uh = Uh0 * self.get_eps_from_Uh0(S, CT, H, z0, Ur, Uh0, Aw, kw)

        # Power production with wake effects
        P_WFy = self.get_Py(Aw, kw, eps_func, alpha, beta, Pg, Ur).sum()

        power = ((Nturb - Nfree) * P_WFy + Nfree * P_y)
        ws_eff = ((Nturb - Nfree) * Uh + Nfree * Uh0) / Nturb
        return power, ws_eff


class Sorensen_etal_2024(Sorensen_Larsen_2021):
    """Sørensen, J.N.; Larsen, G.C.
    A Minimalistic Prediction Model to Determine Energy Production and Costs of Offshore Wind Farms.
    Energies 2021, 14, 448. https://doi.org/10.3390/en14020448

    Jens N. Sørensen et al 2024 J. Phys.: Conf. Ser. 2767 092022, DOI 10.1088/1742-6596/2767/9/092022

    """

    def __init__(self, site, windTurbines, max_cp=None, ws_cutin=None, ws_cutout=None, rho=1.25):

        import joblib
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Trying to unpickle estimator LinearRegression from version')
            warnings.filterwarnings('ignore', 'Trying to unpickle estimator PolynomialFeatures from version')
            self.PolynomialFeature_model = joblib.load(os.path.dirname(__file__) + '/PolynomialFeature_model')
            self.Regression_model = joblib.load(os.path.dirname(__file__) + '/Regression_model')

        def correction_factor(Uh0, sr, Nturb):
            a_input = [[Uh0, sr, np.sqrt(Nturb)]]
            xdata = self.PolynomialFeature_model.fit_transform(a_input)
            return self.Regression_model.predict(xdata)[0]
        Sorensen_Larsen_2021.__init__(self, site, windTurbines, max_cp,
                                      ws_cutin, ws_cutout, correction_factor, rho=rho)


def main():
    if __name__ == '__main__':
        from py_wake.deficit_models.noj import NOJ, NOJLocal
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.wind_turbines.generic_wind_turbines import SimpleGenericWindTurbine

        wt = SimpleGenericWindTurbine(name='Simple', diameter=80, hub_height=70, power_norm=2000)

        ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]

        site = Hornsrev1Site(ti=ti)
        x, y = site.initial_position.T
        for wfm, eff, ref in [(Sorensen_Larsen_2021(site, wt), 1, 553.948821),
                              (Sorensen_etal_2024(site, wt), .91, 592.386668),
                              (Larsen_etal2026(site, wt, 55), .91, 617.022948),
                              (NOJ(site, wt, k=0.032), .91, None),
                              (NOJLocal(site, wt), .91, None),
                              ]:
            res = wfm.aep(x, y) * eff
            print(res)
            if ref:
                npt.assert_allclose(res, ref, rtol=0.001)

        print(.160 * .412 * 24 * 365)  # reference from https://energynumbers.info/capacity-factors-at-danish-offshore-wind-farms


main()
