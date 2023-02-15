from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.site._site import UniformSite, UniformWeibullSite
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctions
from py_wake.site.shear import LogShear
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake import np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from numpy import newaxis as na
from py_wake.deficit_models.noj import NOJ

"""
Function to calculate the total AEP of a wind farm
Created on 12/02/2023

@author: David Fournely & Ariadna Garcia Montes

Description: Calculation of the annual energy production of the model developped
    by Jens N. Sørensen

Inputs:
    Pg    - [W] Nameplate capacity (generator power)
    D     - [m] Rotor diameter
    H     - [m] Tower height
    Aw    - [m/s] Weibull scale parameter
    kw    - [-] Weibull shape parameter
    Nturb - [-] Number of turbines
    Area  - [m2] Area of wind farm

Outputs:
    Production - [Wh/yr] Area of the wind farm
"""

# Import packages
import numpy as np
from scipy.special import gamma, gammainc


class SimpleWindTurbine(WindTurbine):
    def __init__(self, name, diameter, hub_height, Pg, CP=.48, CT=.75, rho=1.225, Uin=4, Uout=25):
        D = diameter
        self.Ur = (8 * Pg / (np.pi * rho * CP * D**2))**(1 / 3)  # [m/s] Rated wind speed

        def power(ws):
            return np.minimum(1 / 8 * rho * diameter**2 * np.pi * CP * np.asarray(ws)**3, Pg)

        def ct(ws):
            return np.full_like(ws, CT)
        WindTurbine.__init__(self, name, diameter, hub_height,
                             powerCtFunction=PowerCtFunctions(power_function=power, power_unit='w', ct_function=ct))


class SimpleSite(UniformWeibullSite):
    def __init__(self, z0, H, Aw, kw):
        kappa = 0.4  # [-] Von Karman constant
        f = 1.2 * 10**(-4) * np.exp(4)  # [Hz] Coriolis parameter at latitude 55°N

        Gx = gamma(1 + 1 / kw)  # [-] Gx = Um/Aw
        Uh0 = Gx * Aw  # [m/s] Mean velocity at hub height

        # Geostrophic wind speed
        nmax = 10
        eps = 10**(-5)
        n = 0
        G1 = Uh0 * (1 + np.log(Uh0 / (f * H)) / np.log(H / z0))
        dG = abs(Uh0 - G1)
        while (n < nmax) and (dG > eps):
            n = n + 1
            G2 = Uh0 * (1 + np.log(G1 / (f * H)) / np.log(H / z0))
            dG = abs(G2 - G1)
            G1 = G2
        G = G1

        # Mean velocity at hub height without wake effects
        Uh0 = G / (1 + np.log(G / (f * H)) / kappa * np.sqrt((kappa / np.log(H / z0))**2))
        UniformWeibullSite.__init__(self, p_wd=[1], a=[Aw], k=[kw], shear=LogShear(h_ref=H, z0=z0))


def power_production(Pg, CT, D, H, z0, Aw, kw, Nturb, Area):

    # Secondary fixed input data - fixed in the model
    CP = 0.48  # [-] Power coefficient
    rho = 1.225  # [kg/m3] Density of air
    Uin = 4  # [m/s] Cut-in wind speed
    Uout = 25  # [m/s] Cut-out wind speed
    kappa = 0.4  # [-] Von Karman constant
    f = 1.2 * 10**(-4) * np.exp(4)  # [Hz] Coriolis parameter at latitude 55°N

    # Derived input data
    Ur = (8 * Pg / (np.pi * rho * CP * D**2))**(1 / 3)  # [m/s] Rated wind speed
    Gx = gamma(1 + 1 / kw)  # [-] Gx = Um/Aw
    Uh0 = Gx * Aw  # [m/s] Mean velocity at hub height
    Nrow = 3 * np.sqrt(Nturb)  # [-] Number of turbines affected by the free wind
    sr = np.sqrt(Area) / (D * (np.sqrt(Nturb) - 1))  # [-] Mean spacing in diameters
    Ctau = np.pi * CT / (8 * sr * sr)  # [-] Wake parameter
    alpha = 1 / (Ur**3 - Uin**3)  # [W.m-3.s3]
    beta = -Uin**3 / (Ur**3 - Uin**3)  # [W]

    # Geostrophic wind speed
    nmax = 10
    eps = 10**(-5)
    n = 0
    G1 = Uh0 * (1 + np.log(Uh0 / (f * H)) / np.log(H / z0))
    dG = abs(Uh0 - G1)
    while (n < nmax) and (dG > eps):
        n = n + 1
        G2 = Uh0 * (1 + np.log(G1 / (f * H)) / np.log(H / z0))
        dG = abs(G2 - G1)
        G1 = G2
    G = G1

    # Mean velocity at hub height without wake effects
    Uh0 = G / (1 + np.log(G / (f * H)) / kappa * np.sqrt((kappa / np.log(H / z0))**2))

    # Power without wake effects
    eta0 = alpha * Aw**3 * gamma(1 + 3 / kw) * (gammainc(1 + 3 / kw, (Ur / Aw)**kw) - gammainc(1 + 3 / kw, (Uin / Aw)**kw)) + beta * (
        np.exp(-(Uin / Aw)**kw) - np.exp(-(Ur / Aw)**kw)) + np.exp(-(Ur / Aw)**kw) - np.exp(-(Uout / Aw)**kw)  # [-] Without wake effects

    # Mean velocity at hub height with wake effects
    Uh = G / (1 + np.log(G / (f * H)) / kappa * np.sqrt(Ctau + (kappa / np.log(H / z0))**2))

    # Auxiliary variables
    gam = np.log(G / (f * H))
    delta = np.log(H / z0)
    eps1 = (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau + (kappa / delta)**2))
    eps2 = (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau * (Ur / Uout)**3.2 + (kappa / delta)**2))

    # Power production with wake effects
    eta = alpha * (eps1 * Aw)**3 * gamma(1 + 3 / kw) * (gammainc(1 + 3 / kw, (Ur / (eps1 * Aw))**kw) - gammainc(1 + 3 / kw, (Uin / (eps1 * Aw))**kw)) + \
        beta * (np.exp(-(Uin / (eps1 * Aw))**kw) - np.exp(-(Ur / (eps1 * Aw))**kw)) + \
        np.exp(-(Ur / (eps1 * Aw))**kw) - np.exp(-(Uout / (eps2 * Aw))**kw)
    power = Pg * ((Nturb - Nrow) * eta + Nrow * eta0)
    ws_eff = ((Nturb - Nrow) * Uh + Nrow * Uh0) / Nturb
    return power, ws_eff


def WTs_area_v2(wt_x, wt_y):
    """Function to calculate the total area covered by all the WTs of a wind farm
    Created on 13/02/2023

    @author: David Fournely & Ariadna Garcia Montes

    Description: The function find the convex hull of the cloud of points of
        WT position. It therefore calculates the area of the wind farm considering
        a buffer of 1D outside the convex hull

    Inputs:
        wt_x - [m] List of the x-axis coordinates of the WTs
        wt_y - [m] List of the y-axis coordinates of the WTs

    Outputs:
        area - [m2] Area of the wind farm
    """

    # Creation of an array with the x- and y-coordinates of the WTs
    coordinates = np.zeros((len(wt_x), 2))  # Empty array to create a structure
    coordinates[:, 0] = wt_x                # x-coordinates in the first column
    coordinates[:, 1] = wt_y                # y-coordinates in the second column

    # Coordinates of the convex hull of the WT positions
    convex_hull_vertices = coordinates[ConvexHull(coordinates).vertices]
    A1 = ConvexHull(coordinates).volume

    # Coordinates and number of WTs on the border of the WF
    corners_hull = ConvexHull(points=coordinates).vertices
    poly = Polygon([coordinates[i] for i in corners_hull])
    dist = [Point(coordinates[i]).distance(poly.boundary) for i in range(len(coordinates))]
    points_on_border = []
    eps = 10
    for i in range(len(coordinates)):
        if dist[i] < eps:
            points_on_border.append(coordinates[i])
    points_on_border = np.array(points_on_border)
    m = len(points_on_border)
    Nturb = len(coordinates)
    LT = np.sqrt(A1 / (Nturb - 1 / 2 * m - 1))

    # Applying a buffer of 1D around that convex hull
    r = 1 / 2 * LT  # [m] Size of the buffer
    n = 10     # Number of additional points found arooundeach corner of the convex hull
    newPoints = []  # List of the additional points created around corners
    for eachPoint in convex_hull_vertices:
        newPoints += [(eachPoint[0] + np.cos(2 * np.pi / n * x) * r, eachPoint[1] +
                       np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]
    newPoints = np.array(newPoints)

    # Calculation of the area of the convex hull crated with the new points
    area = ConvexHull(newPoints).volume
    return area


class MinimalisticWindFarmModel(WindFarmModel):
    def __init__(self, site, windTurbines):
        WindFarmModel.__init__(self, site, windTurbines)

    def calc_wt_interaction(self, x_ilk, y_ilk, h_i=None, type_i=0,
                            wd=None, ws=None, time=False,
                            n_cpu=1, wd_chunks=None, ws_chunks=None, **kwargs):
        rated_power = self.windTurbines.power([8, 12, 16]).max()
        ct_max = self.windTurbines.ct([8, 12, 16]).max()
        localWind = self.site.local_wind(x_ilk, y_ilk, h_i, wd=[0], ws=10)
        area = WTs_area_v2(wt_x=x_ilk.mean((1, 2)), wt_y=y_ilk.mean((1, 2)))
        z0 = self.site.shear.z0

        power, ws_eff = power_production(Pg=rated_power,
                                         CT=ct_max,
                                         D=self.windTurbines.diameter(),
                                         H=self.windTurbines.hub_height(),
                                         z0=z0,
                                         Aw=localWind['Weibull_A_ilk'].mean(),
                                         kw=localWind['Weibull_k_ilk'].mean(),
                                         Nturb=len(x_ilk),
                                         Area=area)
        I, L, K = len(x_ilk), 1, 1
        WS_eff_ilk, power_ilk = np.full((I, L, K), ws_eff), np.full((I, L, K), power / len(x_ilk))
        TI_eff_ilk = localWind['TI_ilk']
        localWind['P_ilk'][:] = 1
        ct_ilk = np.full((I, L, K), ct_max)
        kwargs_ilk = {'type_i': type_i, **kwargs}

        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, kwargs_ilk


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    wt = SimpleWindTurbine(name='Simple', diameter=80, hub_height=70, Pg=2e6)
    # wt.plot_power_ct()
    # plt.show()

    site = Hornsrev1Site(shear=LogShear(h_ref=70, z0=.1))
    x, y = site.initial_position.T
    wfm = MinimalisticWindFarmModel(site, wt)

    sim_res = wfm(x, y)
    print(sim_res)
    print(sim_res.aep().sum())
    print(NOJ(site, wt)(x, y).aep().sum())
