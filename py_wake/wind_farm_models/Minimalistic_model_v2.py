"""
Created on 24/03/2023

Description: Implementation of the Minimalistic Prediction Model developped by 
    Jens N. Sørensen and Gunner C. Larsen
    
Version: #2 - correction factor obtained with a polynomial regression

@author: David Fournely and Ariadna Garcia Montes
"""

# Import general packages
from py_wake import np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from scipy.special import gamma, gammainc
import joblib

# Import packages for PyWake 
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctions
from py_wake.site.shear import LogShear


# Import example of two sites
from py_wake.site import UniformWeibullSite

#%% Creating a new class of wind turbine based on simple parameters
class SimpleWindTurbine(WindTurbine):
    def __init__(self, name, diameter, hub_height, Pg, CP=.48, CT=0.75, rho=1.225, Uin=4, Uout=25):
        D = diameter
        self.Ur = (8 * Pg / (np.pi * rho * CP * D**2))**(1 / 3)  # [m/s] Rated wind speed

        def power(ws):
            return np.minimum(Uin<=np.asarray(ws))*(np.asarray(ws)<=Uout)*np.minimum(Pg/(self.Ur**3-Uin**3)*(np.asarray(ws)**3-Uin**3), Pg)

        def ct(ws):
            return np.minimum(CT, CT*(self.Ur/np.asarray(ws))**(3.2))
            #return np.full_like(ws, CT, dtype=np.double)
        WindTurbine.__init__(self, name, diameter, hub_height,
                             powerCtFunction=PowerCtFunctions(power_function=power, power_unit='w', ct_function=ct))

#%% Creating a new class of site based based on Weibull parameters, H and z0 - NOT USED???
class SimpleSite(UniformWeibullSite):
    def __init__(self, z0, H, Aw, kw):
        kappa = 0.4                     # [-] Von Karman constant
        f = 1.2 * 10**(-4) * np.exp(4)  # [Hz] Coriolis parameter at latitude 55°N

        Gx = gamma(1 + 1 / kw)  # [-] Gx = Um/Aw
        Uh0 = Gx * Aw           # [m/s] Mean velocity at hub height

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

#%% Function to calculate the power production
def power_production_a_calibrated(Pg, CT, D, H, z0, Aw, kw, Nturb, Area):
    
    """
    Function to calculate the total AEP of a wind farm
    Created on 12/02/2023

    @author: David Fournely & Ariadna Garcia Montes

    Description: Calculation of the annual energy production of the model developped
        by Jens N. Sørensen

    Inputs:
        Pg    - [W] Nameplate capacity (generator power)
        CT    - [-] Thrust coefficient
        D     - [m] Rotor diameter
        H     - [m] Tower height
        Aw    - [m/s] Weibull scale parameter
        kw    - [-] Weibull shape parameter
        Nturb - [-] Number of turbines
        Area  - [m2] Area of wind farm

    Outputs:
        Production - [Wh] Annual energy production of the wind farm
    """

    # Secondary fixed input data - fixed in the model
    CP = 0.48                      # [-] Power coefficient
    rho = 1.225                     # [kg/m3] Density of air
    Uin = 4                         # [m/s] Cut-in wind speed
    Uout = 25                       # [m/s] Cut-out wind speed
    kappa = 0.4                     # [-] Von Karman constant
    f = 1.2 * 10**(-4) * np.exp(4)  # [Hz] Coriolis parameter at latitude 55°N

    # Derived input data
    Ur = (8 * Pg / (np.pi * rho * CP * D**2))**(1 / 3)  # [m/s] Rated wind speed
    Gx = gamma(1 + 1 / kw)                              # [-] Gx = Um/Aw
    Uh0 = Gx * Aw                                       # [m/s] Mean velocity at hub height
    sr = np.sqrt(Area) / (D * (np.sqrt(Nturb) - 1))     # [-] Mean spacing in diameters
    Ctau = np.pi * CT / (8 * sr * sr)                   # [-] Wake parameter
    alpha = 1 / (Ur**3 - Uin**3)                        # [W.m-3.s3]
    beta = -Uin**3 / (Ur**3 - Uin**3)                   # [W]
    nu = np.sqrt(0.5*Ctau)*D/(kappa**2*H)*np.log(H/z0)  # [-] wake eddy viscosity
    
    # Determination of the correction factor a
    PolynomialFeature_model = joblib.load('PolynomialFeature_model')
    Regression_model = joblib.load('Regression_model')
    a_input = [[Uh0,sr,np.sqrt(Nturb)]]
    xdata = PolynomialFeature_model.fit_transform(a_input)
    a = Regression_model.predict(xdata)[0]
    Nrow = a*np.sqrt(Nturb)                             # [-] Number of turbines affected by the free wind

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
    Uh = G / (1 + np.log(G / (f * H)) / kappa * np.sqrt(Ctau + (kappa / np.log(H / z0*(1-D/(2*H))**(nu/(1+nu))))**2))

    # Auxiliary variables
    gam = np.log(G / (f * H))
    delta = np.log(H / z0)
    eps1 = (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau + (kappa / delta)**2))
    eps2 = (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau * (Ur / Uh)**3.2 + (kappa / delta)**2))

    # Power production with wake effects
    eta = alpha * (eps1 * Aw)**3 * gamma(1 + 3 / kw) * (gammainc(1 + 3 / kw, (Ur / (eps1 * Aw))**kw) - gammainc(1 + 3 / kw, (Uin / (eps1 * Aw))**kw)) + \
        beta * (np.exp(-(Uin / (eps1 * Aw))**kw) - np.exp(-(Ur / (eps1 * Aw))**kw)) + \
        np.exp(-(Ur / (eps1 * Aw))**kw) - np.exp(-(Uout / (eps2 * Aw))**kw)
    power = Pg * ((Nturb - Nrow) * eta + Nrow * eta0)
    ws_eff = ((Nturb - Nrow) * Uh + Nrow * Uh0) / Nturb
    return power, ws_eff

#%% Function to calculate the area of the wind farm knowing the WT positions
def WTs_area(wt_x,wt_y):
    
    """
    Function to calculate the total area covered by all the WTs of a wind farm
    Created on 13/02/2023

    @author: David Fournely & Ariadna Garcia Montes

    Description: The function find the convex hull of the cloud of points of 
        WT position. It therefore calculates the wind turbine interspacing and uses 
        a simple formula to deduce the area depending only of the interspacing

    Inputs:
        wt_x - [m] List of the x-axis coordinates of the WTs
        wt_y - [m] List of the y-axis coordinates of the WTs

    Outputs:
        area - [m2] Area of the wind farm
    """
    
    # Creation of an array with the x- and y-coordinates of the WTs
    coordinates = np.zeros((len(wt_x),2))  # Empty array to create a structure
    coordinates[:,0] = wt_x                # x-coordinates in the first column
    coordinates[:,1] = wt_y                # y-coordinates in the second column
    
    # Calculation of the area of the convex hull
    A1 = ConvexHull(coordinates).volume
    
    # Coordinates and number of WTs on the border of the WF 
    # (are considered on the border all WTs with a distance < eps = 10m to the boudary of the convex hull)
    corners_hull = ConvexHull(points = coordinates).vertices # [-] Coordinates of the corners of the convex hull
    poly = Polygon([coordinates[i] for i in corners_hull]) # [-] Polygon made with the coordinates of the corners of the convex hull
    dist = [Point(coordinates[i]).distance(poly.boundary) for i in range(len(coordinates))] # [-] Distance of all the WTs to the boundaries of the convex hull
    points_on_border = []
    eps = 10
    for i in range(len(coordinates)):
        if dist[i] < eps:
            points_on_border.append(coordinates[i])
    points_on_border = np.array(points_on_border)
    m = len(points_on_border)  #[-] Number of WTs on the border
    
    # Calculation of the wind turbine interspacing knowing the number of WT on the border "m"
    Nturb = len(coordinates) # [-] Total number of WTs
    LT = np.sqrt(A1/(Nturb-1/2*m-1))
    
    # Calculating the area with formula (7) in "A Minimalist Prediction Model to Determine Energy Production and Costs of Offshore Wind Farms"
    area = ((np.sqrt(Nturb)-1)*LT)**2
    return area

#%% Definition of the model as a new class
class MinimalisticWindFarmModel_a_calibrated(WindFarmModel):
    def __init__(self, site, windTurbines):
        WindFarmModel.__init__(self, site, windTurbines)
    
    def calc_wt_interaction(self, x_ilk, y_ilk, h_i=None, type_i=0,
                            wd=None, ws=None, time=False,
                            n_cpu=1, wd_chunks=None, ws_chunks=None, **kwargs):
    
        rated_power = self.windTurbines.power([8, 12, 16]).max()
        ct_max = self.windTurbines.ct([8, 12, 16]).max()
        area = WTs_area(wt_x=x_ilk.mean((1, 2)), wt_y=y_ilk.mean((1, 2)))
        
        z0 = 0.0001        #[m] Fixed roughness length
        n_sector = 360     #[-] Number of sectors to consider 
        
        power = 0          #[Wh] Initioalisation of the production
        ws_eff = 0         #[m/s] Initioalisation of the effective wind speed
        
        # Initialisation of the Weibull parameters list
        A_ws=np.array(n_sector*[0.0])             # [m/s] Weibull scale parameter
        sector_frequency=np.array(n_sector*[0.0]) # [-] Frequency of occurence of each wind sector
        k_ws=np.array(n_sector*[0.0])             # [-] Weibull shape parameter
        
        # Get the local wind from the site definition
        localWind = self.site.local_wind(x_ilk, y_ilk, h_i)
        
        # Calculation of the AEP for each sector
        for i in range(n_sector):
            
            # Get the local wind parameter from the sector
            A_ws[i] = localWind.Weibull_A_ilk[0,i*int(360/n_sector),0]
            k_ws[i] = localWind.Weibull_k_ilk[0,i*int(360/n_sector),0]
            sector_frequency[i] = localWind.Sector_frequency_ilk[0,i*int(360/n_sector),0]*360/n_sector
            
            # Do not calculate if the sector frequency is equal to 0, to make it faster
            if sector_frequency[i] > 0:
                power_sector, ws_eff_sector = power_production_a_calibrated(Pg=rated_power,
                                                  CT=ct_max,
                                                  D=self.windTurbines.diameter(),
                                                  H=self.windTurbines.hub_height(),
                                                  z0=z0,
                                                  Aw=A_ws[i],
                                                  kw=k_ws[i],
                                                  Nturb=len(x_ilk),
                                                  Area=area)
                power = power + sector_frequency[i]*power_sector
                ws_eff = ws_eff + sector_frequency[i]*ws_eff_sector
        
        # Create LocalWind2 to get the correct structure to return, this one will not be used after
        localWind2 = self.site.local_wind(x_ilk, y_ilk, h_i,wd=0,ws=10)
        
        I, L, K = len(x_ilk), 1, 1
        WS_eff_ilk, power_ilk = np.full((I, L, K), ws_eff), np.full((I, L, K), power / len(x_ilk))
        TI_eff_ilk = localWind2['TI_ilk']
        localWind2['P_ilk'][:] = 1
        ct_ilk = np.full((I, L, K), ct_max)
        kwargs_ilk = {'type_i': type_i, **kwargs}
        
        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind2, kwargs_ilk