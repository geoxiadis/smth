from py_wake import np
import os
from numpy import newaxis as na

# PyWake objects
from py_wake.utils.gradients import hypot
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014, Blondel_Cathelain_2020
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.fuga import Ott_Nielsen_2014, Ott_Nielsen_2014_Blockage
from py_wake.deficit_models.fuga import FugaMultiLUTDeficit
from py_wake.wind_farm_models import All2AllIterative
from py_wake.site import UniformSite
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctions
from py_wake.turbulence_models.crespo import CrespoHernandez

# -----------------------------------------------------
# Default values
# -----------------------------------------------------

data_path = os.path.dirname(__file__) + '/data/'  # path to reference data
cLES = 'b'  # line color for LES results
cRANS = 'g'  # line color for RANS
lw = 2.0  # line width for most resutls


# -----------------------------------------------------
# General functions
# -----------------------------------------------------


def integrate_velocity_deficit_arc(wd, U, R, U0):
    # Integrate velocity deficit
    # Input: velocity deficit defined on an arc
    #        as a function of the relative wd
    # Output: Estimate of a projected stream-wise
    #         velocity deficit, normalized by U0*Ly
    wd = wd / 180.0 * np.pi
    dy = R * (np.sin(wd[1:]) - np.sin(wd[:-1]))
    return np.sum((1 - U[:-1] / U0) * dy) / np.sum(dy)


def gauss(mu, sigma, x):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))


def GaussianFilter(y, wd, nwdGA, sigma):
    # Gaussian filter for cyclic data, i.e. as function of wind direction
    # y is a vector and a function of wd
    # It is assumed that the spacing in wd is uniform
    # yGA is the returned filtered y
    # nwdGA is the amount of points that are used for the smoothing
    wdDelta = wd[1] - wd[0]
    # wdGArel is the smoothing operator
    wdGArel = np.arange(-nwdGA, nwdGA + 1) * wdDelta  # seq(- nwdGA * wdDelta, nwdGA * wdDelta, wdDelta)
    ny = len(y)
    nwd = len(wd)
    assert ny == nwd, 'length of wd does not correspond with length of y'
    yGA = np.zeros((1, ny))
    if (nwd * wdDelta == 360):
        # Full wind rose
        i1 = 0
        i2 = nwd
    else:
        # Partial wind rose: cannot GA the first and last wind directions
        i1 = nwdGA
        i2 = nwd - nwdGA

    gauss_arr = gauss(0, sigma, wdGArel)
    for i, wdrelcor_arr in zip(range(i1, i2), (wd[range(i1, i2), na] + wdGArel) % 360):
        yGA[:, i] = np.sum(y[np.searchsorted(wd, wdrelcor_arr)] * gauss_arr)
    yGA[0, :] = yGA[0, :] / gauss_arr.sum()
    return yGA


def sigmaVarDist(x, y, xref, yref):
    dist = hypot(x - xref, y - yref)
    sigma = hypot(0.00035 * dist + 2.1, 2.5)
    return sigma


def sigma_hornsrev(method, wt_x, wt_y, sigma_cnst=5.0):
    # A general standard deviation (std) is used to Gaussian average results of Power
    # for the Horns Rev 1 wind farm.
    # Three methods can be used:
    if method == 'constant':
        # Constant sigma of 5 deg
        sigma = np.zeros((80))
        sigma[:] = sigma_cnst
    elif method == 'Gaumond':
        # Gaumond et el. (2013): DOI: 10.1002/we.1625
        # Fitted a std per row using the power deficit at WT 2 from Fuga.
        sigma = np.zeros((10, 8))
        sigma[:, 0] = 7.4
        sigma[:, 1] = 7.0
        sigma[:, 2] = 6.2
        sigma[:, 3] = 5.8
        sigma[:, 4] = 5.4
        sigma[:, 5] = 5.0
        sigma[:, 6] = 4.5
        sigma[:, 7] = 4.8
        sigma = sigma.flatten()
    elif method == 'vanderLaan':
        # van der Laan et al. (2014): DOI: 10.1002/we.1804
        # Using empirical data to define a std based on the distance between
        # a wt and the location where the reference wind direction was measured (WT G2)
        sigma = np.zeros((80))
        WTref = 7  # WT G2
        for iAD in range(80):
            sigma[iAD] = sigmaVarDist(wt_x[iAD], wt_y[iAD], wt_x[int(WTref - 1)], wt_y[int(WTref - 1)])
    return sigma


def Gau_k_from_Iu(Iu):
    # Calculate Gaussian wake model expansion parameter k from a fit with LES
    # doi:10.1088/1742-6596/625/1/012039
    if Iu < 0.065:
        k = 0.4 * 0.065
    elif 0.065 <= Iu and Iu <= 0.15:
        k = 0.4 * Iu + 0.004
    elif Iu > 0.15:
        k = 0.4 * 0.15
    # Remove rounding errors
    k = int(k * 10 ** 6) / 10 ** 6
    return k


def NOJ_k_from_location(location):
    # Calculate NOJ wake model expansion parameter k based on onshore and offshore location
    if location == 'offshore':
        k = 0.04
    elif location == 'onshore':
        k = 0.1
    else:
        print('Location', location, 'is undefined. Choose onshore of offshore.')
    return k


def get_downstream_label(sDown_D):
    # Convert downstream distance label at 2.5 - > 2p5 or 4.0 -> 4
    s = str(sDown_D)
    return (s.rstrip('0').rstrip('.') if '.' in s else s).replace('.', 'p')


def wffm_kwargs(case, wffm):
    # Call engineering wind farm models used with appropriate kwargs params to instantiate the
    # configuration of the models represented in the literature setup

    # Use default site/windTurbine models unless explicitly specified
    if 'site' not in case:
        case['site'] = UniformSite(p_wd=[1], ti=case['TItot'], ws=case['U0'])

    if 'wt' not in case:
        case['wt'] = WindTurbine(name=case['name'],
                                 diameter=case['D'],
                                 hub_height=case['zH'],
                                 powerCtFunction=PowerCtFunctions(power_function=lambda ws: ws,  # Dummy power
                                                                  # function for wind turbine
                                                                  power_unit='w',
                                                                  ct_function=lambda ct: case['CT']))  # Dummy constant CT for the wind turbine

    wake_params = {'site': case['site'], 'windTurbines': case['wt']}
    # Set model-specific params
    if wffm == Jensen_1983:
        wake_params.update({'k': NOJ_k_from_location(case['location'])})
    elif wffm == Bastankhah_PorteAgel_2014:
        wake_params.update({'k': Gau_k_from_Iu(case['TItot'])})
    elif wffm == Blondel_Cathelain_2020:
        wake_params.update({'turbulenceModel': CrespoHernandez()})
    # special case for F2F where FugaMultiLUTDeficit requires All2AllIterative method
    elif wffm == Ott_Nielsen_2014 or wffm == Ott_Nielsen_2014_Blockage:
        if 'lut_paths_lst' in case:  # f2f parameters only passed if case defines a list of lut paths
            fuga_multi_deficit = FugaMultiLUTDeficit(LUT_path_lst=case['lut_paths_lst'],
                                                     z_lst=case['wt']._hub_heights.tolist(),
                                                     TI_ref_height=case['wt'].hub_height())
            fuga_multilut_wfm = All2AllIterative(wake_deficitModel=fuga_multi_deficit,
                                                 blockage_deficitModel=fuga_multi_deficit,
                                                 **wake_params)
            return fuga_multilut_wfm
        else:
            wake_params.update({'LUT_path': case['fugaPath']})
    return wffm(**wake_params)
