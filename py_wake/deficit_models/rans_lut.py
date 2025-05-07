import xarray as xr
import sys
from py_wake.deficit_models.deficit_model import XRLUTDeficitModel
from numpy import newaxis as na
from py_wake import np
from py_wake.superposition_models import LinearSum
from py_wake.utils.rans_lut_utils import fugalut2ranslut


def rescale_lut_coords(lut, zRef, Dref):
    if Dref is not None and zRef is not None:
        # Rescale non-dimensional coord to [m]
        lut['x'] = lut['x'] * Dref
        lut['y'] = lut['y'] * Dref
        lut['z'] = lut['z'] * Dref + zRef
    else:
        if Dref is None:
            print('ERROR: lut coords cannot be rescaled because Dref is not specified')
        elif zRef is None:
            print('ERROR: lut coords cannot be rescaled because zRef is not specified')
        sys.exit()
    return lut


def rename_lut(lut):
    lutvars = ['x', 'y', 'z', 'ct', 'ti', 'yaw', 'type_i']
    renamevars = ['dw_ijlk', 'hcw_ijlk', 'z_ijlk', 'ct_ilk', 'TI_eff_ilk', 'yaw_ilk', 'type_i_ilk']
    d = {}
    for i in range(len(lutvars)):
        if lutvars[i] in lut.coords:
            d[lutvars[i]] = renamevars[i]
    return lut.rename(d)


# The rans_lut from mmpe does not lead to satisfying results when using the shear correction for a row of wts.
# The shear correction does work well for the power of a single wt.
# This needs to be future investigated
# lut files with multple turbines can be provided using type_i dimension. In this case, the coords need to be in [m]
class RANSLUTDeficit(XRLUTDeficitModel):
    def __init__(self, lut=None, zRef=None, Dref=None, rotorAvgModel=None, groundModel=None, use_effective_ws=True, use_effective_ti=False, shearcor=False):
        if lut is None:
            # Load default RANS LUT file based on Fuga LUT
            zRef = 70.0
            Dref = 80.0
            lut, da_ti = fugalut2ranslut(Dref=Dref, zRef=zRef)
            lut = lut.sel(ti=0.075)
        if isinstance(lut, str):
            lut = xr.load_dataset(lut).deficits
        if 'type_i' not in lut.coords:
            lut = rescale_lut_coords(lut, zRef, Dref)
        lut = rename_lut(lut)
        XRLUTDeficitModel.__init__(self, lut, bounds='limit', rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                   use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)

    def wake_radius(self, dw_ijlk, **kwargs):
        # Required for PyWake but not needed for RANS LUT model
        wake_radius_ijlk = np.ones(dw_ijlk.shape)
        return wake_radius_ijlk


def main():
    if __name__ == '__main__':
        from py_wake.turbulence_models.rans_lut_turb import RANSLUTTurbulence
        from py_wake.wind_farm_models.engineering_models import All2AllIterative
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
        import matplotlib.pyplot as plt

        site = IEA37Site(16)
        wt = GenericWindTurbine(name='15MW', diameter=236, hub_height=150, power_norm=15000, max_cp=0.45, constant_ct=0.8)
        path_LUT = '/groups/PyWakeEllipSys/LUT/EnergyIsland_WTgen15MW.nc'
        lut = xr.load_dataset(path_LUT)
        zRef = 150.0
        Dref = 236.0
        deficit = RANSLUTDeficit(lut.deficits, zRef, Dref)
        turb = RANSLUTTurbulence(lut.added_ti, zRef, Dref)
        wfm = All2AllIterative(site, wt, wake_deficitModel=deficit,
                               blockage_deficitModel=deficit,
                               turbulenceModel=turb)
        x, y = site.initial_position.T
        sim_res = wfm(x, y, wd=270)
        fm = sim_res.flow_map()
        plt.figure()
        fm.plot_wake_map()
        plt.show()


main()
