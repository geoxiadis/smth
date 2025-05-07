import xarray as xr
import sys
from py_wake.deficit_models.deficit_model import XRLUTDeficitModel
from py_wake.deficit_models.rans_lut import rescale_lut_coords, rename_lut
from numpy import newaxis as na
from py_wake import np
from py_wake.turbulence_models.turbulence_model import XRLUTTurbulenceModel
from py_wake.superposition_models import LinearSum


class RANSLUTTurbulence(XRLUTTurbulenceModel):
    def __init__(self, lut=None, zRef=None, Dref=None, addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        if isinstance(lut, str):
            lut = xr.load_dataset(lut).ti
        if 'type_i' not in lut.coords:
            rescale_lut_coords(lut, zRef, Dref)
        lut = rename_lut(lut)
        XRLUTTurbulenceModel.__init__(self, lut, addedTurbulenceSuperpositionModel=addedTurbulenceSuperpositionModel,
                                      rotorAvgModel=rotorAvgModel, groundModel=groundModel)


def main():
    if __name__ == '__main__':
        from py_wake.deficit_models.rans_lut import RANSLUTDeficit
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
        fm.plot_wake_map()
        plt.figure()
        fm.plot_ti_map()
        plt.show()


main()
