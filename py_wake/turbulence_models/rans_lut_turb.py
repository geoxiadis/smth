import xarray as xr
import sys
from numpy import newaxis as na
from py_wake import np
from py_wake.turbulence_models.turbulence_model import XRLUTTurbulenceModel
from py_wake.superposition_models import LinearSum
from py_wake.utils.rans_lut_utils import fugalut2ranslut


class RANSLUTTurbulence(XRLUTTurbulenceModel):
    """ Expects LUT wake added ti xarray with coords in [m] and PyWake variable names """
    def __init__(self, lut=None, zRef=None, Dref=None, addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        if lut is None:
            # Load default RANS LUT file based on Fuga LUT
            ds = fugalut2ranslut()
            lut = ds.added_ti
        if isinstance(lut, str):
            lut = xr.load_dataset(lut).ti
        XRLUTTurbulenceModel.__init__(self, lut, addedTurbulenceSuperpositionModel=addedTurbulenceSuperpositionModel,
                                      rotorAvgModel=rotorAvgModel, groundModel=groundModel)
