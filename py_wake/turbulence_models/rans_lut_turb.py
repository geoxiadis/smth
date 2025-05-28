import xarray as xr
import sys
from numpy import newaxis as na
from py_wake import np
from py_wake.turbulence_models.turbulence_model import XRLUTTurbulenceModel
from py_wake.superposition_models import LinearSum
from py_wake.utils.rans_lut_utils import convert_ranslut


class RANSLUTTurbulence(XRLUTTurbulenceModel):
    """ Expects LUT wake added ti xarray with coords in [m] and PyWake variable names """
    def __init__(self, lut=None, zRef=None, Dref=None, addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        if lut is None:
            # Load default RANS LUT file based on a demo V80 LUT
            ds = convert_ranslut()
            lut = ds.added_ti
        XRLUTTurbulenceModel.__init__(self, lut, addedTurbulenceSuperpositionModel=addedTurbulenceSuperpositionModel,
                                      rotorAvgModel=rotorAvgModel, groundModel=groundModel)
