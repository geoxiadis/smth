from py_wake.turbulence_models.turbulence_model import XRLUTTurbulenceModel
from py_wake.superposition_models import LinearSum
from py_wake.utils.rans_lut_utils import RANSLUTModel
from py_wake.tests import ptf


class RANSLUTTurbulence(RANSLUTModel, XRLUTTurbulenceModel):
    """ Expects LUT wake added ti xarray with coords in [m] and PyWake variable names """

    def __init__(self, lut, addedTurbulenceSuperpositionModel=LinearSum(),
                 rotorAvgModel=None, groundModel=None):

        XRLUTTurbulenceModel.__init__(self, self.get_lut(lut, 'added_ti'),
                                      addedTurbulenceSuperpositionModel=addedTurbulenceSuperpositionModel,
                                      rotorAvgModel=rotorAvgModel, groundModel=groundModel)


class RANSLUTDemoTurbulence(RANSLUTTurbulence):
    def __init__(self, zRef=None, Dref=None, addedTurbulenceSuperpositionModel=LinearSum(),
                 rotorAvgModel=None, groundModel=None):
        demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                       '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')
        RANSLUTTurbulence.__init__(self, demo_lut,
                                   addedTurbulenceSuperpositionModel=addedTurbulenceSuperpositionModel,
                                   rotorAvgModel=rotorAvgModel, groundModel=groundModel)
