import numpy as np
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import WeightedSum
from py_wake.deficit_models import ZongGaussianDeficit
from py_wake.turbulence_models.crespo import CrespoHernandez


class Zong_PorteAgel_2020(PropagateDownwind):
    """
    Implemented according to:
        Haohua Zong and Fernando Porté-Agel
        "A momentum-conserving wake superposition method for wind farm power prediction"
        J. Fluid Mech. (2020), vol. 889, A8; doi:10.1017/jfm.2020.77

    Features:
        - Conservation of mass and momentum and a Gaussian shape for the wake profile
        - Wake expansion function of local turbulence intensity
        - New wake width expression following the approach by Shapiro et al. (2018)

    Description:
        Extension of the Niayifar et al. (2016) implementation with adapted
        Shapiro wake model components, namely a gradual growth of the thrust
        force and an expansion factor not falling below the rotor diameter.
        Shapiro modelled the pressure and thrust force as a combined momentum
        source, that are distributed in the streamwise direction with a Gaussian
        kernel with a certain characteristic length. As a result the induction
        changes following an error function. Zong chose to use a characteristic
        length of D/sqrt(2) and applies it directly to the thrust not the induction
        as Shapiro. This leads to the full thrust being active only 2D downstream of
        the turbine. Zong's wake width expression is inspired by Shapiro's, however
        the start of the linear wake expansion region (far-wake) was related to
        the near-wake length by Vermeulen (1980). The epsilon factor that in the
        original Gaussian model was taken to be a function of CT is now a constant
        as proposed by Bastankhah (2016), as the near-wake length now effectively
        dictates the origin of the far-wake.
    """

    def __init__(self, site, windTurbines, a=[0.38, 4e-3], deltawD=1. / np.sqrt(2), lam=7.5, B=3,
                 rotorAvgModel=None,
                 superpositionModel=WeightedSum(), deflectionModel=None, turbulenceModel=CrespoHernandez(), groundModel=None,
                 use_effective_ws=False, use_effective_ti=True):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=ZongGaussianDeficit(a=a, deltawD=deltawD, lam=lam, B=B,
                                                                         rotorAvgModel=rotorAvgModel,
                                                                         groundModel=groundModel,
                                                                         use_effective_ws=use_effective_ws,
                                                                         use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
