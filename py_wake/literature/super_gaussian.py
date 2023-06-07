from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import SquaredSum
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2023, BlondelSuperGaussianDeficit2020


class Blondel_Cathelain_2020(PropagateDownwind):
    """
    Implemented according to:
        Blondel and Cathelain (2020)
        An alternative form of the super-Gaussian wind turbine wake model
        Wind Energ. Sci., 5, 1225–1236, 2020 https://doi.org/10.5194/wes-5-1225-2020 [1]

    Features:
        - Wake profile transitions from top-hat at near wake to Gaussian at far wake
        - characteristic wake width (sigma) function of turbulence intensity and CT
        - evolution of super gaussian "n" order function of downwind distance and turbulence intensity

    Description:
        Super gaussian wake order "n" is determined with the calibrated parameters: a_f, b_f, c_f; 
        with a_f kept constant at 3.11
        Calibrated parameters taken from Table 2 and 3 in [1]
    """

    def __init__(self, site, windTurbines, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, rotorAvgModel=None, groundModel=None,
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
                                   wake_deficitModel=BlondelSuperGaussianDeficit2020(rotorAvgModel=rotorAvgModel,
                                                                         groundModel=groundModel,
                                                                         use_effective_ws=use_effective_ws,
                                                                         use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class Blondel_2023(PropagateDownwind):
    """Implemented according to:
        Blondel and Cathelain (2020)
        An alternative form of the super-Gaussian wind turbine wake model
        Wind Energ. Sci., 5, 1225–1236, 2020 https://doi.org/10.5194/wes-5-1225-2020
        
        With calibrated parameters taken from Table 1 in:
        Blondel (2023)
        Brief communication: A momentum-conserving superposition method applied to 
        the super-Gaussian wind turbine wake model
        Wind Energ. Sci., 8, 141–147, 2023, 2023 https://doi.org/10.5194/wes-8-141-2023

    Features:
        - Wake profile transitions from top-hat at near wake to Gaussian at far wake
        - Characteristic wake width (sigma) function of turbulence intensity and CT
        - Evolution of super gaussian "n" order function of downwind distance and turbulence intensity

    Description:
        Linear relationship between turbulence intensity and the Gaussian wake width given by
        the calibrated parameters: a_s, b_s, c_s
        Super gaussian wake order "n" is determined with the calibrated parameters: a_f, b_f, c_f; 
    """

    def __init__(self, site, windTurbines, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, rotorAvgModel=None, groundModel=None,
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
                                   wake_deficitModel=BlondelSuperGaussianDeficit2023(rotorAvgModel=rotorAvgModel,
                                                                         groundModel=groundModel,
                                                                         use_effective_ws=use_effective_ws,
                                                                         use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
