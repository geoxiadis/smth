from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models.fuga import FugaDeficit


class Fuga(PropagateDownwind):
    """
    Implemented according to:
        Ott, S., & Nielsen, M. (2014).
        Developments of the offshore wind turbine wake model Fuga.
        DTU Wind Energy. DTU Wind Energy E No. 0046
        
    Description:
        - Fuga is a linearized CFD model that can predict wake effects for offshore wind farms.
        - It has the ability to work with different types of turbines for the same project, which makes it useful for inter farm interactions
    """
    
    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=None, deflectionModel=None, turbulenceModel=None, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=FugaDeficit(LUT_path, remove_wriggles=remove_wriggles),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)
