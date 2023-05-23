from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models.fuga import FugaDeficit

class FugaBlockage(All2AllIterative):
    def __init__(self, LUT_path, site, windTurbines, rotorAvgModel=None,
                 deflectionModel=None, turbulenceModel=None, convergence_tolerance=1e-6, remove_wriggles=False):
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
        fuga_deficit = FugaDeficit(LUT_path, remove_wriggles=remove_wriggles)
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel=fuga_deficit,
                                  rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                  deflectionModel=deflectionModel, blockage_deficitModel=fuga_deficit,
                                  turbulenceModel=turbulenceModel, convergence_tolerance=convergence_tolerance)
