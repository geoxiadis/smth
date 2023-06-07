from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models.utils import ct2a_madsen
from py_wake.superposition_models import SquaredSum
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit


class Bastankhah_PorteAgel_2014(PropagateDownwind):
    """
    Implemented according to:
        Bastankhah M and Porté-Agel F.
        A new analytical model for wind-turbine wakes.
        J. Renew. Energy. 2014;70:116-23.

    Description:
        - Conservation of mass and momentum is applied with the assumption of a Gaussian shape for the wake profile
        in the calculation of the wake deficit
        - Only one parameter needed to determine the velocity distribution: the wake expansion parameter k
    """

    def __init__(self, site, windTurbines, k=0.0324555, ceps=.2, ct2a=ct2a_madsen, use_effective_ws=False,
                 rotorAvgModel=None, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float
            Wake expansion factor
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=BastankhahGaussianDeficit(ct2a=ct2a, k=k, ceps=ceps,
                                                                               use_effective_ws=use_effective_ws,
                                                                               rotorAvgModel=rotorAvgModel,
                                                                               groundModel=groundModel),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
