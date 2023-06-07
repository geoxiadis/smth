from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models import NiayifarGaussianDeficit
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.superposition_models import LinearSum


class Niayifar_PorteAgel_2016(PropagateDownwind):
    """
    Implemented according to:
        Amin Niayifar and Fernando Porté-Agel
        Analytical Modeling of Wind Farms: A New Approach for Power Prediction
        Energies 2016, 9, 741; doi:10.3390/en9090741 [1]

    Features:
        - Conservation of mass and momentum and a Gaussian shape for the wake profile
        - Wake expansion function of local turbulence intensity

    Description:
        - The expansion rate 'k' varies linearly with local turbluence intensity: k = a1 I + a2.
        The default constants are set according to publications by Porte-Agel's group, which are based
        on LES simulations. Lidar field measurements by Fuertes et al. (2018) indicate that a = [0.35, 0.0] is also a valid selection.
        - Wake superposition is represented by linearly adding the wakes
        - The Crespo Hernandez turbulence model is used to calculate the added streamwise turbulence intensity, Eq 14 in [1]
    """

    def __init__(self, site, windTurbines, a=[0.38, 4e-3], ceps=.2, superpositionModel=LinearSum(),
                 deflectionModel=None, turbulenceModel=CrespoHernandez(), rotorAvgModel=None, groundModel=None,
                 use_effective_ws=False, use_effective_ti=True):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default LinearSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default CrespoHernandez
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NiayifarGaussianDeficit(a=a, ceps=ceps,
                                                                             rotorAvgModel=rotorAvgModel,
                                                                             groundModel=groundModel,
                                                                             use_effective_ws=use_effective_ws,
                                                                             use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
