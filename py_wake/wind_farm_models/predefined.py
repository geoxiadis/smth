from py_wake import np

from py_wake.deficit_models.utils import ct2a_madsen
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.utils.model_utils import DeprecatedModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind


class PredefinedWindFarmModel():
    pass


class NOJ(PropagateDownwind, DeprecatedModel, PredefinedWindFarmModel):
    def __init__(self, site, windTurbines, rotorAvgModel=AreaOverlapAvgModel(),
                 ct2a=ct2a_madsen, k=.1, superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None,
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float, default 0.1
            wake expansion factor
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel, default None
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        from py_wake.deficit_models.deprecated.noj import NOJDeficit
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NOJDeficit(
                                       k=k, ct2a=ct2a, rotorAvgModel=rotorAvgModel, groundModel=groundModel),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
        DeprecatedModel.__init__(self, 'py_wake.literature.noj.Jensen_1983')


class NOJLocal(PropagateDownwind, PredefinedWindFarmModel):
    def __init__(self, site, windTurbines, rotorAvgModel=AreaOverlapAvgModel(),
                 ct2a=ct2a_madsen, a=[0.38, 4e-3], use_effective_ws=True,
                 superpositionModel=LinearSum(),
                 deflectionModel=None,
                 turbulenceModel=STF2017TurbulenceModel(),
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float, default 0.1
            wake expansion factor
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel, default None
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        from py_wake.deficit_models.deprecated.noj import NOJLocalDeficit
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NOJLocalDeficit(ct2a=ct2a, a=a, use_effective_ws=use_effective_ws,
                                                                     rotorAvgModel=rotorAvgModel,
                                                                     groundModel=groundModel),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class BastankhahGaussian(PropagateDownwind, DeprecatedModel, PredefinedWindFarmModel):
    """Predefined wind farm model"""

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
        from py_wake.deficit_models.deprecated.gaussian import BastankhahGaussianDeficit
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=BastankhahGaussianDeficit(ct2a=ct2a, k=k, ceps=ceps,
                                                                               use_effective_ws=use_effective_ws,
                                                                               rotorAvgModel=rotorAvgModel,
                                                                               groundModel=groundModel),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
        DeprecatedModel.__init__(self, 'py_wake.literature.gaussian_models.Bastankhah_PorteAgel_2014')


class NiayifarGaussian(PropagateDownwind, DeprecatedModel, PredefinedWindFarmModel):
    def __init__(self, site, windTurbines, a=[0.38, 4e-3], ceps=.2, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, rotorAvgModel=None, groundModel=None):
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
        from py_wake.deficit_models.deprecated.gaussian import NiayifarGaussianDeficit
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NiayifarGaussianDeficit(a=a, ceps=ceps,
                                                                             rotorAvgModel=rotorAvgModel,
                                                                             groundModel=groundModel),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
        DeprecatedModel.__init__(self, 'py_wake.literature.gaussian_models.Niayifar_PorteAgel_2016')


class IEA37SimpleBastankhahGaussian(PropagateDownwind, DeprecatedModel, PredefinedWindFarmModel):
    """Predefined wind farm model"""

    def __init__(self, site, windTurbines,
                 rotorAvgModel=None, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
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
        from py_wake.deficit_models.deprecated.gaussian import IEA37SimpleBastankhahGaussianDeficit

        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)
        DeprecatedModel.__init__(self, 'py_wake.literature.iea37_case_study1.IEA37CaseStudy1')


class ZongGaussian(PropagateDownwind, DeprecatedModel, PredefinedWindFarmModel):
    def __init__(self, site, windTurbines, a=[0.38, 4e-3], deltawD=1. / np.sqrt(2), lam=7.5, B=3,
                 rotorAvgModel=None,
                 superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None, groundModel=None):
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
        from py_wake.deficit_models.deprecated.gaussian import ZongGaussianDeficit
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=ZongGaussianDeficit(a=a, deltawD=deltawD, lam=lam, B=B,
                                                                         rotorAvgModel=rotorAvgModel,
                                                                         groundModel=groundModel),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
        DeprecatedModel.__init__(self, 'py_wake.literature.gaussian_models.Zong_PorteAgel_2020')
