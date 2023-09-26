from abc import abstractmethod, ABC

from py_wake.utils.model_utils import method_args
from py_wake import np
from numpy import newaxis as na
from py_wake.deficit_models.deficit_model import WakeDeficitModel
from py_wake.deficit_models.utils import ct2a_madsen


class EngineeringDeficitModel(WakeDeficitModel):
    def __init__(self, ct2a, rotorAvgModel=None, groundModel=None, use_effective_ws=True, use_effective_ti=False):
        WakeDeficitModel.__init__(
            self,
            rotorAvgModel=rotorAvgModel,
            groundModel=groundModel,
            use_effective_ws=use_effective_ws,
            use_effective_ti=use_effective_ti)
        self.ct2a = ct2a

    @property
    def args4model(self):
        args4model = WakeDeficitModel.args4deficit.fget(self)  # @UndefinedVariable
        args4model |= method_args(self.center_wake_factor)
        args4model |= method_args(self.shape_factor)
        return args4model

    @property
    def args4deficit(self):
        return self.args4model

    def calc_deficit(self, **kwargs):
        WS_ref_ilk = kwargs[self.WS_key]
        return WS_ref_ilk[:, na] * self.center_wake_factor(**kwargs) * self.shape_factor(**kwargs)

    @abstractmethod
    def center_wake_factor(self):
        "Normalized deficit along center line (max deficit)"

    @abstractmethod
    def shape_factor(self):
        "Normalized off-center-line deficit"


class LinearExpansionModel(ABC):
    def __init__(self, k):
        self.k = k

    @property
    def args4model(self):
        return method_args(self.expansion_factor) | method_args(self.expansion_offset) | method_args(self.wake_radius)

    def expansion_factor(self, **kwargs):
        if len(np.atleast_1d(self.k)) == 2:
            TI_ref_ilk = kwargs[self.TI_key]
            return self.k[0] * TI_ref_ilk + self.k[1]
        else:
            return np.array([[[self.k]]])

    @abstractmethod
    def expansion_offset(self):
        "wake radius at rotor"

    def wake_radius(self, dw_ijlk, **kwargs):
        """Radius of tophat profile and sigma of gaussian profile"""
        return self.expansion_factor(dw_ijlk=dw_ijlk, **kwargs) * dw_ijlk + self.expansion_offset(**kwargs)


class TopHatWakeDeficit(EngineeringDeficitModel):
    def shape_factor(self, dw_ijlk, cw_ijlk, wake_radius_ijlk, **_):
        return (cw_ijlk < wake_radius_ijlk) * (dw_ijlk > 0)
