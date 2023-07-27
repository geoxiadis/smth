from abc import abstractmethod, ABC
from py_wake.superposition_models import AddedTurbulenceSuperpositionModel, LinearSum
from py_wake.utils.model_utils import check_model, method_args, RotorAvgAndGroundModelContainer, XRLUTModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake.ground_models.ground_models import GroundModel
import inspect
import numpy as np


class TurbulenceModel(ABC, RotorAvgAndGroundModelContainer):

    def __init__(self, addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        for model, cls, name in [(addedTurbulenceSuperpositionModel, AddedTurbulenceSuperpositionModel, 'addedTurbulenceSuperpositionModel'),
                                 (rotorAvgModel, RotorAvgModel, 'rotorAvgModel'),
                                 (groundModel, GroundModel, 'groundModel')]:
            check_model(model, cls, name)

        self.addedTurbulenceSuperpositionModel = addedTurbulenceSuperpositionModel
        RotorAvgAndGroundModelContainer.__init__(self, groundModel, rotorAvgModel)

    @property
    def args4model(self):
        args4model = RotorAvgAndGroundModelContainer.args4model.fget(self)  # @UndefinedVariable
        args4model |= method_args(self.calc_added_turbulence)
        return args4model

    def __call__(self, **kwargs):
        f = self.calc_added_turbulence
        if self.rotorAvgModel:
            f = self.rotorAvgModel.wrap(f)
        return f(**kwargs)

    @abstractmethod
    def calc_added_turbulence(self):
        """Calculate added turbulence intensity caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        See class documentation for examples and available arguments

        Returns
        -------
        add_turb_jlk : array_like
        """

    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return self.addedTurbulenceSuperpositionModel.calc_effective_TI(TI_xxx, add_turb_jxxx)


class XRLUTTurbulenceModel(TurbulenceModel, XRLUTModel, ):
    def __init__(self, da, get_input=None, get_output=None, bounds='limit',
                 addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        XRLUTModel.__init__(self, da, get_input=get_input, get_output=get_output, bounds=bounds)
        TurbulenceModel.__init__(self, addedTurbulenceSuperpositionModel, rotorAvgModel=rotorAvgModel,
                                 groundModel=groundModel)

    def calc_added_turbulence(self, **kwargs):
        return XRLUTModel.__call__(self, **kwargs)

    @property
    def args4model(self):
        return TurbulenceModel.args4model.fget(self) | XRLUTModel.args4model.fget(self)  # @UndefinedVariable


class SurrogateTurbulenceModel(TurbulenceModel):
    """Deficit model based on py_wake.utils.tensorflow_surrogate_utils.TensorFlowModel"""

    def __init__(self, surrogateModel, get_input, get_output=None,
                 addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        """
        Parameters
        ----------
        surrogateModel : SurrogateModel
            Surrogate model, e.g. a TensorFlowModel
        get_input : function or None, optional
            if None (default): The get_input method of XRDeficitModel is used. This option requires that the
            names of the input dimensions matches names of the default PyWake keyword arguments, e.g. dw_ijlk, WS_ilk,
            D_src_il, etc, or user-specified custom inputs
            if function: The names of the input for the function should match the names of the default PyWake
            keyword arguments, e.g. dw_ijlk, WS_ilk, D_src_il, etc, or user-specified custom inputs.
            The function should output interpolation coordinates [x_ijlk, y_ijlk, ...], where (x,y,...) match
            the order of the dimensions of the dataarray
        get_output : function or None, optional
            if None (default): The interpolated output is scaled with the local wind speed, WS_ilk,
            or local effective wind speed, WS_eff_ilk, depending on the value of <use_effective_ws>.
            if function: The function should take the argument output_ijlk and an optional set of PyWake inputs. The
            names of the PyWake inputs should match the names of the default PyWake keyword arguments,
            e.g. dw_ijlk, WS_ilk, D_src_il, etc, or user-specified custom inputs.
            The function should return deficit_ijlk
        """
        TurbulenceModel.__init__(self, addedTurbulenceSuperpositionModel, rotorAvgModel=rotorAvgModel,
                                 groundModel=groundModel)

        self.surrogateModel = surrogateModel
        self.get_input = get_input
        if get_output:
            self.get_output = get_output
        self._args4model = (set(inspect.getfullargspec(self.get_input).args) |
                            set(inspect.getfullargspec(self.get_output).args)) - {'self', 'output_ijlk'}

    @property
    def args4deficit(self):
        return TurbulenceModel.args4deficit.fget(self) | set(self._args4model)

    def get_output(self, output_ijlk, **_):
        """Default get_output function.
        This function just returns the surrogate output"""
        return output_ijlk

    def calc_added_turbulence(self, **kwargs):
        inputs = self.get_input(**kwargs)
        max_dim = max([len(v.shape) for v in inputs])
        shape = np.max([(v.shape + (1,) * max_dim)[:max_dim] for v in inputs], 0)
        inputs = np.array([np.broadcast_to(v, shape).ravel() for v in inputs]).T
        return self.get_output(self.surrogateModel.predict_output(inputs).reshape(shape), **kwargs)
