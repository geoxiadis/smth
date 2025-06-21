import inspect
import os
import pkgutil

import pandas as pd
import xarray as xr

from py_wake import np
from numpy import newaxis as na
from py_wake.site._site import Site
import warnings

from py_wake.utils.fuga_utils import interp_lut_coordinate
from py_wake.utils.grid_interpolator import GridInterpolator
import py_wake
try:
    from xarray.core.merge import merge_attrs
except ModuleNotFoundError:  # pragma: no cover
    from xarray.structure.merge import merge_attrs  # Moved to xarray.structure in v2025.03.0


class Model():
    @property
    def args4model(self):
        return method_args(self.__call__)


class XRLUTModel(Model):
    """Model based on xarray.dataarray look-up table with linear interpolation"""

    def __init__(self, da_lst, get_input=None, get_output=None, method='linear', bounds='limit'):
        """
        Parameters
        ----------
        da_lst : xarray.dataarray
            dataarray containing lookup table.
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
        method : {'linear' or 'nearest} or [{'linear' or 'nearest}, ...]
            interpolation method
        bounds : {'limit', 'check', 'ignore'}
            how to handle out-of-bounds coordinate interpolation, see GridInterpolator
        """
        self.da = da_lst
        self._args4model = getattr(self, '_args4model', set())
        if get_input:
            self.get_input = get_input
        else:
            self._args4model |= set(self.da.dims)
        if get_output:
            self.get_output = get_output
        self._args4model |= (set(inspect.getfullargspec(self.get_input).args) |
                             set(inspect.getfullargspec(self.get_output).args)) - {'self', 'output_ijlk'}

        self.interp = MultiLUTInterpolator(da_lst, method)

    @property
    def args4model(self):
        args4model = Model.args4model.fget(self)  # @UndefinedVariable
        return args4model | set(self._args4model)

    def get_input(self, **kwargs):
        """Default get_input function. This function makes a list of interpolation coordinates based on the input
        dimensions of the dataarray, which must have names that matches the names of the default PyWake
        keyword arguments, e.g. dw_ijlk, WS_ilk, D_src_il, etc, or user-specified custom inputs"""
        return [np.expand_dims(kwargs[k], [i for i, d in enumerate('ijlk') if d not in k.split('_')[-1]])
                for k in self.da.dims]

    def get_output(self, output_ijlk, **kwargs):
        """Default get_output function.
        This function just returns the interpolated values"""
        return output_ijlk

    def __call__(self, **kwargs):
        input_ijlk = self.get_input(**kwargs)
        IJLK = tuple(np.max([inp.shape for inp in input_ijlk], 0))
        output_ijlk = self.interp(np.array([np.broadcast_to(inp, IJLK).flatten()
                                            for inp in input_ijlk]).T).reshape(IJLK)
        return self.get_output(output_ijlk, **kwargs)


class DeprecatedModel():
    def __init__(self, new_model):
        warnings.warn(
            f"""The {self.__class__.__name__} model is not representative of the setup used in the literature. For this, use {new_model} instead""",
            stacklevel=2)


class ModelMethodWrapper():
    def wrap(self, f, wrapper_name='__call__'):
        wrapper = getattr(self, wrapper_name)

        def w(*args, **kwargs):
            return wrapper(f, *args, **kwargs)
        w.superpositionModel = getattr(f, '__self__', f).superpositionModel
        return w


class RotorAvgAndGroundModelContainer():
    def __init__(self, superpositionModel, groundModel, rotorAvgModel):
        self.superpositionModel = superpositionModel
        self.groundModel = groundModel
        self.rotorAvgModel = rotorAvgModel

    @property
    def args4model(self):
        args4model = set()
        if self.groundModel:
            args4model |= self.groundModel.args4model
        if self.rotorAvgModel:
            args4model |= self.rotorAvgModel.args4model
        return args4model

    @property
    def windFarmModel(self):
        return self._windFarmModel

    @windFarmModel.setter
    def windFarmModel(self, wfm):
        self._windFarmModel = wfm
        if self.groundModel:
            self.groundModel.windFarmModel = wfm
        if self.rotorAvgModel:
            self.rotorAvgModel.windFarmModel = wfm

    def wrap(self, f, wrapper_name='__call__'):
        if self.rotorAvgModel:
            f = self.rotorAvgModel.wrap(f, wrapper_name)
        if self.groundModel:
            f = self.groundModel.wrap(f, wrapper_name)
        return f


class MultiLUTInterpolator:
    methods_map: pd.DataFrame
    interpolator_map: pd.DataFrame
    extents: list

    def __init__(self, luts: list[xr.DataArray], methods: list[str]):

        self.set_methods_map(luts[0].dims, methods)

        luts = self.squeeze_luts(luts)
        nearest_dims = [rule["dim"] for _, rule in self.methods_map.iterrows() if rule["method"] == "nearest"]
        lut_map = pd.DataFrame(
            index=range(len(luts)),
            columns=nearest_dims,
        )
        for dimension in nearest_dims:
            lut_map[dimension] = [float(lut[dimension].values) for lut in luts]

        # collapse
        if len(nearest_dims) > 0:
            nearest_inputs_mapping = lut_map.groupby(nearest_dims).apply(lambda df: list(df.index))
        else:
            nearest_inputs_mapping = [[0]]
        self.interpolator_map = pd.DataFrame(
            index=range(len(nearest_inputs_mapping)),
            columns=nearest_dims + ["interpolator"],
        )

        continuous_methods = [rule["method"] for _, rule in self.methods_map.iterrows() if rule["method"] != "nearest"]
        self.extents = list()
        new_index = 0
        for indexes in nearest_inputs_mapping:
            if len(nearest_dims) > 0:
                self.interpolator_map.loc[new_index, nearest_dims] = lut_map[nearest_dims].iloc[indexes[0]].values
            if len(indexes) == 1:
                lut = luts[indexes[0]].squeeze()
            else:
                lut = self.merge_luts([luts[index] for index in indexes])
            self.interpolator_map.loc[new_index, "interpolator"] = GridInterpolator(
                [lut[dim].values for dim in lut.dims],
                lut.values,
                method=continuous_methods,
                bounds="limit"
            )
            self.extents.append(
                dict(
                    x_min=lut.x.values.min(),
                    x_max=lut.x.values.max(),
                    y_min=lut.y.values.min(),
                    y_max=lut.y.values.max(),
                )
            )
            new_index += 1

    def set_methods_map(self, dimensions: tuple, methods: list[str]):
        self.methods_map = pd.DataFrame(
            index=range(len(dimensions)),
            data=dict(
                dim=dimensions,
                method=methods,
            ),
            columns=["dim", "method"],
        )
        self.methods_map.index.rename("input_index")

    def squeeze_luts(self, luts: list[xr.DataArray]) -> list[xr.DataArray]:
        """gets rid of the dimensions with 1 coordinate"""
        squeeze_indexes, squeeze_dims = self.find_squeezable_dims(luts)
        self.methods_map.drop(index=squeeze_indexes, inplace=True)
        self.methods_map.reset_index(drop=True, inplace=True)
        luts = [lut.squeeze(squeeze_dims) for lut in luts]
        return luts

    @staticmethod
    def find_squeezable_dims(luts: list[xr.DataArray]) -> (list[int], list[str]):
        squeeze_indexes = []
        squeeze_dims = []
        for i, dimension in enumerate(luts[0].dims):
            if dimension in ["x", "y", "z"]:
                continue
            values = np.array([lut[dimension].values for lut in luts])
            if np.unique(values).size == 1:
                squeeze_dims.append(dimension)
                squeeze_indexes.append(i)
        return squeeze_indexes, squeeze_dims

    @staticmethod
    def merge_luts(luts: list[xr.DataArray]) -> xr.DataArray:
        merge = False
        for dim in luts[0].dims:
            coords = [lut[dim] for lut in luts]
            if not all(index.equals(coords[0]) for index in coords[1:]):
                merge = True
                break
        if merge:
            z_lst = np.sort(np.unique([z for da in luts for z in da.z.values]))

            x_lst = np.sort(np.unique([da.x for da in luts]))
            x_lst = x_lst[(x_lst >= np.max([da.x[0] for da in luts])) & (x_lst <= np.min([da.x[-1] for da in luts]))]

            y_lst = np.sort(np.unique([da.y for da in luts]))
            y_lst = y_lst[(y_lst >= np.max([da.y[0] for da in luts])) & (y_lst <= np.min([da.y[-1] for da in luts]))]

            luts = [interp_lut_coordinate(da, x=x_lst, y=y_lst, z=z_lst) for da in luts]

            # combine_by_coords does not always merge attributes correctly
            attrs = xr.merge_attrs([da.attrs for da in luts], combine_attrs="drop_conflicts")
            merged_lut = xr.combine_by_coords(luts, combine_attrs="drop").squeeze()
            merged_lut.attrs = attrs
        else:
            merged_lut = luts[0].squeeze()
        return merged_lut

    def __call__(self, xp: np.ndarray) -> np.ndarray:
        result = np.zeros(xp.shape[0])

        nearest_dim_indexes = [index for index, rule in self.methods_map.iterrows() if rule["method"] == "nearest"]
        continuous_dim_indexes = [index for index, rule in self.methods_map.iterrows() if rule["method"] != "nearest"]
        x_index = self.methods_map[self.methods_map["dim"] == "x"].index
        y_index = self.methods_map[self.methods_map["dim"] == "y"].index

        nearest_dims = [rule["dim"] for _, rule in self.methods_map.iterrows() if rule["method"] == "nearest"]
        nearest_dim_data = xp[:, nearest_dim_indexes]

        diff = nearest_dim_data[:, np.newaxis, :] - self.interpolator_map[nearest_dims].values[np.newaxis, :, :]
        dist_squared = np.sum(diff ** 2, axis=2)
        xp_lut_indexes = np.argmin(dist_squared, axis=1)

        for lut_index, interpolator_data in self.interpolator_map.iterrows():
            lut_filter = xp_lut_indexes == lut_index
            if np.all(~lut_filter):
                continue

            xp_y = xp[:, y_index]
            xp_x = xp[:, x_index]
            extent = self.extents[lut_index]

            in_domain = (
                (xp_y <= extent["y_max"])
                & (extent["y_min"] <= xp_y)
                & (xp_x <= extent["x_max"])
                & (extent["x_min"] <= xp_x)
            ).squeeze()
            xp_in_domain = xp[lut_filter & in_domain, :][:, continuous_dim_indexes]
            result[lut_filter & in_domain] = interpolator_data["interpolator"](xp_in_domain)
        return result


def get_exclude_dict():
    from py_wake.deficit_models.deficit_model import ConvectionDeficitModel, WakeDeficitModel, \
        BlockageDeficitModel
    from py_wake.deficit_models.deficit_model import XRLUTDeficitModel
    from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel, NodeRotorAvgModel
    from py_wake.wind_farm_models.engineering_models import EngineeringWindFarmModel, PropagateDownwind
    from py_wake.deflection_models.deflection_model import DeflectionIntegrator

    from py_wake.superposition_models import LinearSum
    from py_wake.deficit_models.noj import NOJDeficit
    from py_wake.turbulence_models.turbulence_model import XRLUTTurbulenceModel
    from py_wake.ground_models.ground_models import NoGround
    from py_wake.site.jit_streamline_distance import JITStreamlineDistance
    return {
        "WindFarmModel": ([EngineeringWindFarmModel], [], PropagateDownwind),
        "EngineeringWindFarmModel": ([], [], PropagateDownwind),
        "DeficitModel": ([ConvectionDeficitModel, BlockageDeficitModel, WakeDeficitModel, XRLUTDeficitModel],
                         [RotorAvgModel], NOJDeficit),
        "WakeDeficitModel": ([ConvectionDeficitModel, XRLUTDeficitModel], [RotorAvgModel], NOJDeficit),
        "RotorAvgModel": ([NodeRotorAvgModel], [], None),
        "SuperpositionModel": ([], [], LinearSum),
        "BlockageDeficitModel": ([XRLUTDeficitModel], [], None),
        "DeflectionModel": ([DeflectionIntegrator], [], None),
        "TurbulenceModel": ([XRLUTTurbulenceModel], [], None),
        "AddedTurbulenceSuperpositionModel": ([], [], None),
        "GroundModel": ([], [], NoGround),
        "Shear": ([], [], None),
        "StraightDistance": ([], [JITStreamlineDistance], None),

    }


def cls_in(A, cls_lst):
    return str(A) in map(str, cls_lst)


def get_models(base_class, exclude_None=False, include_dirs=[]):
    if base_class is Site:
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
        return [IEA37Site, Hornsrev1Site, ParqueFicticioSite]
    exclude_cls_lst, exclude_subcls_lst, default = get_exclude_dict()[base_class.__name__]

    model_lst = []
    base_class_module = inspect.getmodule(base_class)
    for loader, module_name, is_pkg in pkgutil.walk_packages(
            [os.path.dirname(base_class_module.__file__)] + include_dirs):
        if 'test' in module_name:
            continue
        parent_module = os.path.relpath(loader.path, os.path.dirname(
            py_wake.__file__) + "/../").replace("\\", '.').replace("/", '.')
        if parent_module.startswith("..."):  # pragma: no cover
            continue
        module_name = parent_module + '.' + module_name
        import importlib
        try:
            _module = importlib.import_module(module_name)
            for n in dir(_module):
                v = _module.__dict__[n]
                if inspect.isclass(v):
                    if (cls_in(base_class, v.mro()) and
                        not cls_in(v, exclude_cls_lst + [base_class]) and
                        not any([issubclass(v, cls) for cls in exclude_subcls_lst]) and
                            not cls_in(v, model_lst)):
                        model_lst.append(v)
        except ModuleNotFoundError:  # pragma: no cover
            pass

    if default is not None:
        model_lst.remove(model_lst[[m.__name__ for m in model_lst].index(default.__name__)])
    model_lst.insert(0, default)
    if exclude_None and None in model_lst:
        model_lst.remove(None)
    return model_lst


# def list_models():
#     for model_type in list(get_exclude_dict().keys()):
#         print("%s (from %s import *)" % (model_type.__name__, ".".join(model_type.__module__.split(".")[:2])))
#         for model in get_models(model_type):
#             if model is not None:
#                 print("\t%s%s" % (model.__name__, str(inspect.signature(model.__init__)).replace('self, ', '')))


def get_signature(cls, kwargs={}, indent_level=0):
    sig = inspect.signature(cls.__init__)

    def get_arg(n, arg_value):
        if arg_value is None:
            arg_value = sig.parameters[n].default
            if 'object at' in str(arg_value):
                arg_value = get_signature(arg_value.__class__, indent_level=(indent_level + 1, 0)[indent_level == 0])
            elif '<function' in str(arg_value) and 'at 0x' in str(arg_value):
                # arg_value = get_signature(arg_value, indent_level=(indent_level + 1, 0)[indent_level == 0])
                arg_value = arg_value.__name__
            elif isinstance(arg_value, str):
                arg_value = "'%s'" % arg_value
        else:
            arg_value = get_signature(arg_value, indent_level=(indent_level + 1, 0)[indent_level == 0])
        if arg_value is inspect._empty:
            return n
        if isinstance(arg_value, np.ndarray):
            arg_value = arg_value.tolist()
        return "%s=%s" % (n, arg_value)
    if indent_level:
        join_str = ",\n%s" % (" " * 4 * indent_level)
    else:
        join_str = ", "
    arg_str = join_str.join([get_arg(n, kwargs.get(n, None))
                             for n in sig.parameters if n not in {'self', 'args', 'kwargs'}])
    if indent_level and arg_str:
        return "%s(%s%s)" % (cls.__name__, join_str[1:], arg_str)
    else:
        return "%s(%s)" % (cls.__name__, arg_str)


def get_model_input(wfm, x, y, ws=10, wd=270, **kwargs):
    ws, wd = [np.atleast_1d(v) for v in [ws, wd]]
    x, y = map(np.asarray, [x, y])
    wfm.site.distance.setup(src_x_ilk=[[[0]]], src_y_ilk=[[[0]]], src_h_ilk=[[[0]]], src_z_ilk=[[[0]]],
                            dst_xyhz_j=(x, y, x * 0, x * 0))
    dw_ijlk, hcw_ijlk, dh_ijlk = wfm.site.distance(wd_l=wd)
    sim_res = wfm([0], [0], ws=ws, wd=wd, **kwargs)

    args = {'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk,
            'D_src_il': np.atleast_1d(wfm.windTurbines.diameter())[na]}
    args.update({k: sim_res[n].ilk() for k, n in [('yaw_ilk', 'yaw'),
                                                  ('tilt_ilk', 'tilt'),
                                                  ('WS_ilk', 'WS'),
                                                  ('WS_eff_ilk', 'WS_eff'),
                                                  ('ct_ilk', 'CT'),
                                                  ('TI_ilk', 'TI'),
                                                  ('TI_eff_ilk', 'TI_eff'),
                                                  ('h_ilk', 'h')]
                 if n in sim_res})
    args['IJLK'] = (1, len(x), len(wd), len(ws))
    return args


def check_model(model, cls, arg_name=None, accept_None=True):
    if not isinstance(model, cls):
        if model is None and accept_None:
            return

        if arg_name is not None:
            s = f'Argument, {arg_name}, '
        else:
            s = f'{model} '
        s += f'must be a {cls.__name__} instance'
        if inspect.isclass(model) and issubclass(model, cls):
            raise ValueError(s + f'. Did you forget the brackets: {model.__name__}()')

        raise ValueError(s + f', but is a {model.__class__.__name__} instance')


def fix_shape(arr, shape_or_arr_to_match, allow_number=False):

    if allow_number and isinstance(arr, (int, float)):
        return arr

    arr = np.asarray(arr)
    if isinstance(shape_or_arr_to_match, tuple):
        shape = shape_or_arr_to_match
    else:
        shape = np.asarray(shape_or_arr_to_match).shape
    return np.broadcast_to(np.expand_dims(arr, tuple(range(len(arr.shape), len(shape)))), shape)


def method_args(method):
    return set(inspect.getfullargspec(method).args) - {'self', 'func'}


def main():
    if __name__ == '__main__':
        from py_wake.superposition_models import SuperpositionModel
        print(get_models(SuperpositionModel))
        for c in get_models(SuperpositionModel):
            print(isinstance(c(), SuperpositionModel), c)


main()
