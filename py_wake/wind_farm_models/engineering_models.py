from py_wake import wind_farm_models
import multiprocessing
import warnings
from abc import abstractmethod

from numpy import newaxis as na
from tqdm import tqdm

from py_wake import np
from py_wake.deficit_models.deficit_model import (
    BlockageDeficitModel,
    ConvectionDeficitModel,
    WakeDeficitModel,
)
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.input_modifier_models.input_modifier_model import InputModifierModel
from py_wake.rotor_avg_models.rotor_avg_model import (
    NodeRotorAvgModel,
    RotorAvgModel,
    WSPowerRotorAvg,
)
from py_wake.superposition_models import (
    CumulativeWakeSum,
    LinearSum,
    SquaredSum,
    SuperpositionModel,
    WeightedSum,
)
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.utils import gradients
from py_wake.utils.gradients import cabs, item_assign
from py_wake.utils.model_utils import check_model, fix_shape
from py_wake.utils.parallelization import get_map_func
from py_wake.wind_farm_models.external_wind_farm_models import ExternalWindFarm
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines._wind_turbines import WindTurbines


class EngineeringWindFarmModel(WindFarmModel):
    """
    Base class for engineering wake models

    General suffixes:

    - i: turbines ordered by id
    - j: downstream points/turbines
    - k: wind speeds
    - l: wind directions

    Arguments available for calc_deficit (specifiy in args4deficit):

    - WS_ilk: Local wind speed without wake effects
    - TI_ilk: Local turbulence intensity without wake effects
    - TI_std_ilk: Standard deviation of local turbulence intensity
    - WS_eff_ilk: Local wind speed with wake effects
    - TI_eff_ilk: Local turbulence intensity with wake effects
    - D_src_il: Diameter of source turbine
    - D_dst_ijl: Diameter of destination turbine
    - dw_ijlk: Downwind distance from turbine i to point/turbine j
    - hcw_ijlk: Horizontal cross wind distance from turbine i to point/turbine j
    - dh_ijl: vertical distance from turbine i to point/turbine j
    - cw_ijlk: Cross wind(horizontal and vertical) distance from turbine i to point/turbine j
    - ct_ilk: Thrust coefficient

    """
    default_grid_resolution = 500

    def __init__(self, site, windTurbines: WindTurbines, wake_deficitModel, superpositionModel, rotorAvgModel=None,
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None, inputModifierModels=[],
                 externalWindFarms=[]):

        WindFarmModel.__init__(self, site, windTurbines)
        if not isinstance(inputModifierModels, (list, tuple)):
            inputModifierModels = [inputModifierModels]
        for model, cls, name in ([(wake_deficitModel, WakeDeficitModel, 'wake_deficitModel'),
                                  (superpositionModel, SuperpositionModel, 'superpositionModel'),
                                  (blockage_deficitModel, BlockageDeficitModel, 'blockage_deficitModel'),
                                  (deflectionModel, DeflectionModel, 'deflectionModel'),
                                  (turbulenceModel, TurbulenceModel, 'turbulenceModel')] +
                                 [(imm, InputModifierModel, 'inputModificierModels') for imm in inputModifierModels] +
                                 [(ewf, ExternalWindFarm, 'externalWindFarms') for ewf in externalWindFarms]):
            check_model(model, cls, name)
            if model is not None:
                setattr(model, 'windFarmModel', self)
            setattr(self, name, model)
        self.inputModifierModels = inputModifierModels
        self.externalWindFarms = externalWindFarms

        if isinstance(superpositionModel, (WeightedSum, CumulativeWakeSum)):
            assert isinstance(wake_deficitModel, ConvectionDeficitModel)
            wake_rotorAvgModel = rotorAvgModel or self.wake_deficitModel.rotorAvgModel
            assert wake_rotorAvgModel is None or isinstance(wake_rotorAvgModel, NodeRotorAvgModel), \
                "WeightedSum and CumulativeWakeSum only work with NodeRotorAvgModel-based rotor average models"
            assert not isinstance(wake_rotorAvgModel, WSPowerRotorAvg), \
                'WeightedSum and CumulativeWakeSum does not work WSPowerRotorAvg'
            assert len(externalWindFarms) == 0, \
                'WeightedSum and CumulativeWakeSum does not work with ExternalWindFarms'
        # TI_eff requires a turbulence model
        assert 'TI_eff_ilk' not in wake_deficitModel.args4deficit or turbulenceModel
        self.wake_deficitModel = wake_deficitModel
        if rotorAvgModel is not None:
            warnings.warn("""The rotorAvgModel-argument of WindFarmModel is ambiguous and therefore deprecated.
            Set the rotorAvgModel of the wake_deficitModel, the blockage_deficitModel and/or turbulenceModel instead.
            Until removed, the rotorAvgModel of WindFarmModel will apply the rotorAvgModel to the wake_deficitModel only
            if a rotorAvgModel has not already been specified for the wake_deficitModel""",
                          DeprecationWarning, stacklevel=2)
            check_model(rotorAvgModel, RotorAvgModel, 'rotorAvgModel')
            self.wake_deficitModel.rotorAvgModel = self.wake_deficitModel.rotorAvgModel or rotorAvgModel

        self.superpositionModel = superpositionModel
        self.wake_deficitModel.superpositionModel = superpositionModel
        self.blockage_deficitModel = blockage_deficitModel
        self.deflectionModel = deflectionModel
        self.turbulenceModel = turbulenceModel

        # wake expansion continuation (wake-width scale factor) see
        self.wec = 1
        # Thomas, J. J. and Ning, A., "A Method for Reducing Multi-Modality in the Wind Farm Layout Optimization Problem,"
        # Journal of Physics: Conference Series, Vol. 1037, The Science of Making
        # Torque from Wind, Milano, Italy, jun 2018, p. 10.
        self.deficit_initalized = False

        self.args4deficit = self.wake_deficitModel.args4deficit
        # self.args4deficit = set(self.args4deficit) | {'yaw_ilk'}
        if self.blockage_deficitModel:
            self.args4deficit = set(self.args4deficit) | set(self.blockage_deficitModel.args4deficit)
            # use linear sum as default blockage superpositionModel
            is_w_or_c_sum = isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum))
            alt_model = [self.superpositionModel, LinearSum()][is_w_or_c_sum]
            self.blockage_deficitModel.superpositionModel = self.blockage_deficitModel.superpositionModel or alt_model

        self.args4all = set(self.args4deficit)
        if self.turbulenceModel:
            self.args4all |= set(self.turbulenceModel.args4model)
        if self.deflectionModel:
            self.args4all |= set(self.deflectionModel.args4deflection)
        for input_modifier in self.inputModifierModels:
            self.args4all |= set(input_modifier.args4model)
        for ext_wf in self.externalWindFarms:
            self.args4all |= set(ext_wf.args4model)

    def __str__(self):
        def name(o):
            return o.__class__.__name__

        models = [self.__class__.__bases__[0].__name__, "%s-wake" % name(self.wake_deficitModel)]
        if self.blockage_deficitModel:
            models.append("%s-blockage" % name(self.blockage_deficitModel))
        models.append("%s-superposition" % (name(self.superpositionModel)))
        if self.deflectionModel:
            models.append("%s-deflection" % name(self.deflectionModel))
        if self.turbulenceModel:
            models.append("%s-turbulence" % name(self.turbulenceModel))
        return "%s(%s)" % (name(self), ", ".join(models))

    def _init_deficit(self, **kwargs):
        """Calculate layout dependent wake (and blockage) deficit terms"""
        self.wake_deficitModel.calc_layout_terms(**kwargs)
        self.wake_deficitModel.deficit_initalized = True
        if self.blockage_deficitModel:
            if self.blockage_deficitModel != self.wake_deficitModel:
                self.blockage_deficitModel.calc_layout_terms(**kwargs)
            self.blockage_deficitModel.deficit_initalized = True

    def _reset_deficit(self):
        self.wake_deficitModel.deficit_initalized = False
        if self.blockage_deficitModel:
            self.blockage_deficitModel.deficit_initalized = False

    def _add_blockage(self, deficit, dw_ijlk, **kwargs):
        # the split line between wake and blockage is set slightly upstream to handle
        # numerical inaccuracy in the trigonometric functions that calculates dw_ijlk
        rotor_pos = -1e-10
        if self.blockage_deficitModel is None:
            deficit *= (dw_ijlk > rotor_pos)
            blockage = None
        elif (self.blockage_deficitModel != self.wake_deficitModel):
            blockage = self.blockage_deficitModel.calc_blockage_deficit(
                dw_ijlk=dw_ijlk, WS_ref_ilk=kwargs[self.blockage_deficitModel.WS_key], **kwargs)
            deficit *= (dw_ijlk > rotor_pos)
        else:
            # Same model for both wake and blockage
            # keep blockage in deficit and set blockage to zero
            blockage = np.zeros_like(deficit)
        return deficit, blockage

    def _calc_deficit(self, dw_ijlk, **kwargs):
        """Calculate wake (and blockage) deficit"""
        deficit = self.wake_deficitModel(dw_ijlk=dw_ijlk, **kwargs)
        deficit, blockage = self._add_blockage(deficit, dw_ijlk, **kwargs)
        return deficit, blockage

    def _calc_deficit_convection(self, dw_ijlk, **kwargs):
        """Calculate wake convection deficit (and blockage)"""
        deficit_centre, uc, sigma_sqr = self.wake_deficitModel.calc_deficit_convection(dw_ijlk=dw_ijlk, **kwargs)
        deficit_centre, blockage = self._add_blockage(deficit_centre, dw_ijlk, **kwargs)
        deficit, blockage = self._calc_deficit(dw_ijlk=dw_ijlk, **kwargs)
        return deficit, deficit_centre, uc, sigma_sqr, blockage

    def _calc_added_turbulence(self, **kwargs):
        """Calculate added turbulence intensity."""
        return self.turbulenceModel.calc_added_turbulence(**kwargs)

    def _calc_wt_interaction_args(self, kwargs):
        """Used for parallel execution"""
        return self.calc_wt_interaction(**kwargs)

    def calc_wt_interaction(self, x_ilk, y_ilk, h_ilk=None, type_i=0, wd=None, ws=None, time=False,
                            WS_eff_ilk=None,
                            n_cpu=1, wd_chunks=None, ws_chunks=1,
                            **kwargs):
        """See WindFarmModel.calc_wt_interaction and additional parameters below

        Parameters
        ----------
        n_cpu : int or None, optional
            Number of CPUs to be used for execution.
            If 1 (default), the execution is not parallized
            If None, the available number of CPUs are used
        wd_chunks : int or None, optional
            If n_cpu>1, the wind directions are divided into <wd_chunks> chunks and executed in parallel.
            If wd_chunks is None, wd_chunks is set to the available number of CPUs
        ws_chunks : int or None, optional
            If n_cpu>1, the wind speeds are divided into <ws_chunks> chunks and executed in parallel.
            If ws_chunks is None, ws_chunks is set to 1
        """

        h_ilk, D_i = self.windTurbines.get_defaults(len(x_ilk), type_i, h_ilk)
        wd, ws = self.site.get_defaults(wd, ws, time)
        I, L, K, = len(x_ilk), len(wd), (1, len(ws))[time is False]
        kwargs.update(dict(x_ilk=x_ilk, y_ilk=y_ilk, h_ilk=h_ilk, wd=wd, ws=ws, time=time,
                           type_i=np.zeros_like(D_i) + type_i))

        for inputModifierModel in self.inputModifierModels:
            kwargs.update(inputModifierModel.setup(**kwargs))

        # Find local wind speed, wind direction, turbulence intensity and probability
        lw = self.site.local_wind(x=np.mean(x_ilk, (1, 2)), y=np.mean(y_ilk, (1, 2)), h=np.mean(h_ilk, (1, 2)),
                                  wd=kwargs['wd'], ws=kwargs['ws'], time=kwargs['time'])
        if time is False:
            assert lw['WS_ilk'].shape[2] == len(ws), \
                "Conflicting sizes. Number of ws must match wind speed dimension of local wind, WS"

        for k in ['WS_ilk', 'WD_ilk', 'TI_ilk']:
            if k in kwargs:
                lw.add_ilk(k, kwargs.pop(k))

        ri, oi = self.windTurbines.function_inputs

        kwargs.update({'WD_ilk': lw.WD_ilk,
                       'WS_ilk': lw.WS_ilk,
                       'WS_eff_ilk': WS_eff_ilk,
                       'D_i': D_i,
                       'I': I, 'L': L, 'K': K,
                       **{k + '_ilk': self.site.interp(self.site.ds[k], lw) for k in ri + oi if k in self.site.ds},
                       })
        if hasattr(lw, 'TI_ilk'):
            kwargs['TI_ilk'] = lw.TI_ilk
            kwargs['TI_eff_ilk'] = lw.TI_ilk + 0.  # autograd-friendly copy
        if 'time' in lw:
            kwargs['time'] = lw.time

        self._check_input(kwargs)

        if n_cpu != 1 or wd_chunks or ws_chunks > 1:
            # parallel execution
            map_func, arg_lst, wd_chunks, ws_chunks = self._multiprocessing_chunks(
                n_cpu=n_cpu, wd_chunks=wd_chunks, ws_chunks=ws_chunks,
                **kwargs)

            WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, _, kwargs = list(
                zip(*map_func(self._calc_wt_interaction_args, arg_lst)))

            def concatenate(v_ilk):
                v_ilk = [fix_shape(v, WS_eff.shape) for v, WS_eff in zip(v_ilk, WS_eff_ilk)]
                if kwargs[0]['time'] is False:
                    return np.concatenate([np.concatenate(v_ilk[i::ws_chunks], axis=1)
                                           for i in range(ws_chunks)], axis=2)
                else:
                    return np.concatenate(v_ilk, axis=1)

            return ([concatenate(v) for v in [WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk]] +
                    [lw,
                     {'type_i': kwargs[0]['type_i'],
                      **{k: concatenate([wt_i[k] for wt_i in kwargs]) for k in kwargs[0] if k.endswith('_ilk')}}])

        # Calculate down-wind and cross-wind distances
        WS_eff_ilk, TI_eff_ilk, ct_ilk, kwargs = self._calc_wt_interaction(**kwargs)

        power_ilk = self.windTurbines.power(ws=WS_eff_ilk, **self.get_wt_kwargs(TI_eff_ilk, kwargs))
        kwargs.update({'time': time, 'type_i': type_i})
        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, lw, kwargs

    @abstractmethod
    def _calc_wt_interaction(self, **kwargs):
        """calculate WT interaction"""

    def get_map_args(self, x_j, y_j, h_j, lw, wd, ws, sim_res_data, D_dst=0):
        type_i = sim_res_data.type.values
        wt_d_i = self.windTurbines.diameter(type_i)
        if 'time' in sim_res_data:
            time = np.atleast_1d(sim_res_data['time'].values)
            map_arg_funcs = {'time': lambda l, j: time[l]}
        else:
            time = False
            map_arg_funcs = {}
        if 'wd' in sim_res_data.dims:
            sim_res_data = sim_res_data.sel(wd=wd)
        wt_x_ilk = sim_res_data['x'].ilk()
        WD_il = sim_res_data.WD.ilk()

        lw_j = self.site.local_wind(x=x_j, y=y_j, h=h_j, wd=wd, ws=ws, time=time)
        I, J, L, K = [len(x) for x in [wt_x_ilk, x_j, wd, ws]]
        if time is not False:
            K = 1

        def get_ilk(k):
            v = sim_res_data[k].ilk()

            def wrap(l, j):
                l_ = [l, slice(0, 1)][v.shape[1] == 1]
                return v[:, l_]
            return wrap
        map_arg_funcs.update({k.replace('CT', 'ct') + '_ilk': get_ilk(k)
                              for k in sim_res_data if k not in ['wd_bin_size', 'ws_l', 'ws_u']})
        map_arg_funcs.update({
            'type_il': lambda l, j: type_i[:, na],
            'D_src_il': lambda l, j: wt_d_i[:, na],
            'D_dst_ijl': lambda l, j: np.zeros((1, 1, 1)) + D_dst,
            'IJLK': lambda l=slice(None), j=slice(None), I=I, J=J, L=L, K=K: (I, len(np.arange(J)[j]), len(np.arange(L)[l]), K)})

        for k in ['WS', 'WD', 'TI']:
            if k in sim_res_data:
                if k + '_ilk' in lw.overwritten:
                    if 'wt' not in sim_res_data[k].dims and 'i' not in sim_res_data[k].dims:
                        lw_j.add_ilk(k + "_ilk", sim_res_data[k])
                    else:
                        warnings.warn(
                            f"The WT dependent {k} that was provided for the simulation is not available at the flow map points and therefore ignored")
        return map_arg_funcs, lw_j, WD_il

    def _get_flow_l(self, model_kwargs, wt_x_ilk, wt_y_ilk, wt_h_ilk, x_jl, y_jl, h_jl, wd, WD_ilk, WS_jlk, TI_jlk):
        dw_ijlk, hcw_ijlk, dh_ijlk = self.site.distance(wt_x_ilk, wt_y_ilk, wt_h_ilk, wd_l=wd, WD_ilk=WD_ilk,
                                                        time=model_kwargs.get('time', False), dst_xyh_jlk=(x_jl, y_jl, h_jl))

        if self.wec != 1:
            hcw_ijlk = hcw_ijlk / self.wec

        if self.deflectionModel:
            dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, dh_ijlk=dh_ijlk, z_ijlk=wt_h_ilk[:, na] + dh_ijlk,
                **model_kwargs)

        model_kwargs.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk,
                             'x_ilk': wt_x_ilk, 'y_ilk': wt_y_ilk})

        if 'cw_ijlk' in self.args4all:
            model_kwargs['cw_ijlk'] = gradients.hypot(dh_ijlk, hcw_ijlk)

        if 'wake_radius_ijlk' in self.args4all:
            model_kwargs['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**model_kwargs)

        if 'wake_radius_ijl' in self.args4all:
            model_kwargs['wake_radius_ijl'] = self.wake_deficitModel.wake_radius(**model_kwargs)[..., 0]
        if 'z_ijlk' in self.args4all:
            model_kwargs['z_ijlk'] = wt_h_ilk[:, na] + dh_ijlk
        if 'WS_jlk' in self.args4all:
            model_kwargs['WS_jlk'] = WS_jlk

        # ===============================================================================================================
        # Calculate deficit
        # ===============================================================================================================
        if isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum)):
            deficit_ijlk, deficit_centre_ijlk, uc_ijlk, sigma_sqr_ijlk, blockage_ijlk = self._calc_deficit_convection(
                **model_kwargs)
        else:
            deficit_ijlk, blockage_ijlk = self._calc_deficit(**model_kwargs)

        # ===============================================================================================================
        # Replace deficit from external farms
        # ===============================================================================================================
        for i, ewf in enumerate(self.externalWindFarms, len(wt_x_ilk) - len(self.externalWindFarms)):
            deficit_ijlk[i] = ewf(i=i, l=np.ones(len(wd), dtype=bool), deficit_jlk=None,
                                  **model_kwargs,
                                  dst_xyh_jlk=[x_jl[:, :, na], y_jl[:, :, na], h_jl[:, :, na]]
                                  )
        # ===============================================================================================================
        # Calculate added Turbulence
        # ===============================================================================================================
        if self.turbulenceModel:
            add_turb_ijlk = self._calc_added_turbulence(**model_kwargs)

        # ===============================================================================================================
        # Sum up deficits
        # ===============================================================================================================
        sp_kwargs = dict(deficit_jxxx=deficit_ijlk)
        if isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum)):
            if isinstance(self.superpositionModel, WeightedSum):
                sp_kwargs.update({'WS_xxx': WS_jlk,
                                  'convection_velocity_jxxx': uc_ijlk,
                                  'deficit_centre_jxxx': deficit_centre_ijlk})
            else:
                sp_kwargs.update({'WS0_xxx': model_kwargs['WS_ilk'] * np.ones_like(model_kwargs['WS_eff_ilk']),
                                  'WS_eff_xxx': model_kwargs['WS_eff_ilk'], 'ct_xxx': model_kwargs['ct_ilk'],
                                  'D_xx': model_kwargs['D_src_il']})
                sigma_sqr_ijlk = sigma_sqr_ijlk * (dw_ijlk > 1e-10)
            sp_kwargs.update(dict(sigma_sqr_jxxx=sigma_sqr_ijlk, cw_jxxx=model_kwargs['cw_ijlk'],
                                  hcw_jxxx=hcw_ijlk, dh_jxxx=dh_ijlk))

        WS_eff_jlk = WS_jlk - self.superpositionModel.superpose_deficit(**sp_kwargs)
        if self.blockage_deficitModel:
            WS_eff_jlk -= self.blockage_deficitModel.superpositionModel(blockage_ijlk)

        # ===============================================================================================================
        # Sum up added Turbulence
        # ===============================================================================================================
        if self.turbulenceModel:
            TI_eff_jlk = self.turbulenceModel.calc_effective_TI(TI_jlk, add_turb_ijlk)
        else:
            TI_eff_jlk = None
        model_kwargs.clear()
        return WS_eff_jlk, TI_eff_jlk

    def _aep_map(self, x_jl, y_jl, h_jl, type_j, sim_res_data, memory_GB=1, n_cpu=1):
        lw_j, WS_eff_jlk, _ = self._flow_map(x_jl, y_jl, h_jl, sim_res_data.localWind, sim_res_data.wd.values,
                                             sim_res_data.ws.values, sim_res_data, memory_GB=memory_GB, n_cpu=n_cpu)
        power_kwargs = {}
        if 'type' in (self.windTurbines.powerCtFunction.required_inputs +
                      self.windTurbines.powerCtFunction.optional_inputs):
            power_kwargs['type'] = type_j
        power_jlk = self.windTurbines.power(WS_eff_jlk, **power_kwargs)

        aep_j = (power_jlk * lw_j.P_ilk).sum((1, 2))
        return aep_j * 365 * 24 * 1e-9

    def _flow_map(self, x_jl, y_jl, h_jl, lw, wd, ws, sim_res_data, D_dst=0, memory_GB=1, n_cpu=1, verbose=None):
        """call this function via SimulationResult.flow_map"""
        wd, ws = np.atleast_1d(wd), np.atleast_1d(ws)
        arg_funcs, lw_j, WD_il = self.get_map_args(x_jl, y_jl, h_jl, lw, wd, ws, sim_res_data, D_dst=D_dst)
        I, J, L, K = arg_funcs['IJLK']()

        if I == 0:
            return (lw_j, np.broadcast_to(lw_j.WS_ilk, (len(x_jl), L, K)).astype(float),
                    np.broadcast_to(lw_j.TI_ilk, (len(x_jl), L, K)).astype(float))
        # *6=dx_ijlk, dy_ijlk, dz_ijlk, dh_ijlk, deficit, blockage
        size_GB = I * J * L * K * np.array([]).itemsize * 6 * n_cpu / 1024**3
        min_wd_chunks = np.minimum(n_cpu, L)
        wd_chunks = int(np.clip(np.ceil(size_GB / memory_GB), min_wd_chunks, L))
        min_j_chunks = np.minimum(int(np.ceil(n_cpu / wd_chunks)), J)
        j_chunks = int(np.clip(np.ceil(size_GB / wd_chunks / memory_GB), min_j_chunks, J))

        verbose = verbose or self.verbose
        map_func = get_map_func(n_cpu=n_cpu, starmap=True, verbose=wd_chunks + j_chunks > 2 and verbose,
                                desc='Calculate flow map', unit='chunk', leave=0)
        wt_x_ilk, wt_y_ilk, wt_h_ilk = [sim_res_data[k].ilk() for k in ['x', 'y', 'h']]

        def get_jl_args(j, l):
            js, ls = slice(j[0], j[-1] + 1), slice(l[0], l[-1] + 1)
            if x_jl.shape[1] == 1:
                ls = slice(None)

            return ({k: arg_funcs[k](l, j) for k in arg_funcs},
                    *[(v, v[:, slice(l[0], l[-1] + 1)])[np.shape(v)[1] == L] for v in [wt_x_ilk, wt_y_ilk, wt_h_ilk]],
                    x_jl[js, ls], y_jl[js, ls], np.broadcast_to(h_jl, x_jl.shape)[js, ls], wd[l], WD_il[:, l],
                    lw_j.WS_ilk[(j, [0])[lw_j.WS_ilk.shape[0] == 1]][:, (l, [0])[lw_j.WS_ilk.shape[1] == 1]],
                    lw_j.TI_ilk[(j, [0])[lw_j.TI_ilk.shape[0] == 1]][:, (l, [0])[lw_j.TI_ilk.shape[1] == 1]])

        l_iter = [get_jl_args(j, l)
                  for l in np.array_split(np.arange(L).astype(int), wd_chunks)
                  for j in np.array_split(np.arange(J).astype(int), j_chunks)]
        WS_eff_jlk, TI_eff_jlk = zip(*map_func(self._get_flow_l, l_iter))
        WS_eff_jlk = np.concatenate([np.concatenate(WS_eff_jlk[i * j_chunks:(i + 1) * j_chunks], 0)
                                     for i in range(wd_chunks)], 1)
        if self.turbulenceModel:
            TI_eff_jlk = np.concatenate([np.concatenate(TI_eff_jlk[i * j_chunks:(i + 1) * j_chunks], 0)
                                         for i in range(wd_chunks)], 1)
        else:
            TI_eff_jlk = np.zeros_like(WS_eff_jlk) + lw_j.TI_ilk
        return lw_j, WS_eff_jlk, TI_eff_jlk

    def _check_input(self, kwargs):
        x_ilk, y_ilk, h_ilk = kwargs['x_ilk'], kwargs['y_ilk'], kwargs['h_ilk']
        i1, i2, *_ = np.where((cabs(x_ilk[:, na] - x_ilk[na]) +
                               cabs(y_ilk[:, na] - y_ilk[na]) +
                               cabs(h_ilk[:, na] - h_ilk[na]) +
                               np.eye(len(x_ilk))[:, :, na, na]) == 0)
        if len(i1):
            msg = "\n".join(["Turbines %d and %d are at the same position" % (i1[i], i2[i]) for i in range(len(i1))])
            raise ValueError(msg)
        for k in self.args4all:
            if k.endswith('_ilk') and k not in ['ct_ilk'] and k not in kwargs:
                n = k.replace('_ilk', '')
                needed_by = str(self)
                for model in [self.wake_deficitModel, self.superpositionModel, self.blockage_deficitModel,
                              self.deflectionModel, self.turbulenceModel] + self.inputModifierModels:
                    if ((hasattr(model, 'args4model') and k in model.args4model) or
                            (hasattr(model, 'args4deficit') and k in model.args4deficit)):
                        needed_by = model.__class__.__name__
                        break
                raise ValueError(f"'{n}' needed by {needed_by} is missing")
        ri, oi = self.windTurbines.function_inputs
        for k in kwargs:
            n = k.replace('_ilk', '').replace('_i', '')
            if (n not in ri + oi and k not in self.args4all and
                    n not in {'x', 'y', 'h', 'wd', 'ws', 'time', 'type', 'D', 'WD', 'WS',
                              'WS_eff', 'TI', 'TI_eff', 'I', 'L', 'K'}):
                raise ValueError(f"WindFarmModel an got unexpected keyword argument: '{n}'")


class PropagateUpDownIterative(EngineeringWindFarmModel):
    """Downstream wake deficits calculated and propagated in downstream direction.
    """

    def __init__(self, site, windTurbines, wake_deficitModel,
                 superpositionModel=LinearSum(),
                 blockage_deficitModel=None,
                 deflectionModel=None, turbulenceModel=None, rotorAvgModel=None,
                 inputModifierModels=[], convergence_tolerance=1e-6, max_iter=10):
        """Initialize flow model

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        wake_deficitModel : DeficitModel
            Model describing the wake(downstream) deficit
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        superpositionModel : SuperpositionModel
            Model defining how deficits sum up
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel, rotorAvgModel,
                                          blockage_deficitModel=blockage_deficitModel, deflectionModel=deflectionModel,
                                          turbulenceModel=turbulenceModel, inputModifierModels=inputModifierModels)
        msg = 'PropagateUpDownIterative requires a wake deficit model that scales with the effective wind speed. For most models this can be achieved by setting the argument use_effective_ws=True'
        assert isinstance(wake_deficitModel, NoWakeDeficit) or wake_deficitModel.WS_key == 'WS_eff_ilk', msg
        msg = "PropagateUpDownIterative only works with blockage deficit models that scales with the effective wind speed. For most models this can be achieved by setting the argument use_effective_ws=True"
        assert blockage_deficitModel is None or blockage_deficitModel.WS_key == 'WS_eff_ilk', msg
        self.convergence_tolerance = convergence_tolerance
        self.max_iter = max_iter

    def _calc_wt_interaction(self, wd, WS_eff_ilk,
                             **kwargs):
        WS_ilk = kwargs.pop('WS_ilk')
        blockage_deficit = WS_ilk * 0.
        I = kwargs['I']
        dw_order_indices_ld = self.site.distance.dw_order_indices(kwargs['x_ilk'], kwargs['y_ilk'], wd)[:, 0]

        if self.blockage_deficitModel:
            # use linear sum as default blockage superpositionModel
            msg = "PropagateDownwind does not work with SquaredSum when the blockage model has both up and downstream effects"
            assert self.blockage_deficitModel.upstream_only or not isinstance(
                self.blockage_deficitModel.superpositionModel, SquaredSum), msg

        for j in tqdm(range(I), disable=I <= 1 or not self.verbose,
                      desc="Calculate flow interaction (PropagateUpDownIterative)", unit="wt"):
            # wake deficit
            self.direction = 'down'
            WS_eff_wake_ilk, TI_eff_ilk, ct_ilk, res_kwargs = self._propagate_deficit(
                wd, dw_order_indices_ld,
                WS_ilk - blockage_deficit, **kwargs)
            wake_deficit = (WS_ilk - blockage_deficit) - WS_eff_wake_ilk

            if j == 0:
                # set initial value to wakes-only
                WS_eff_ilk_last = WS_ilk - wake_deficit
                # initialize
                diff_ilk_last = WS_eff_ilk_last * 1e6
                iconverging = np.full(WS_eff_ilk_last.shape, True)

            # blockage deficit
            self.direction = 'up'
            WS_eff_blockage_ilk = self._propagate_deficit(wd, dw_order_indices_ld[:, ::-1],
                                                          WS_ilk - wake_deficit, **kwargs)[0]
            blockage_deficit = (WS_ilk - wake_deficit) - WS_eff_blockage_ilk
            WS_eff_ilk = WS_ilk - wake_deficit - blockage_deficit

            # check convergence of local turbine inflow wind speed
            diff_ilk = cabs(WS_eff_ilk_last - WS_eff_ilk)
            # compare convergence rate to previous iteration to check if converging
            iconverging &= (diff_ilk - diff_ilk_last) <= 0.
            # compute max difference of all converging cases
            max_diff = diff_ilk[iconverging].max()
            # convergence check
            if (max_diff < self.convergence_tolerance) or ((j + 1) >= self.max_iter):
                break
            # update and keep iterating
            diff_ilk_last = diff_ilk
            WS_eff_ilk_last = WS_eff_ilk

        self.direction = 'down'
        self.iterations = j
        return WS_eff_ilk, TI_eff_ilk, ct_ilk, res_kwargs

    def _calc_deficit(self, dw_ijlk, **kwargs):
        """Calculate wake (and blockage) deficit"""
        if self.direction == 'up':
            deficit = dw_ijlk * 0
            deficit, blockage = self._add_blockage(deficit, dw_ijlk, **kwargs)
        else:
            deficit, blockage = EngineeringWindFarmModel._calc_deficit(self, dw_ijlk=dw_ijlk, **kwargs)

        return deficit, blockage

    def _propagate_deficit(self, wd,
                           wt_order_indices_ld,
                           WS_ilk,
                           TI_eff_ilk,
                           D_i,
                           I, L, K, **kwargs):
        """
        Additional suffixes:

        - m: turbines and wind directions (il.flatten())
        - n: from_turbines, to_turbines and wind directions (iil.flatten())

        """

        deficit_nk = []
        deficit_centre_nk = []
        blockage_nk = []
        uc_nk = []
        sigma_sqr_nk = []
        cw_nk = []
        hcw_nk = []
        dh_nk = []

        def ilk2mk(v_ilk):
            dtype = (float, np.complex128)[np.iscomplexobj(v_ilk)]
            _K = np.shape(v_ilk)[2]
            return np.broadcast_to(np.asarray(v_ilk).astype(dtype), (I, L, _K)).reshape((I * L, _K))

        WS_mk = ilk2mk(WS_ilk)
        WD_mk, TI_mk, h_mk = [ilk2mk(kwargs[k + '_ilk']) for k in ['WD', 'TI', 'h']]
        WS_eff_mk, TI_eff_mk = [], []
        yaw_mk = ilk2mk(kwargs.get('yaw_ilk', [[[0]]]))
        tilt_mk = ilk2mk(kwargs.get('tilt_ilk', [[[0]]]))
        modified_input_dict_mk = []
        WS_free_mk = []
        D_mk = []
        ct_jlk = []

        if self.turbulenceModel:
            add_turb_nk = []

        i_wd_l = np.arange(L).astype(int)

        wt_kwargs = self.get_wt_kwargs(TI_eff_ilk, kwargs)

        # Iterate over turbines in down wind order
        for j in tqdm(range(I), disable=I <= 1 or not self.verbose,
                      desc=f"Calculate flow interaction (propagate_deficit {self.direction})", unit="wt"):
            i_wt_l = wt_order_indices_ld[:, j]
            # current wt (j'th most upstream wts for all wdirs)
            m = i_wt_l * L + i_wd_l

            # Calculate effectiv wind speed at current turbines(all wind directions and wind speeds) and
            # look up power and thrust coefficient
            if j == 0:  # Most upstream turbines (no wake)
                WS_eff_lk = WS_mk[m]
                WS_eff_mk.append(np.broadcast_to(WS_eff_lk, (L, K)))
                if self.turbulenceModel:
                    TI_eff_lk = TI_mk[m]
                    TI_eff_mk.append(np.broadcast_to(TI_eff_lk, (L, K)))
            else:  # 2..n most upstream turbines (wake)
                def get_value2WT(value_nk):
                    """Get value input to current turbine, j
                    value_nk triangular is a list j elements.
                    First element contains e.g. the defict to
                    """
                    return np.array([d_nk2[i] for d_nk2, i in zip(value_nk, range(j)[::-1])])

                sp_kwargs = {'deficit_jxxx': get_value2WT(deficit_nk)}
                if isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum)):
                    sp_kwargs.update({k: get_value2WT(v_nk) for k, v_nk in [('sigma_sqr_jxxx', sigma_sqr_nk),
                                                                            ('cw_jxxx', cw_nk),
                                                                            ('hcw_jxxx', hcw_nk),
                                                                            ('dh_jxxx', dh_nk)]})

                    if isinstance(self.superpositionModel, WeightedSum):
                        sp_kwargs.update({'WS_xxx': WS_mk[m],
                                          'convection_velocity_jxxx': get_value2WT(uc_nk),
                                          'deficit_centre_jxxx': get_value2WT(deficit_centre_nk)})
                    else:
                        sp_kwargs.update({'WS0_xxx': np.array(WS_free_mk),
                                          'WS_eff_xxx': np.array(WS_eff_mk),
                                          'ct_xxx': np.array(ct_jlk),
                                          'D_xx': np.array(D_mk)})
                WS_eff_lk = WS_mk[m]
                if self.direction == 'down':
                    WS_eff_lk = WS_eff_lk - self.superpositionModel.superpose_deficit(**sp_kwargs)
                if self.blockage_deficitModel:
                    WS_eff_lk = WS_eff_lk - self.blockage_deficitModel.superpositionModel(get_value2WT(blockage_nk))
                WS_eff_mk.append(WS_eff_lk)

                if self.turbulenceModel:
                    add_turb2WT = np.array([d_nk2[i] for d_nk2, i in zip(add_turb_nk, range(j)[::-1])])
                    TI_eff_lk = self.turbulenceModel.calc_effective_TI(TI_mk[m], add_turb2WT)
                    TI_eff_mk.append(TI_eff_lk)
            # assemble free wind speed (ask mmpe why it is not) and diameter to allow cumulative superposition
            WS_free_mk.append(WS_mk[m])
            D_mk.append(D_i[i_wt_l])

            # Calculate Power/CT
            def mask(k, v):
                if len(np.squeeze(v).shape) == 0:
                    return np.squeeze(v)
                v = np.asarray(v)
                if v.shape[:2] == (I, L):
                    return v[i_wt_l, i_wd_l]
                elif v.shape[0] == I:
                    return v[i_wt_l].flatten()
                else:
                    assert v.shape[1] == L
                    return v[0, i_wd_l]

            _wt_kwargs = {k: mask(k, v) for k, v in wt_kwargs.items()}
            if 'TI_eff' in _wt_kwargs:
                _wt_kwargs['TI_eff'] = TI_eff_mk[-1]

            ct_lk = self.windTurbines.ct(WS_eff_lk, **_wt_kwargs)

            ct_jlk.append(np.broadcast_to(ct_lk, (L, K)))

            if j < I - 1 or len(self.inputModifierModels):
                i_dw = wt_order_indices_ld[:, j + 1:]

                # Calculate required args4deficit parameters
                arg_funcs = {'WS_ilk': lambda: WS_mk[m][na],
                             'WS_jlk': lambda: np.moveaxis(np.array(
                                 [WS_ilk[(slice(0, 1), j)[WS_ilk.shape[0] > 1], (0, l)[WS_ilk.shape[1] > 1]]
                                  for j, l in zip(i_dw, i_wd_l)]), 0, 1),
                             'WS_eff_ilk': lambda: WS_eff_mk[-1][na],
                             'TI_ilk': lambda: TI_mk[m][na],
                             'TI_eff_ilk': lambda: TI_eff_mk[-1][na],
                             'D_src_il': lambda: D_i[i_wt_l][na],
                             'yaw_ilk': lambda: yaw_mk[m][na],
                             'tilt_ilk': lambda: tilt_mk[m][na],
                             'D_dst_ijl': lambda: D_i[wt_order_indices_ld[:, j + 1:]].T[na],
                             'h_ilk': lambda: h_mk[m][na],
                             'ct_ilk': lambda: ct_lk[na],
                             'IJLK': lambda: (1, i_dw.shape[1], L, K),
                             'WD_ilk': lambda: WD_mk[m][na],
                             **{k + '_ilk': lambda k=k: ilk2mk(kwargs[k + '_ilk'])[m][na] for k in 'xyh'},
                             'type_il': lambda: kwargs['type_i'][i_wt_l][na]

                             }
                model_kwargs = {k: arg_funcs[k]() for k in self.args4all if k in arg_funcs}

                # custom model arguments
                custom_args = (set([k for k in self.args4all if k.endswith('_ilk')]) - set(model_kwargs)) & set(kwargs)
                model_kwargs.update({k: ilk2mk(kwargs[k])[m][na] for k in custom_args})

                def get_pos(v_ilk, i_wt_l, i_wd_l):
                    return v_ilk[i_wt_l, np.minimum(i_wd_l, v_ilk.shape[1] - 1).astype(int)]
                dst_xyh_jlk = [get_pos(kwargs[k + '_ilk'], i_dw.T, i_wd_l) for k in 'xyh']
                dw_ijlk, hcw_ijlk, dh_ijlk = self.site.distance(
                    *[get_pos(kwargs[k + '_ilk'], i_wt_l, i_wd_l)[na] for k in 'xyh'], wd_l=wd, WD_ilk=WD_mk[m][na],
                    time=kwargs.get('time', False), dst_xyh_jlk=dst_xyh_jlk)

                for inputModidifierModel in self.inputModifierModels:
                    modified_input_dict = inputModidifierModel(**model_kwargs)
                    modified_input_dict_mk.append(modified_input_dict)
                    model_kwargs.update(modified_input_dict)

                if self.wec != 1:
                    hcw_ijlk = hcw_ijlk / self.wec

                if self.deflectionModel:
                    dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                        dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, dh_ijlk=dh_ijlk, **model_kwargs)

                model_kwargs.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk})
                if 'z_ijlk' in self.args4all:
                    model_kwargs['z_ijlk'] = h_mk[m][na, na] + dh_ijlk

                hcw_nk.append(hcw_ijlk[0])
                dh_nk.append(dh_ijlk[0])

                if 'cw_ijlk' in self.args4all:
                    # sqrt(a**2+b**2) as hypot does not support complex numbers
                    model_kwargs['cw_ijlk'] = np.sqrt(dh_ijlk**2 + hcw_ijlk**2)
                    cw_nk.append(model_kwargs['cw_ijlk'][0])

                if 'wake_radius_ijl' in self.args4all:
                    model_kwargs['wake_radius_ijl'] = self.wake_deficitModel.wake_radius(**model_kwargs)[..., 0]

                if 'wake_radius_ijlk' in self.args4all:
                    model_kwargs['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**model_kwargs)

                # ======================================================================================================
                # Calculate deficit
                # ======================================================================================================
                if isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum)) and self.direction == 'down':
                    # only cw needs to be rotor averaged as remaining super position input is
                    # the same all over the rotor
                    if self.wake_deficitModel.rotorAvgModel:
                        cw_nk[-1] = (self.wake_deficitModel.rotorAvgModel(lambda **kwargs: kwargs['cw_ijlk'],
                                                                          **model_kwargs))[0]
                    if isinstance(self.superpositionModel, WeightedSum):
                        deficit, deficit_centre, uc, sigma_sqr, _ = self._calc_deficit_convection(**model_kwargs)
                        uc_nk.append(uc[0])
                        sigma_sqr_nk.append(sigma_sqr[0])
                        deficit_centre_nk.append(deficit_centre[0])
                    elif isinstance(self.superpositionModel, CumulativeWakeSum):
                        # only sigma needed in cumulative wake model, centerline deficit computed inside superpostion model
                        # sigma set to zero upstream to ensure downwind activation only
                        sigma_sqr = (self.wake_deficitModel.sigma_ijlk(**model_kwargs))**2 * (dw_ijlk > 1e-10)
                        sigma_sqr_nk.append(sigma_sqr[0])
                        deficit = np.zeros_like(sigma_sqr)
                else:
                    deficit, blockage = self._calc_deficit(**model_kwargs)

                    for i, ewf in enumerate(self.externalWindFarms, I - len(self.externalWindFarms)):
                        # print(i)
                        deficit = ewf(i=0, l=i_wt_l == i, deficit_jlk=deficit[0],
                                      **model_kwargs, dst_xyh_jlk=dst_xyh_jlk)[na]

                    if self.blockage_deficitModel:
                        blockage_nk.append(blockage[0])
                deficit_nk.append(deficit[0])

                # Calculate added turbulence intensity.
                if self.turbulenceModel:
                    add_turb_nk.append(self.turbulenceModel(**model_kwargs)[0])

        WS_eff_jlk, ct_jlk = np.array(WS_eff_mk), np.array(ct_jlk)

        wt_inv_indices = (np.argsort(wt_order_indices_ld, 1).T * L + np.arange(L).astype(int)[na]).flatten()
        WS_eff_ilk = WS_eff_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))

        ct_ilk = ct_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))
        if self.turbulenceModel:
            TI_eff_jlk = np.array(TI_eff_mk)
            TI_eff_ilk = TI_eff_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))

        if len(self.inputModifierModels):
            for k in modified_input_dict_mk[0].keys():
                mi_jlk = np.array([mi_dict[k] for mi_dict in modified_input_dict_mk])
                kwargs[k] = mi_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))

        return WS_eff_ilk, TI_eff_ilk, ct_ilk, kwargs


class PropagateDownwind(PropagateUpDownIterative):
    """Downstream wake deficits calculated and propagated in downstream direction.
    Very fast, but ignoring blockage effects
    """

    def __init__(self, site, windTurbines, wake_deficitModel,
                 superpositionModel=LinearSum(),
                 deflectionModel=None, turbulenceModel=None, rotorAvgModel=None,
                 inputModifierModels=[],
                 externalWindFarms=[]):
        """Initialize flow model

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        wake_deficitModel : DeficitModel
            Model describing the wake(downstream) deficit
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        superpositionModel : SuperpositionModel
            Model defining how deficits sum up
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel, rotorAvgModel,
                                          blockage_deficitModel=None, deflectionModel=deflectionModel,
                                          turbulenceModel=turbulenceModel, inputModifierModels=inputModifierModels,
                                          externalWindFarms=externalWindFarms)
        self.direction = 'down'

    def _calc_deficit(self, dw_ijlk, **kwargs):
        return EngineeringWindFarmModel._calc_deficit(self, dw_ijlk, **kwargs)

    def _calc_wt_interaction(self, wd, WS_eff_ilk, **kwargs):
        WS_ilk = kwargs.pop('WS_ilk')

        dw_order_indices_ld = self.site.distance.dw_order_indices(kwargs['x_ilk'], kwargs['y_ilk'], wd)[:, 0]
        return self._propagate_deficit(wd, dw_order_indices_ld, WS_ilk, **kwargs)


# import here for backward compatibility
from py_wake.wind_farm_models.all2alliterative import All2AllIterative, All2All  # noqa


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt

        from py_wake.deficit_models.gaussian import ZongGaussianDeficit
        from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
        from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site
        from py_wake.flow_map import XYGrid
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        from py_wake.wind_farm_models.all2alliterative import All2AllIterative

        site = IEA37Site(16)
        x, y = site.initial_position.T

        windTurbines = IEA37_WindTurbines()
        from py_wake.deficit_models.noj import NOJDeficit
        from py_wake.superposition_models import SquaredSum

        # NOJ wake model
        noj = PropagateDownwind(site, windTurbines, wake_deficitModel=NOJDeficit(), superpositionModel=SquaredSum())

        # NOJ wake and selfsimilarity blockage
        noj_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NOJDeficit(), superpositionModel=SquaredSum(),
                                  blockage_deficitModel=SelfSimilarityDeficit(upstream_only=True))

        # Zong convection superposition
        zongp = PropagateDownwind(site, windTurbines, wake_deficitModel=ZongGaussianDeficit(), superpositionModel=WeightedSum(),
                                  turbulenceModel=STF2017TurbulenceModel())

        # Zong convection superposition
        zong_ss = All2AllIterative(site, windTurbines, wake_deficitModel=ZongGaussianDeficit(), superpositionModel=WeightedSum(),
                                   blockage_deficitModel=SelfSimilarityDeficit(), turbulenceModel=STF2017TurbulenceModel())

        # Cumulativ wake superposition
        cwp = PropagateDownwind(site, windTurbines, wake_deficitModel=ZongGaussianDeficit(), superpositionModel=CumulativeWakeSum(),
                                turbulenceModel=STF2017TurbulenceModel())

        # Cumulativ wake superposition
        cw_ss = All2AllIterative(site, windTurbines, wake_deficitModel=ZongGaussianDeficit(), superpositionModel=CumulativeWakeSum(),
                                 blockage_deficitModel=SelfSimilarityDeficit(), turbulenceModel=STF2017TurbulenceModel())

        for wm in [noj, noj_ss, zongp, zong_ss, cwp, cw_ss]:
            sim = wm(x=x, y=y, wd=[30], ws=[9])
            plt.figure()
            sim.flow_map(XYGrid(resolution=200)).plot_wake_map(levels=np.linspace(0, 1, 21) * 9.)
            plt.title(' AEP: %.3f GWh' % sim.aep().sum())
        plt.show()


main()
