import warnings

from numpy import newaxis as na
from tqdm import tqdm

from py_wake import np
from py_wake.superposition_models import CumulativeWakeSum, LinearSum, WeightedSum
from py_wake.utils import gradients
from py_wake.utils.gradients import cabs, item_assign
from py_wake.wind_farm_models.engineering_models import (
    EngineeringWindFarmModel,
    PropagateUpDownIterative,
)
import abc
from abc import ABC


class All2AllIterative(EngineeringWindFarmModel):
    """Wake and blockage deficits calculated from all wt to all points of interest (wt/map points).
    The calculations are iteratively repeated until convergence (change of effective wind speed < convergence_tolerance)"""

    def __init__(self, site, windTurbines, wake_deficitModel,
                 superpositionModel=LinearSum(),
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None,
                 rotorAvgModel=None, inputModifierModels=[], externalWindFarms=[],
                 solver=None, **kwargs):
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
        blockage_deficitModel : DeficitModel
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        convergence_tolerance : float or None
            if float: maximum accepted change in WS_eff_ilk [m/s]
            if None: return after first iteration. This only makes sense for benchmark studies where CT,
            wakes and blockage are independent of effective wind speed WS_eff_ilk

        """
        EngineeringWindFarmModel.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel, rotorAvgModel,
                                          blockage_deficitModel=blockage_deficitModel, deflectionModel=deflectionModel,
                                          turbulenceModel=turbulenceModel, inputModifierModels=inputModifierModels,
                                          externalWindFarms=externalWindFarms)
        try:
            kwargs.pop('convergence_tolerance')
            raise ValueError("""The `convergence_tolerance` argument is deprecated. Use the `solver` argument instead, e.g.
`solver=FixedPointSolver(tolerance=1e-6)`""")
        except BaseException:
            assert len(kwargs) == 0, f'Invalid keyword argument(s) provided: {list(kwargs.keys())}'

        self.solver = solver or FixedPointSolver(tolerance=1e-6, verbose=self.verbose)

    def _calc_wt_interaction(self, ws, wd, WD_ilk, WS_ilk, TI_ilk,
                             WS_eff_ilk, TI_eff_ilk,
                             D_i, time,
                             I, L, K, **kwargs):
        if any([np.iscomplexobj(v) for v in ([kwargs.get(k, 0)
                                              for k in ['x_ilk', 'y_ilk', 'h_ilk', 'D_i', 'yaw_ilk', 'tilt_ilk']] +
                                             [ws, wd])]):
            dtype = np.complex128
        else:
            dtype = float
        WS_ILK = np.broadcast_to(WS_ilk, (I, L, K))
        # calculate WS_eff without blockage as a first guess
        if WS_eff_ilk is None:
            # Initialize with PropagateDownwind
            blockage_deficitModel = self.blockage_deficitModel
            self.blockage_deficitModel = None
            dw_order_indices_ld = self.site.distance.dw_order_indices(kwargs['x_ilk'], kwargs['y_ilk'], wd)[:, 0]
            self.direction = 'down'
            WS_eff_ilk, TI_eff_ilk = PropagateUpDownIterative._propagate_deficit(
                self, wd, dw_order_indices_ld, WD_ilk=WD_ilk, WS_ilk=WS_ilk, TI_ilk=TI_ilk,
                WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk, D_i=D_i, I=I, L=L, K=K, time=time, **kwargs)[:2]
            self.blockage_deficitModel = blockage_deficitModel
        elif np.all(WS_eff_ilk == 0):
            WS_eff_ilk = WS_ILK + 0.
            TI_eff_ilk = TI_ilk
        else:
            WS_eff_ilk = np.zeros((I, L, K)) + WS_eff_ilk
            TI_eff_ilk = TI_ilk

        WS_eff_ilk = WS_eff_ilk.astype(dtype)

        dst_xyh_jlk = [kwargs[k + '_ilk'] for k in 'xyh']
        dw_iilk, hcw_iilk, dh_iilk = self.site.distance(*[kwargs[k + '_ilk'] for k in 'xyh'],
                                                        wd_l=wd, WD_ilk=WD_ilk, time=time)
        kwargs['WD_ilk'] = WD_ilk

        wt_kwargs = self.get_wt_kwargs(TI_eff_ilk, kwargs)
        ct_ilk = self.windTurbines.ct(ws=WS_eff_ilk, **wt_kwargs)

        model_kwargs = {'WS_ilk': WS_ilk,
                        'WS_eff_ilk': WS_eff_ilk,
                        'WS_jlk': WS_ilk,
                        'WD_ilk': WD_ilk,
                        'TI_ilk': TI_ilk,
                        'TI_eff_ilk': TI_eff_ilk,
                        'D_src_il': D_i[:, na],
                        'D_dst_ijl': D_i[na, :, na],
                        'dw_ijlk': dw_iilk,
                        'hcw_ijlk': hcw_iilk,
                        'cw_ijlk': np.sqrt(hcw_iilk**2 + dh_iilk**2),
                        'dh_ijlk': dh_iilk,
                        'z_ijlk': kwargs['h_ilk'][:, na] + dh_iilk,
                        'IJLK': (I, I, L, K),
                        'type_il': kwargs['type_i'][:, na],
                        ** kwargs,
                        }
        if 'wake_radius_ijl' in self.args4all:
            model_kwargs['wake_radius_ijl'] = self.wake_deficitModel.wake_radius(**model_kwargs)[:, :, :, 0]

        if not self.deflectionModel:
            self._init_deficit(**model_kwargs)

        cw_iilk = np.sqrt(hcw_iilk**2 + dh_iilk**2)

        i2i_zero = ~np.eye(I).astype(bool)[:, :, na, na]
        self.ct_ilk_lst = []
        self.unstable = np.zeros((I, L, K)).astype(bool)
        max_iter = max(20, I)
        self.iterations = 0

        def get_new_WS_eff_ilk(WS_eff_ilk, TI_eff_ilk, gradient_evaluation=False):
            ct_ilk = self.windTurbines.ct(np.maximum(WS_eff_ilk, 0), **wt_kwargs)
            if not gradient_evaluation:
                self.iterations += 1
                if self.iterations > max_iter:
                    raise StopIteration()

                if len(self.ct_ilk_lst) > 1:
                    # consider last max 6 iterations
                    N = 6
                    ct_change = np.diff(np.array(self.ct_ilk_lst[-N:] + [ct_ilk]), axis=0)

                    # max ct step (cummax)
                    mc = [np.real(cabs(ct_change[-1]))]
                    for ct_c in ct_change[::-1][1:]:
                        mc.append(np.maximum(mc[-1], np.real(cabs(ct_c))))

                    # actual change in ct compared to previuos iterations
                    ac = np.real(cabs(np.cumsum(ct_change[::-1], 0)))
                    sc = np.real(np.cumsum(cabs(ct_change[::-1]), 0))  # sum of step changes
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in divide')
                        unstable = ((ac / sc < 0.1) &  # change in ct < 10% of step sum
                                    (sc / mc > 2.8))  # step sum > 2.5 x max step, e.g. up-down-up
                    self.unstable |= np.any(unstable, 0)
                    # for unstable turbines, ct can only go down (typically shut-down)
                    ct_ilk = np.where(self.unstable, np.minimum(self.ct_ilk_lst[-1], ct_ilk), ct_ilk)
                self.ct_ilk_lst.append(ct_ilk)

            # def p(v): np.round(np.array(v).squeeze(), 3)
            #
            # for i, ct in enumerate(np.array(self.ct_ilk_lst).squeeze().T):
            #     import matplotlib.pyplot as plt
            #     plt.plot(ct, '-', marker=['.', 'x'][int(self.unstable.squeeze()[i])], label=i)
            # plt.legend()
            # plt.show()

            model_kwargs.update(dict(ct_ilk=ct_ilk, WS_eff_ilk=WS_eff_ilk))
            if self.inputModifierModels:
                # x_ilk, y_ilk and h_ilk is may be updated by an inputModifierModel and
                # must be reset in every iterations
                model_kwargs.update(dict(x_ilk=kwargs['x_ilk'], y_ilk=kwargs['y_ilk'], h_ilk=kwargs['h_ilk']))

            if self.deflectionModel:
                model_kwargs.update(dict(
                    # dw_ijlk, hcw_ijlk and dh_ijlk is updated by deflection model and must be reset in every iterations
                    dw_ijlk=dw_iilk,
                    hcw_ijlk=hcw_iilk,
                    cw_ijlk=cw_iilk,
                    dh_ijlk=dh_iilk,
                    z_ijlk=kwargs['h_ilk'][:, na] + dh_iilk))

            modified_input_dict = {}
            for inputModidifierModel in self.inputModifierModels:
                modified_input_dict = inputModidifierModel(**model_kwargs)
                model_kwargs.update(modified_input_dict)
                if any([k in modified_input_dict for k in ['x_ilk', 'y_ilk']]):
                    model_kwargs.update({k: v for k, v in zip(
                        ['dw_ijlk', 'hcw_ijlk', 'dh_ijlk'],
                        self.site.distance(*[model_kwargs[k + '_ilk'] for k in 'xyh'], wd_l=wd, WD_ilk=WD_ilk))})
                    model_kwargs['cw_ijlk'] = gradients.hypot(model_kwargs['dh_ijlk'], model_kwargs['hcw_ijlk'])
                    if not self.deflectionModel:
                        self._init_deficit(**model_kwargs)

            if self.deflectionModel:
                dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(**model_kwargs)
                model_kwargs.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk,
                                     'cw_ijlk': gradients.hypot(dh_ijlk, hcw_ijlk)})
                self._reset_deficit()
            if 'wake_radius_ijlk' in self.args4all:
                model_kwargs['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**model_kwargs)

            if self.turbulenceModel:
                model_kwargs['TI_eff_ilk'] = TI_eff_ilk

            # Calculate deficit
            if isinstance(self.superpositionModel, WeightedSum):
                deficit_iilk, deficit_centre_iilk, uc_iilk, sigmasqr_iilk, blockage_iilk = self._calc_deficit_convection(
                    **model_kwargs)
                deficit_centre_iilk *= i2i_zero
            elif isinstance(self.superpositionModel, CumulativeWakeSum):
                sigmasqr_iilk = (self.wake_deficitModel.sigma_ijlk(**model_kwargs))**2 * \
                    (model_kwargs['dw_ijlk'] > 1e-10)
                deficit_iilk, blockage_iilk = self._calc_deficit(**model_kwargs)
            else:
                deficit_iilk, blockage_iilk = self._calc_deficit(**model_kwargs)

            for i, ewf in enumerate(self.externalWindFarms, I - len(self.externalWindFarms)):
                deficit_jlk = ewf(i=i, l=np.ones(L, dtype=bool), deficit_jlk=None, dst_xyh_jlk=dst_xyh_jlk,
                                  **model_kwargs)
                deficit_iilk = item_assign(deficit_iilk, idx=i, values=deficit_jlk)

            # set own deficit to 0
            deficit_iilk *= i2i_zero
            if blockage_iilk is not None:
                blockage_iilk *= i2i_zero

            sp_kwargs = {'deficit_jxxx': deficit_iilk}
            if isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum)):
                cw_ijlk = model_kwargs['cw_ijlk']
                if self.wake_deficitModel.rotorAvgModel:
                    cw_ijlk = self.wake_deficitModel.rotorAvgModel(lambda **kwargs: kwargs['cw_ijlk'], **model_kwargs)

                sp_kwargs.update({'sigma_sqr_jxxx': sigmasqr_iilk,
                                  'cw_jxxx': cw_ijlk,
                                  'hcw_jxxx': model_kwargs['hcw_ijlk'],
                                  'dh_jxxx': dh_iilk})

                if isinstance(self.superpositionModel, WeightedSum):
                    sp_kwargs.update({'WS_xxx': WS_ilk,
                                      'convection_velocity_jxxx': uc_iilk,
                                      'deficit_centre_jxxx': deficit_centre_iilk})
                else:
                    sp_kwargs.update({'WS0_xxx': WS_ILK,
                                      'WS_eff_xxx': model_kwargs['WS_eff_ilk'],
                                      'ct_xxx': model_kwargs['ct_ilk'],
                                      'D_xx': model_kwargs['D_src_il']})

            WS_eff_ilk = WS_ilk.astype(dtype) - self.superpositionModel.superpose_deficit(**sp_kwargs)
            if self.blockage_deficitModel:
                WS_eff_ilk -= self.blockage_deficitModel.superpositionModel(blockage_iilk)

            if self.turbulenceModel:
                add_turb_ijlk = self.turbulenceModel(**model_kwargs)
                add_turb_ijlk *= i2i_zero
                TI_eff_ilk = self.turbulenceModel.calc_effective_TI(TI_ilk, add_turb_ijlk)

            return WS_eff_ilk, TI_eff_ilk, ct_ilk, modified_input_dict

        WS_eff_ilk, TI_eff_ilk, ct_ilk, modified_input_dict = self.solver(get_new_WS_eff_ilk, WS_eff_ilk, TI_eff_ilk,
                                                                          max_iter)
        # print("All2AllIterative converge after %d iterations" % (j + 1))
        # print(self.iterations, np.abs(get_new_WS_eff_ilk(WS_eff_ilk, TI_eff_ilk)[0] - WS_eff_ilk).max())
        self._reset_deficit()
        kwargs.update({k: modified_input_dict[k] for k in modified_input_dict})
        return WS_eff_ilk, np.broadcast_to(TI_eff_ilk, (I, L, K)), ct_ilk, kwargs


class All2AllIterativeSolver(ABC):
    def __init__(self, no_convergence='warning', debug=False):
        assert no_convergence in ['warning', 'error', 'ignore']
        self.no_convergence = no_convergence
        self.debug = debug

    def __call__(self, get_new_WS_eff, WS_eff_ilk, TI_eff_ilk, max_iter):
        self.diff = []
        self.iteration = 0

        def wrap(WS_eff_ilk_last, TI_eff_ilk_last):

            try:
                WS_eff_ilk, TI_eff_ilk, ct_ilk, modified_input_dict = get_new_WS_eff(WS_eff_ilk_last, TI_eff_ilk_last)
                self.last_result = WS_eff_ilk, TI_eff_ilk, ct_ilk, modified_input_dict

                err = WS_eff_ilk - WS_eff_ilk_last
                self.diff.append([err.max(), err.min()][int(cabs(err.min()) > cabs(err.max()))])
                if self.debug:
                    i, l, k = list(zip(*np.where(np.abs(err) == np.abs(err).max())))[0]
                    print(
                        f"Iteration: {len(self.diff)}, max diff_ilk: {self.diff[-1]:.6f}, ilk: ({i},{l},{k}), WS_eff: {WS_eff_ilk[i, l, k]}")
            except StopIteration:
                msg = f'All2AllIterative did not converge, max WS_eff difference from last iteration {self.diff[-1]}'

                if self.no_convergence == 'warning':
                    warnings.warn(msg)
                elif self.no_convergence == 'error':
                    raise Exception(msg)
                return self.last_result

            return WS_eff_ilk, TI_eff_ilk, ct_ilk, modified_input_dict

        return self._solve(wrap, WS_eff_ilk, TI_eff_ilk, max_iter)

    @abc.abstractmethod
    def _solve(self, get_new_WS_eff, WS_eff_ilk, TI_eff_ilk, max_iter):
        ""

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.diff, '.-')
        plt.axhline(0, color='k')


class ScipyOptimizeSolver(All2AllIterativeSolver):
    def __init__(self, optimizer, no_convergence='warning', debug=False, **kwargs):
        All2AllIterativeSolver.__init__(self, no_convergence, debug)
        self.optimizer = optimizer
        self.kwargs = kwargs

    def _solve(self, get_new_WS_eff, WS_eff_ilk, TI_eff_ilk, max_iter):
        self.TI_eff_ilk = TI_eff_ilk
        I, L, K = WS_eff_ilk.shape

        def f(WS_eff):
            # transform solving f(x)=x to root finding problem: 0 = f(x) - x
            self.WS_eff_ilk, self.TI_eff_ilk = get_new_WS_eff(WS_eff.reshape((I, L, K)), self.TI_eff_ilk)[:2]
            return self.WS_eff_ilk.ravel() - WS_eff

        self.optimizer(f, WS_eff_ilk.ravel(), **{**self.kwargs})
        return self.last_result


class FixedStep():
    def reset(self, solver):
        pass

    def __call__(self, x, x_last, **_):
        return x


class AdaptiveStep(FixedStep):
    def __init__(self, a_min=0.7, a_max=1, a_decr=.1, a_incr=.1, start=10):
        self.a_min = a_min
        self.a_max = a_max
        self.a_decr = a_decr
        self.a_incr = a_incr
        self.start = start

    def reset(self, solver):
        self.a = 1
        self.a_lst = []
        self.solver = solver

    def __call__(self, x, x_last, **_):
        if len(self.solver.diff) >= self.start:
            if np.sign(self.solver.diff[-2:]).sum() == 0:  # diff sign due to overshoot
                self.a = np.maximum(self.a - self.a_decr, self.a_min)
            else:
                self.a = np.minimum(self.a + self.a_incr, self.a_max)

        self.a_lst.append(self.a)
        return (x - x_last) * self.a + x_last


class FixedPointSolver(All2AllIterativeSolver):
    def __init__(self, tolerance=1e-6, step_func=FixedStep(), verbose=False, debug=False, no_convergence='warning'):
        All2AllIterativeSolver.__init__(self, no_convergence)
        self.tolerance = tolerance
        self.step_func = step_func
        self.verbose = verbose
        self.debug = debug

    def _solve(self, get_new_WS_eff, WS_eff_ilk, TI_eff_ilk, max_iter):

        WS_eff_ilk_last = WS_eff_ilk + 0  # fast autograd-friendly copy
        self.step_func.reset(self)
        # Iterate until convergence
        for j in tqdm(range(max_iter + 1), disable=not self.verbose,
                      desc="Calculate flow interaction (All2AllIterative)", unit="Iteration"):
            WS_eff_ilk, TI_eff_ilk, ct_ilk, modified_input_dict = get_new_WS_eff(WS_eff_ilk, TI_eff_ilk)

            max_diff = cabs(self.diff[-1])
            if self.tolerance is None or max_diff < self.tolerance:
                break

            WS_eff_ilk = self.step_func(WS_eff_ilk, WS_eff_ilk_last,
                                        TI_eff_ilk=TI_eff_ilk, get_new_WS_eff=get_new_WS_eff)
            WS_eff_ilk_last = WS_eff_ilk + 0  # fast autograd-friendly copy

        return WS_eff_ilk, TI_eff_ilk, ct_ilk, modified_input_dict

    def plot(self):
        All2AllIterativeSolver.plot(self)
        import matplotlib.pyplot as plt
        plt.plot(getattr(self.step_func, 'a_lst', []), '.-')


class All2All(All2AllIterative):
    def __init__(self, site, windTurbines, wake_deficitModel,
                 superpositionModel=LinearSum(),
                 blockage_deficitModel=None, deflectionModel=None, turbulenceModel=None,
                 rotorAvgModel=None):
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel, superpositionModel=superpositionModel,
                                  blockage_deficitModel=blockage_deficitModel, deflectionModel=deflectionModel,
                                  turbulenceModel=turbulenceModel, solver=self.solver,
                                  rotorAvgModel=rotorAvgModel)

    def _calc_wt_interaction(self, WS_eff_ilk, **kwargs):
        return All2AllIterative._calc_wt_interaction(self, WS_eff_ilk=0, **kwargs)

    def solver(self, get_new_WS_eff, WS_eff_ilk, TI_eff_ilk, max_iter):
        return get_new_WS_eff(WS_eff_ilk, TI_eff_ilk)
