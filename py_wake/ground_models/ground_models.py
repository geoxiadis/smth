from py_wake import np
from numpy import newaxis as na
from py_wake.utils.model_utils import ModelMethodWrapper, Model
from py_wake.superposition_models import LinearSum


class GroundModel(Model, ModelMethodWrapper):
    """"""


class NoGround(GroundModel):
    """ Using this model corresponds to groundModel=None, but it can be used to override the groundModel
     specified for the windFarmModel in e.g. the turbulence model"""

    def __call__(self, func, **kwargs):
        return func(**kwargs)


class Mirror(GroundModel):
    """Consider the ground as a mirror (modeled by adding underground wind turbines)"""

    def __init__(self, superpositionModel=None):
        """
        Parameters
        ----------
        superpositionModel : SuperPositionModel, optional
            Superposition model used to sum wakes from the above and below-ground turbine.
            If None, default, the superposition model used for the deficit model is applied.
            I.e. WindFarmModel.superpositionModel when applied to wake deficits and
            WindFarmModel.blockageModel.superpositionModel when applied to blockage deficits.
        """
        self.superpositionModel = superpositionModel

    def _update_kwargs(self, **kwargs):
        I = kwargs['IJLK'][0]

        def add_mirror_wt(k, v):
            if (np.shape(v)[0] > 1 or ('_ijlk' in k and I == 1)) and '_jlk' not in k:
                return np.concatenate([v, v], 0)
            else:
                return v
        new_kwargs = {k: add_mirror_wt(k, v) for k, v in kwargs.items()}
        new_kwargs['dh_ijlk'] = np.concatenate([kwargs['dh_ijlk'],
                                                kwargs['dh_ijlk'] + (2 * kwargs['h_ilk'][:, na, :])],
                                               0)
        if 'cw_ijlk' in kwargs:
            new_kwargs['cw_ijlk'] = np.sqrt(new_kwargs['dh_ijlk']**2 + new_kwargs['hcw_ijlk']**2)
        return new_kwargs

    def __call__(self, func, h_ilk, dh_ijlk, IJLK, **kwargs):
        new_kwargs = self._update_kwargs(h_ilk=h_ilk, dh_ijlk=dh_ijlk, IJLK=IJLK, **kwargs)
        above_ground = ((new_kwargs['h_ilk'][:, na, :] + new_kwargs['dh_ijlk']) > 0)
        values_pijlk = func(**new_kwargs)
        deficit_mijlk = np.reshape(values_pijlk * above_ground, (2,) + IJLK)
        superpositionModel = self.superpositionModel or getattr(func, '__self__', func).superpositionModel
        return superpositionModel(deficit_mijlk)

    def _calc_layout_terms(self, func, **kwargs):
        new_kwargs = self._update_kwargs(**kwargs)
        func(**new_kwargs)


class MultiMirror(Mirror):
    """
    Considers the ground and a mirror aloft as a mirror by default. Two
    mirrors lead to infinitely many mirror planes, so the user can set how
    many mirrors should be enforced (n_mirrors = 2 + n_reps). Be aware that
    the number of turbines grows quickly n_turbines = 2^(n_mirrors).
    """

    def __init__(self, mirror_height=1000., n_reps=0, superpositionModel=None):
        """
        Parameters
        ----------
        mirror_heigth : int or float, optional
        n_reps : int optional
        superpositionModel : SuperPositionModel, optional
            Superposition model used to sum wakes from the above and below-ground turbine.
            If None, default, the superposition model used for the deficit model is applied.
            I.e. WindFarmModel.superpositionModel when applied to wake deficits and
            WindFarmModel.blockageModel.superpositionModel when applied to blockage deficits.
        """
        Mirror.__init__(self, superpositionModel)
        # no. of request repeats
        self.n_reps = n_reps
        # initialize mirror heights
        h_mirrors = [0., mirror_height]
        # for each repeat the last mirror is reflected in the ones preceding it
        for i in range(1, n_reps + 1):
            if i % 2 == 0:
                fac = i // 2 + 1
            else:
                fac = -(i // 2 + 1)
            h_mirrors.append(fac * mirror_height)
        self.h_mirrors = h_mirrors
        self.n_mirrors = len(h_mirrors)

    def _update_kwargs(self, **kwargs):
        I = kwargs['IJLK'][0]

        def add_mirror_wt(k, v, n):
            if (np.shape(v)[0] > 1 or ('_ijlk' in k and I == 1)) and '_jlk' not in k:
                for _ in range(n):
                    v = np.concatenate([v, v], axis=0)
                return v
            else:
                return v
        new_kwargs = {k: add_mirror_wt(k, v, self.n_mirrors) for k, v in kwargs.items()}

        # init
        h0 = kwargs['h_ilk']
        dh = kwargs['h_ilk'] * 0.
        # hub heights including mirrors (new_kwargs['h_ilk'] are just copies of kwargs['h_ilk'])
        h_ilk = kwargs['h_ilk']
        for n in range(self.n_mirrors):
            # hub heights of imaginary turbines
            img_height = - (h_ilk - self.h_mirrors[n]) + self.h_mirrors[n]
            h_ilk = np.concatenate([h_ilk,
                                    img_height],
                                   0)
            # vertical offset with respect to non-imaginary hub height
            dh = np.concatenate([dh,
                                 h0 - img_height],
                                0)
            # keep shape of reference hub height and deltas in line
            h0 = np.concatenate([h0, h0], axis=0)
        # add dh from mirror turbines
        new_kwargs['dh_ijlk'] += dh[:, na, :]

        if 'cw_ijlk' in kwargs:
            new_kwargs['cw_ijlk'] = np.sqrt(new_kwargs['dh_ijlk']**2 + new_kwargs['hcw_ijlk']**2)

        return new_kwargs

    def __call__(self, func, h_ilk, dh_ijlk, IJLK, **kwargs):
        new_kwargs = self._update_kwargs(h_ilk=h_ilk, dh_ijlk=dh_ijlk, IJLK=IJLK, **kwargs)
        # above_ground = ((new_kwargs['h_ilk'][:, na, :] + new_kwargs['dh_ijlk']) > 0)
        above_ground = np.ones_like(new_kwargs['dh_ijlk'])
        values_pijlk = func(**new_kwargs)
        deficit_mijlk = np.reshape(values_pijlk * above_ground, (2**self.n_mirrors,) + IJLK)
        superpositionModel = self.superpositionModel or func.__self__.superpositionModel
        return superpositionModel(deficit_mijlk)

    def _calc_layout_terms(self, func, **kwargs):
        new_kwargs = self._update_kwargs(**kwargs)
        func(**new_kwargs)
