from py_wake import np
from numpy import newaxis as na
from numpy import atleast_1d
from py_wake.utils import gradients
from autograd.numpy.numpy_boxes import ArrayBox
from scipy.interpolate._rgi import RegularGridInterpolator as RGI
from py_wake.utils.gradients import set_gradient_function
from autograd.core import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f


class AutogradRGI(RGI):
    def __init__(self, *args, **kwargs):
        RGI.__init__(self, *args, **kwargs)

    def evaluate_autograd(self, *xp_lst, method=None):
        if not hasattr(self, 'pf'):
            self.pf = primitive(self.evaluate)

            defvjp(self.pf, *[lambda ans, *x, df=df, **kwargs:
                   unbroadcast_f(np.transpose(x).squeeze(), lambda g:
                                 (df(np.transpose(x), **kwargs) * g).T) for df in self.get_gradient_functions()])
        return self.pf(*xp_lst, method=method)

    def get_gradient_functions(self):
        if not hasattr('self', 'RGI_gradient_lst'):
            self.RGI_gradient_lst = []
            for i in range(len(self.grid)):
                grid = [g for g in self.grid]
                dims = [j for j in np.arange(len(self.values.shape)) if i != j]
                values = np.repeat(np.diff(self.values, 1, i) / np.expand_dims(np.diff(grid[i]), dims), 2, i)
                grid[i] = np.repeat(self.grid[i], 2)[1:-1]
                for e in np.arange(10):
                    idx = np.where(np.diff(grid[i]) == 0)[0]
                    if len(idx) == 0:
                        break
                    grid[i][idx] -= np.finfo(np.float).eps * 10**e

                self.RGI_gradient_lst.append(RGI(grid, values))
        return self.RGI_gradient_lst

    def evaluate(self, *xp_lst, method=None):

        xp = np.transpose(xp_lst)
        if all([np.isrealobj(v) for v in [xp]]):
            return RGI.__call__(self, xp, method=method)
        else:
            # return self.PPInterpolator(xp)
            yp = RGI.__call__(self, np.real(xp), method=method)
            res = yp + np.sum([(x.imag.T * df(xp).T).T for x, df in zip(xp_lst, self.get_gradient_functions())], 0) * 1j
            from py_wake.tests import npt

            npt.assert_array_almost_equal(res, self.PPInterpolator(xp))
            return res

    def __call__(self, xp, method=None):
        assert len(np.shape(xp)) == 2, f"xp must have shape (n_point, n_interp_dims), but has {np.shape(xp)}"
        if isinstance(xp, ArrayBox):
            return self.evaluate_autograd(*list(np.transpose(xp)), method=method)
        else:
            return self.evaluate(*list(np.transpose(xp)), method=method)


class GridInterpolator(AutogradRGI):

    def __init__(self, x, V, method='linear', bounds='check', deg=False):
        self.bounds = bounds
        self.deg = deg
        AutogradRGI.__init__(self, x, V, method=method)

    @property
    def PPInterpolator(self):
        if not hasattr(self, '_PPInterpolator'):
            self._PPInterpolator = PPGridInterpolator(list(self.grid), self.values, method=self.method)
        return self._PPInterpolator

    def __call__(self, xp, method=None, deg=False):

        if self.bounds == 'limit':
            eps = np.finfo(np.float).eps
            xp = np.maximum(np.minimum(xp, [g[-1] - eps for g in self.grid]), [g[0] + eps for g in self.grid])
        # ref = PPGridInterpolator(list(self.grid), self.values)(xp, bounds=bounds, method=method, deg=deg)
        # print(np.abs(RGI.__call__(self, xp) - ref).max())
        try:
            res = AutogradRGI.__call__(self, xp, method=method)
            if deg or self.deg:
                res = gradients.mod(res, 360)
            return res
        except ValueError as e:
            if str(e).startswith('One of the requested xi is out'):
                self.PPInterpolator(xp)
            raise


class PPGridInterpolator(object):
    # Faster than scipy.interpolate.interpolate.RegularGridInterpolator
    def __init__(self, x, V, method='linear', bounds='check'):
        """
        Parameters
        ----------
        x : list of array_like
            Interpolation coordinates
        V : array_like
            Interpolation data (more dimensions than coordinates is possible)
        method : {'linear' or 'nearest} or [{'linear' or 'nearest}, ...]
            default interpolation method
        bounds : {'check', 'limit', 'ignore'}
            Specifies how bounds is handled:\n
            - 'check': bounds check is performed. An error is raised if interpolation point outside area
            - 'limit': interpolation points are forced inside the area
            - 'ignore': Faster option with no check. Use this option if data is guaranteed to be inside the area
        """
        self.x = self.grid = x.copy()

        self.bounds = bounds
        self.method = method
        self.n = np.array([len(x) for x in x], dtype=int)
        self.x0 = np.array([x[0] for x in x])
        dx = np.array([x[min(1, len(x) - 1)] - x[0] for x in x])
        self.dx = np.where(dx == 0, 1, dx)
        self.irregular_axes = np.array([np.allclose(np.diff(x), dx) is False for dx, x in zip(self.dx, x)])
        self.irregular_axes_indexes = np.where(self.irregular_axes)[0]
        for i in self.irregular_axes_indexes:
            self.x[i] = np.r_[self.x[i], self.x[i][-1] + 1]
        self.V = self.values = np.asarray(V)
        if not np.all(np.asarray(self.V.shape[:len(self.n)]) == self.n):
            raise ValueError("Lengths of x does not match shape of V")
        ui = np.array([[0], [1]])
        for _ in range(len(x) - 1):
            ui = np.array([(np.r_[ui, 0], np.r_[ui, 1]) for ui in ui])
            ui = ui.reshape((ui.shape[0] * ui.shape[1], ui.shape[2]))
        ui[:, dx == 0] = 0
        self.ui = ui.astype(int)

    def __call__(self, xp, method=None, bounds=None, deg=False):
        """Interpolate points

        Parameters
        ----------
        xp : array_like
            Interpolation points, shape=(n_points, interpolation_dimensions)
        method : {'linear' or 'nearest} or [{'linear' or 'nearest}, ...] or None
            Overrides self.method if not None
        bounds : {'check', 'limit', 'ignore'} or None
            Overrides self.bounds if not None
        """
        if len(np.atleast_1d(xp)) == 0:
            return np.array([])
        method = atleast_1d(method or self.method)
        assert all([m in ['linear', 'nearest'] for m in method]), 'method must be "linear" or "nearest"'
        assert len(method) in [1, len(self.x)]
        linear = [method[min(len(method) - 1, i)] == 'linear' for i in range(len(self.x))]
        bounds = bounds or self.bounds
        assert bounds in ['check', 'limit', 'ignore'], 'bounds must be "check", "limit" or "ignore"'
        xp = np.atleast_2d(xp)
        xp_shape = xp.shape
        assert xp_shape[-1] == len(self.x), xp_shape
        xp = np.reshape(xp, (-1, xp_shape[-1]))
        if len(self.irregular_axes_indexes):
            xpi0 = np.array([np.clip(np.searchsorted(x, xp, side='right') - 1, 0, n - 2, dtype=int)
                             for x, xp, n in zip(self.x, xp.T, self.n)], dtype=int)
            xp0 = np.array([np.asarray(x)[xpi0] for x, xpi0 in zip(self.x, xpi0)])
            xp1 = np.array([np.asarray(x)[xpi0 + 1] for x, xpi0 in zip(self.x, xpi0)])
            xp_dx = xp1 - xp0
            xpi = xpi0.T + (xp - xp0.T) / xp_dx.T
        else:
            xpi = (xp - self.x0) / self.dx

        if bounds == 'check' and (np.any(xpi < 0) or np.any(xpi + 1 > self.n[na])):
            if -xpi.min() > (xpi + 1 - self.n[na]).max():
                point, dimension = np.unravel_index(xpi.argmin(), np.atleast_2d(xpi).shape)
            else:
                point, dimension = np.unravel_index(((xpi + 1 - self.n[na])).argmax(), np.atleast_2d(xpi).shape)
            raise ValueError("Point %d, dimension %d with value %f is outside range %f-%f" %
                             (point, dimension, np.atleast_2d(xp)[point, dimension], self.x[dimension][0], self.x[dimension][-1]))
        if bounds == 'limit':
            xpi = np.minimum(np.maximum(xpi, 0), self.n - 1)

        if 'nearest' in method:
            # round x.5 down to match old results
            xpi = np.where(np.asarray(linear), xpi, np.round(xpi - .1 * (gradients.mod(xpi, 2) == 1.5)))
        xpif, xpi0 = gradients.modf(xpi)

        int_box_axes = [(np.asarray([0], dtype=int), np.asarray([0, 1], dtype=int))[l] for l in linear]
        ui = np.moveaxis(np.asarray(np.meshgrid(*int_box_axes, indexing='ij'), dtype=int), 0, -1)

        ui = ui.reshape((-1, len(self.x)))
        indexes = (ui.T[:, :, na] + xpi0.T[:, na])

        indexes = np.minimum(indexes, (self.n - 1)[:, na, na], dtype=int)
        v = np.moveaxis(self.V[tuple(indexes)], [0, 1], [-2, -1])
        if deg:
            v = (v + 180) % 360 - 180  # -180..180 > 0-360

        for i, x in enumerate(zip((1 - xpif).T, xpif.T)):
            if linear[i]:
                if i == 0:
                    w = np.array(x)
                else:
                    w = w[..., na, :] * np.expand_dims(np.array(x), tuple(range(i)))
            else:
                if i == 0:
                    w = x[0]
                else:
                    w = w * x[0]  # np.expand_dims(x[0], tuple(range(i)))
        w = np.reshape(w, (-1, xpif.shape[0]))

        # w = np.prod(np.array([xpif1, xpif])[ui, :, range(len(self.x))], 1) # slower
        # w = np.product(np.take_along_axis(xpif10[:, :, na], self.ui[na, na], 0).squeeze(), 2)  # even slower

        res = np.moveaxis((w * v).sum(-2), -1, 0)
        if deg:
            res = gradients.mod(res, 360)
        return np.reshape(res, xp_shape[:-1] + self.V.shape[len(self.x):])


class EqDistRegGrid2DInterpolator():
    def __init__(self, x, y, Z):
        self.x = x
        self.y = y
        self.Z = Z
        self.dx, self.dy = [xy[1] - xy[0] for xy in [x, y]]
        assert all(np.diff(x) == self.dx), "x is not equidistant"
        assert all(np.diff(y) == self.dy), "y is not equidistant"
        self.x0 = x[0]
        self.y0 = y[0]
        xi_valid = np.where(np.any(~np.isnan(self.Z), 1))[0]
        yi_valid = np.where(np.any(~np.isnan(self.Z), 0))[0]
        self.xi_valid_min, self.xi_valid_max = xi_valid[0], xi_valid[-1]
        self.yi_valid_min, self.yi_valid_max = yi_valid[0], yi_valid[-1]

    def __call__(self, x, y, mode='valid'):
        xp, yp = x, y
        xif, xi0 = gradients.modf((xp - self.x0) / self.dx)
        yif, yi0 = gradients.modf((yp - self.y0) / self.dy)

        if mode == 'extrapolate':
            xif[xi0 < self.xi_valid_min] = 0
            xif[xi0 > self.xi_valid_max - 2] = 1
            yif[yi0 < self.yi_valid_min] = 0
            yif[yi0 > self.yi_valid_max - 2] = 1
            xi0 = np.minimum(np.maximum(xi0, self.xi_valid_min), self.xi_valid_max - 2)
            yi0 = np.minimum(np.maximum(yi0, self.yi_valid_min), self.yi_valid_max - 2)
        xi1 = xi0 + (xif > 0)
        yi1 = yi0 + (yif > 0)
        if isinstance(xp, ArrayBox) or isinstance(yp, ArrayBox):
            valid = slice(None)
        else:
            valid = (xif >= 0) & (yif >= 0) & (xi1 < len(self.x)) & (yi1 < len(self.y))
        xi0, xi1, xif, yi0, yi1, yif = [v[valid] for v in [xi0, xi1, xif, yi0, yi1, yif]]
        z00 = self.Z[xi0, yi0]
        z10 = self.Z[xi1, yi0]
        z01 = self.Z[xi0, yi1]
        z11 = self.Z[xi1, yi1]
        z0 = z00 + (z10 - z00) * xif
        z1 = z01 + (z11 - z01) * xif
        if isinstance(xp, ArrayBox) or isinstance(yp, ArrayBox):
            z = z0 + (z1 - z0) * yif
        else:
            dtype = float
            if np.iscomplexobj(xp):
                dtype = xp.dtype
            if np.iscomplexobj(yp):
                dtype = yp.dtype

            z = np.full(xp.shape, np.nan, dtype=dtype)
            z[valid] = z0 + (z1 - z0) * yif
        return z
