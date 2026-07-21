from abc import abstractmethod, ABC
from py_wake import np
from py_wake.utils.most import psi


class Shear(ABC):
    # Upon implementing a new shear type, please specify if it is relative or absolute
    # by setting _is_relative.
    @abstractmethod
    def __call__(self, localWind, WS_ilk, h, return_gain=False):
        """Get wind speed at height

        Parameters
        ----------
        localWind : LocalWind
            Local wind and coordinates
        WS_ilk : array_like
            Wind speed
        h : array_like
            Height
        return_gain : bool
            Only return the gain, instead of multiplying by the wind speed.
            This only applies to `PowerShear`, `MOSTShear` and `LogShear`.
        Returns
        -------
        WS_ilk : array_like
            Wind speed or gain at height h.
        """


class PowerShear(Shear):
    # Read-only property indicating that the power shear is relative to the "raw" wind speed.
    _is_relative = True

    def __init__(self, h_ref=100, alpha=.1, interp_method='nearest'):
        self.h_ref = h_ref
        from py_wake.site._site import get_sector_xr
        self.alpha = get_sector_xr(alpha, "Power shear coefficient")
        self.interp_method = interp_method

    def __call__(self, localWind, WS_ilk, h, return_gain=False):
        alpha = self.alpha.interp_ilk(localWind.coords, interp_method=self.interp_method)
        gain = (h / self.h_ref).reshape((h.shape + (1, 1))[:3]) ** alpha
        if return_gain:
            return gain
        else:
            return gain * WS_ilk


class MOSTShear(Shear):
    # Read-only property indicating that the MOST shear is relative to the "raw" wind speed.
    _is_relative = True

    def __init__(self, h_ref=100, z0=.03, h_zeta=0.0, Cm1=5.0, Cm2=-19.3, interp_method='nearest'):
        self.h_ref = h_ref
        from py_wake.site._site import get_sector_xr
        self.z0 = get_sector_xr(z0, "Roughness length")
        self.h_zeta = h_zeta
        self.Cm1 = Cm1
        self.Cm2 = Cm2
        self.interp_method = interp_method

    def __call__(self, localWind, WS_ilk, h, return_gain=False):
        assert np.all(h > 0), "LogShear invalid at z=0"
        z0 = self.z0.interp_ilk(localWind.coords, interp_method=self.interp_method)
        L_inv = self.h_zeta / self.h_ref  # 1 / Obukhov length
        h = np.reshape(h, (h.shape + (1, 1))[:3])
        gain = (np.log(h / z0) - psi(h * L_inv, Cm1=self.Cm1, Cm2=self.Cm2)) / \
            (np.log(self.h_ref / z0) - psi(self.h_zeta, Cm1=self.Cm1, Cm2=self.Cm2))
        if return_gain:
            return gain
        else:
            return gain * WS_ilk


class LogShear(MOSTShear):
    def __init__(self, h_ref=100, z0=.03, interp_method='nearest'):
        MOSTShear.__init__(self, h_ref=h_ref, z0=z0, interp_method=interp_method)


class LLJRiemer(Shear):
    """
    Low Level Jet (LLJ) according to Leonard Riemer M.Sc. thesis.

    References:
    - Leonard Riemer; Exploring effects of the atmospheric boundary layer capping-inversion on wind turbine loads:
      gravity waves, turbulence, and shear. July 2025. Supervised by Mark Kelly and Michael Kenneth McWilliam.
      https://findit.dtu.dk/en/catalog/688ea91ba45c240102df8b75
    - Pierangelo Libianchi, Elena Shabalina, Mark Kelly, Jonas Brunskog, Finn Agerkvist; Sensitivity of the predicted
      acoustic pressure field to the wind and temperature profiles in a conventionally neutral boundary layer.
      J. Acoust. Soc. Am. 1 August 2023; 154 (2): 763–771. https://doi.org/10.1121/10.0020580
    """
    # Read-only property indicating that the LLJ is an offset to the "raw" wind speed.
    _is_relative = False

    def __init__(self, strength=2.0, width=100.0, h_ref=450.0, wsp_ref=None, interp_method="nearest"):
        """
        Parameters
        ----------
        strength : float, optional
            Strength of the low level jet. If `wsp_ref` is `None` (default), then the strength is interpreted as absolute,
            and measured in m/s. Otherwise, it is relative to `wsp_ref` and non-dimensional.
            The default is 2 m/s.
        width : float, optional
            Width of the jet (similar to the variance of a Gaussian distribution). The default is 100 m.
        h_ref : float, optional
            Height at which the maximum speed of the LLJ occours (similar to the mean of a Gaussian distribution).
            This parameter should not be confused with the hub height.
            The default is 450 m.
        wsp_ref : float, optional
            Wind speed at the reference height, used to scale the LLJ [m/s]. The default is `None`.
        interp_method : str, optional
            Interpolation method. The default is 'nearest'.

        Returns
        -------
        None
        """
        from py_wake.site._site import get_sector_xr
        self.strength = get_sector_xr(strength, "LLJ strength")
        self.width = get_sector_xr(width, "LLJ width")
        self.h_ref = get_sector_xr(h_ref, "LLJ reference height")
        self.wsp_ref = get_sector_xr(wsp_ref, "Wind speed at reference height") if wsp_ref else None
        self.interp_method = interp_method

    def __call__(self, localWind, WS_ilk, h, return_gain=False):
        assert not return_gain
        strength = self.strength.interp_ilk(localWind.coords, interp_method=self.interp_method)
        width = self.width.interp_ilk(localWind.coords, interp_method=self.interp_method)
        h_ref = self.h_ref.interp_ilk(localWind.coords, interp_method=self.interp_method)
        # LLJ expression, implemented according to Eq. (34) in Riemer's thesis.
        # At this stage, the strength is absolute.
        llj = np.reshape(strength * np.exp(- ((h - h_ref) / width)**2), (h.shape + (1, 1))[:3])
        if self.wsp_ref:  # Strength is relative.
            wsp_ref = self.wsp_ref.interp_ilk(localWind.coords, interp_method=self.interp_method)
            llj = llj * wsp_ref  # Cannot use *= with Automatic Differentiation.
        shape = np.broadcast_shapes(llj.shape, WS_ilk.shape)
        return np.broadcast_to(llj, shape)


# ======================================================================================================================
# Potentially the code below can be used to implement power/log shear interpolation between grid layers
# ======================================================================================================================
# class InterpolationShear(ABC):
#     @abstractmethod
#     def setup(self, ds):
#         """"""
#
#     @abstractmethod
#     def __call__(self):
#         """"""
#
#
# class LinearInterpolationShear():
#     def setup(self, ds):
#         pass
#
#     def __call__(self, WS_ilk, WD_ilk, h_i):
#         return WS_ilk
#
#
# class PowerInterpolationShear():
#     """Apply wind shear coefficient based on speed-up factor at different
#     # height and a reference far field wind shear coefficient (alpha_far)"""
#
#     def __init__(self, alpha_far=.143):
#         self.alpha_far = alpha_far
#
#     def setup(self, ds):
#         ds['wind_shear'] = copy.deepcopy(ds['spd'])
#
#         heights = ds['wind_shear'].coords['z'].data
#
#         # if there is only one layer, assign default value
#         if len(heights) == 1:
#
#             ds['wind_shear'].data = (np.zeros_like(ds['wind_shear'].data) + self.alpha_far)
#
#             print('Note there is only one layer of wind resource data, ' +
#                   'wind shear are assumed as uniform, i.e., {0}'.format(self.alpha_far))
#         else:
#             ds['wind_shear'].data[:, :, 0, :] = (self.alpha_far +
#                                                  np.log(ds['spd'].data[:, :, 0, :] / ds['spd'].data[:, :, 1, :]) /
#                                                  np.log(heights[0] / heights[1]))
#
#             for h in range(1, len(heights)):
#                 ds['wind_shear'].data[:, :, h, :] = (
#                     self.alpha_far +
#                     np.log(ds['spd'].data[:, :, h, :] / ds['spd'].data[:, :, h - 1, :]) /
#                     np.log(heights[h] / heights[h - 1]))
#
#     def __call__(self, WS_ilk, WD_ilk, h_i):
#         ????? WS_ilk = (WS_ilk * (H_hub / self.height_ref) ** wind_shear_il[i_wt, l_wd])
