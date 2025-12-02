import xarray as xr
# from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.deficit_models.deficit_model import XRLUTDeficitModel, ConvectionDeficitModel, XRLUTConvDeficitModel
from py_wake import np
from py_wake.superposition_models import LinearSum, SquaredSum, WeightedSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative, PropagateDownwind
from py_wake.turbulence_models.rans_lut_turb import RANSLUTTurbulence
from py_wake.utils.rans_lut_utils import RANSLUTModel
from py_wake.tests import ptf
from pathlib import Path
from py_wake.deficit_models.utils import ct2a_madsen
from numpy import newaxis as na


class RANSLUT(All2AllIterative):
# class RANSLUT(PropagateDownwind):
    def __init__(self, lut, site, windTurbines, convlut=None, rotorAvgModel=None):
        """
        Parameters
        ----------
        lut : str, Path or xarray.Dataset
            if str or Path: path to xarray.Dataset with tables including velocity deficit and wake added ti
            if Dataset: Dataset containing deficit and wake added_ti
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        """
        if not isinstance(lut, (list, tuple)):
            lut = [lut]
        lut_lst = lut

        def load_lut(lut):
            if not isinstance(lut, xr.Dataset):
                lut = xr.load_dataset(lut)
            return lut
        lut_lst = [load_lut(lut) for lut in lut_lst]

        if convlut is not None:
            # Use convection wake deficit model for WeigthedSum superposition
            superpositionModel=WeightedSum()
            if not isinstance(convlut, (list, tuple)):
                convlut = [convlut]
                convlut_lst = convlut
                convlut_lst = [load_lut(convlut) for convlut in convlut_lst]
                wake_deficit = RANSLUTConvDeficit(lut=[lut.deficits for lut in lut_lst], convlut=convlut_lst ,rotorAvgModel=rotorAvgModel, superpositionModel=WeightedSum())
        else:
            # Use linear wake superposition
            superpositionModel=LinearSum()
            wake_deficit = RANSLUTDeficit(lut=[lut.deficits for lut in lut_lst], rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum())

        blockage_deficit = RANSLUTDeficit(lut=[lut.deficits for lut in lut_lst], rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum())
        turb = RANSLUTTurbulence(lut=[lut.added_ti for lut in lut_lst], rotorAvgModel=rotorAvgModel)

        All2AllIterative.__init__(self, site, windTurbines,
        # PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=wake_deficit,
                                   blockage_deficitModel=blockage_deficit,
                                   turbulenceModel=turb,
                                   superpositionModel=superpositionModel)


class RANSLUTDeficit(RANSLUTModel, XRLUTDeficitModel):
    """Expects LUT velocity deficit xarray"""

    def __init__(self, lut, superpositionModel=None, rotorAvgModel=None, groundModel=None, use_effective_ws=True,
                 use_effective_ti=False):
        assert use_effective_ws, "RANSLUTDeficit only makes sense when scaling with effective wind speed"
        XRLUTDeficitModel.__init__(self, self.get_lut(lut, 'deficits'),
                                   bounds='limit', superpositionModel=superpositionModel, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                   use_effective_ws=True, use_effective_ti=use_effective_ti)

    def wake_radius(self, dw_ijlk, **kwargs):
        # Required for PyWake but not needed for RANS LUT model
        wake_radius_ijlk = np.ones(dw_ijlk.shape)
        return wake_radius_ijlk


class RANSLUTConvDeficit(RANSLUTModel, XRLUTConvDeficitModel):
    """Expects LUT velocity deficit and convection variables xarray"""

    def __init__(self, lut, convlut, superpositionModel=None, rotorAvgModel=None, groundModel=None, use_effective_ws=True,
                 use_effective_ti=False, 
                 ct2a=ct2a_madsen, k=0.0324555, ceps=.2, ctlim=0.899):
        self._k = k
        self._ceps = ceps
        self._ctlim = ctlim
        self.ct2a = ct2a
        # TODO: add WT type
        self.convlut = convlut[0]
        assert use_effective_ws, "RANSLUTDeficit only makes sense when scaling with effective wind speed"
        XRLUTConvDeficitModel.__init__(self, self.get_lut(lut, 'deficits'),
                                       bounds='limit', superpositionModel=superpositionModel, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                       use_effective_ws=True, use_effective_ti=use_effective_ti)
        # TODO: use fast GridInterpolator
        # self.sigma_interp_lst = [GridInterpolator([da[k].values for k in da.dims], da.values, method='linear', bounds='limit')
        #                          for da in [convlut[0]['sigma']]]
        # print('self.convlut', self.convlut)
 
    def wake_radius(self, dw_ijlk, **kwargs):
        # Required for PyWake but not needed for RANS LUT model
        wake_radius_ijlk = np.ones(dw_ijlk.shape)
        return wake_radius_ijlk

    '''
    def k_ilk(self, **kwargs):
        return np.array([[[self._k]]])

    def epsilon_ilk(self, ct_ilk, **kwargs):
        # only valid for CT < 0.9.
        sqrt1ct_ilk = np.sqrt(1 - np.minimum(self._ctlim, ct_ilk))
        beta_ilk = 1 / 2 * (1 + sqrt1ct_ilk) / sqrt1ct_ilk

        return self._ceps * np.sqrt(beta_ilk)
    '''

    def sigma_ijlk(self, D_src_il, dw_ijlk, ct_ilk, WS_ref_ijlk, **kwargs):
        # dimensional wake expansion
        # TODO: use fast GridInterpolator
        I, _, L, K = WS_ref_ijlk.shape
        J = dw_ijlk.shape[1]
        sigma_ijlk = np.zeros((I, J, L, K))
        for i in range(I):
            for l in range(L):
                for k in range(K):
                    sigma_ijlk[i, :, l, k] = self.convlut.sigma.interp(x=(dw_ijlk[i, :, l, 0] / D_src_il[i, l]), ct=ct_ilk[i, l, k], ti=kwargs['TI_eff_ilk'][i, l, k]).values
        # print('sigma', sigma)
        return sigma_ijlk * D_src_il[:, na, :, na]
        # Gaussian wake model:
        # return self.k_ilk(**kwargs)[:, na] * dw_ijlk + \
        #    self.epsilon_ilk(ct_ilk, **kwargs)[:, na] * D_src_il[:, na, :, na]

    def ct_func(self, ct_ilk, **_):
        return ct_ilk[:, na]

    def _calc_deficit(self, D_src_il, dw_ijlk, ct_ilk, WS_ref_ijlk, **kwargs):
        # dimensional wake expansion
        sigma_sqr_ijlk = (self.sigma_ijlk(D_src_il=D_src_il, dw_ijlk=dw_ijlk, ct_ilk=ct_ilk, WS_ref_ijlk=WS_ref_ijlk, **kwargs))**2
        # print('sigma_sqr_ijlk', sigma_sqr_ijlk)
        ctx_ijlk = self.ct_func(ct_ilk=ct_ilk, dw_ijlk=dw_ijlk, D_src_il=D_src_il)
        # Gaussian wake model:
        # deficit_centre_ijlk = WS_ref_ijlk * np.minimum(1,
        #                                              2. * self.ct2a(ctx_ijlk * D_src_il[:, na, :, na]**2 /
        #                                                             (8. * sigma_sqr_ijlk)))
        # TODO: use fast GridInterpolator
        I, _, L, K = WS_ref_ijlk.shape
        J = dw_ijlk.shape[1]
        deficit_centre_ijlk = np.zeros((I, J, L, K))
        for i in range(I):
            for l in range(L):
                for k in range(K):
                    deficit_centre_ijlk[i, :, l, k] = self.convlut.centerline_deficit.interp(x=(dw_ijlk[i, :, l, 0] / D_src_il[i, l]), ct=ct_ilk[i, l, k], ti=kwargs['TI_eff_ilk'][i, l, k]).values
        deficit_centre_ijlk = WS_ref_ijlk * np.minimum(1.0, deficit_centre_ijlk)
        # print('deficit_centre_ijlk', deficit_centre_ijlk)
        return WS_ref_ijlk, sigma_sqr_ijlk, deficit_centre_ijlk, ctx_ijlk
 
    def calc_deficit_convection(self, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        if self.groundModel:
            raise NotImplementedError(
                "calc_deficit_convection (WeightedSum) cannot be used in combination with GroundModels")
        # Here, we need to replace sigma_sqr_ijlk, deficit_centre_ijlk and uc_ijlk with values from the LUT model
        # This may/should be added to XRLUTConvDeficitModel directly instead of here
        WS_ref_ijlk, sigma_sqr_ijlk, deficit_centre_ijlk, ctx_ijlk = self._calc_deficit(
            D_src_il, dw_ijlk, ct_ilk, **kwargs, **self.get_WS_ref_kwargs(kwargs))

        # actualdeficit = self.calc_deficit(D_src_il=D_src_il, dw_ijlk=dw_ijlk, ct_ilk=ct_ilk, **kwargs)
        # print('actualdeficit2', actualdeficit)
        # Convection velocity
        # Gaussian wake model:
        # uc_ijlk = WS_ref_ijlk * (1. - self.ct2a(ctx_ijlk * D_src_il[:, na, :, na]**2 / (8. * sigma_sqr_ijlk)))
        uc_ijlk = WS_ref_ijlk - 0.5 * deficit_centre_ijlk
        # print('uv_ijlk', uc_ijlk)
        sigma_sqr_ijlk = np.broadcast_to(sigma_sqr_ijlk, deficit_centre_ijlk.shape)
        return deficit_centre_ijlk, uc_ijlk, sigma_sqr_ijlk


class RANSLUTDemoDeficit(RANSLUTDeficit):
    def __init__(self, superpositionModel=None, rotorAvgModel=None, groundModel=None, use_effective_ws=True, use_effective_ti=False):
        # Load default RANS LUT file based on a demo V80 LUT
        demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                       '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')

        RANSLUTDeficit.__init__(self, demo_lut, superpositionModel=superpositionModel, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)
