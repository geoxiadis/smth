import xarray as xr
from py_wake.deficit_models.deficit_model import XRLUTDeficitModel, ConvectionDeficitModel
from py_wake.utils.model_utils import XRLUTModel
from py_wake import np
from py_wake.superposition_models import LinearSum, WeightedSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.turbulence_models.rans_lut_turb import RANSLUTTurbulence
from py_wake.utils.rans_lut_utils import RANSLUTModel
from py_wake.tests import ptf
from pathlib import Path
from numpy import newaxis as na


class RANSLUT(All2AllIterative):
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
            superpositionModel = WeightedSum(weight_limit=1.0)
            if not isinstance(convlut, (list, tuple)):
                convlut = [convlut]
            convlut_lst = convlut
            convlut_lst = [load_lut(convlut) for convlut in convlut_lst]
            wake_deficit = RANSLUTConvDeficit(lut=[lut.deficits for lut in lut_lst], convlut=convlut_lst, rotorAvgModel=rotorAvgModel, superpositionModel=WeightedSum(weight_limit=1.0))
        else:
            # Use linear wake superposition
            superpositionModel = LinearSum()
            wake_deficit = RANSLUTDeficit(lut=[lut.deficits for lut in lut_lst], rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum())

        blockage_deficit = RANSLUTDeficit(lut=[lut.deficits for lut in lut_lst], rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum())
        turb = RANSLUTTurbulence(lut=[lut.added_ti for lut in lut_lst], rotorAvgModel=rotorAvgModel)

        All2AllIterative.__init__(self, site, windTurbines,
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


class RANSLUTConvDeficit(RANSLUTModel, XRLUTDeficitModel, ConvectionDeficitModel):
    """Expects LUT velocity deficit and convection variables xarray"""

    def __init__(self, lut, convlut, superpositionModel=None, rotorAvgModel=None, groundModel=None, use_effective_ws=True,
                 use_effective_ti=False):
        assert use_effective_ws, "RANSLUTDeficit only makes sense when scaling with effective wind speed"
        XRLUTDeficitModel.__init__(self, self.get_lut(lut, 'deficits'),
                                   bounds='limit', superpositionModel=superpositionModel, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                   use_effective_ws=True, use_effective_ti=use_effective_ti)
        self.sigma_lut = XRLUTModel(self.get_lut([lut.sigma for lut in convlut], 'sigma'),
                                    self.get_conv_input, bounds='limit')
        self.center_def_lut = XRLUTModel(self.get_lut([lut.centerline_deficit for lut in convlut], 'centerline_deficit'),
                                         self.get_conv_input, bounds='limit')

    def get_conv_input(self, dw_ijlk, D_src_il, TI_eff_ilk, ct_ilk, type_il, **kwargs):
        kwargs = dict(x=dw_ijlk / D_src_il[:, na, :, na],
                      ti=TI_eff_ilk[:, na],
                      ct=ct_ilk[:, na],
                      type_i=type_il[:, na, :, na])

        return [kwargs[k] for k in self.sigma_lut.da_lst[0].dims]

    def sigma_ijlk(self, D_src_il, dw_ijlk, ct_ilk, WS_ref_ijlk, **kwargs):
        # dimensional wake expansion
        sigma_ijlk = self.sigma_lut.__call__(D_src_il=D_src_il, dw_ijlk=dw_ijlk, ct_ilk=ct_ilk, **kwargs)
        # Replace undefined sigma values (upstream of WT) with zeros
        sigma_ijlk[:, :, :][dw_ijlk[:, :, :, 0] < 0] = 0.0
        return sigma_ijlk * D_src_il[:, na, :, na]

    def _calc_deficit(self, D_src_il, dw_ijlk, ct_ilk, WS_ref_ijlk, **kwargs):
        # dimensional wake expansion
        sigma_sqr_ijlk = (self.sigma_ijlk(D_src_il=D_src_il, dw_ijlk=dw_ijlk, ct_ilk=ct_ilk, WS_ref_ijlk=WS_ref_ijlk, **kwargs))**2
        deficit_centre_ijlk = WS_ref_ijlk * np.minimum(1.0, self.center_def_lut.__call__(D_src_il=D_src_il, dw_ijlk=dw_ijlk, ct_ilk=ct_ilk, **kwargs))
        # Replace undefined center deficit values (upstream of WT) with zeros
        deficit_centre_ijlk[:, :, :][dw_ijlk[:, :, :, 0] < 0] = 0.0
        return WS_ref_ijlk, sigma_sqr_ijlk, deficit_centre_ijlk

    def calc_deficit_convection(self, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        if self.groundModel:  # pragma: no cover
            raise NotImplementedError(
                "calc_deficit_convection (WeightedSum) cannot be used in combination with GroundModels")
        WS_ref_ijlk, sigma_sqr_ijlk, deficit_centre_ijlk = self._calc_deficit(
            D_src_il, dw_ijlk, ct_ilk, **kwargs, **self.get_WS_ref_kwargs(kwargs))
        # Convection velocity
        uc_ijlk = WS_ref_ijlk - 0.5 * deficit_centre_ijlk
        sigma_sqr_ijlk = np.broadcast_to(sigma_sqr_ijlk, deficit_centre_ijlk.shape)
        return deficit_centre_ijlk, uc_ijlk, sigma_sqr_ijlk

    @property
    def args4deficit(self):
        return XRLUTModel.args4model.fget(self) | ConvectionDeficitModel.args4deficit.fget(self)


class RANSLUTDemoDeficit(RANSLUTDeficit):
    def __init__(self, superpositionModel=None, rotorAvgModel=None, groundModel=None, use_effective_ws=True, use_effective_ti=False):
        # Load default RANS LUT file based on a demo V80 LUT
        demo_lut = ptf('ranslut/V80_ranslut_demo.nc',
                       '846213eb655255f6e2201a47c2406f9e77f243f369398cb389bf7320b457dea8')

        RANSLUTDeficit.__init__(self, demo_lut, superpositionModel=superpositionModel, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)
