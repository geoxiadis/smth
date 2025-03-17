from numpy import newaxis as na
import xarray as xr
from py_wake import np
from py_wake.deficit_models.deficit_model import XRLUTDeficitModel
from py_wake.superposition_models import LinearSum
from py_wake.tests.test_files import tfp
from py_wake.utils.fuga_utils import FugaXRLUT, interp_lut_coordinate
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.utils import fuga_utils, gradients
from py_wake.utils.gradients import cabs

from py_wake.utils.model_utils import DeprecatedModel
import glob
from xarray.core.merge import merge_attrs
import warnings
import os


class Fuga(PropagateDownwind, DeprecatedModel):
    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=None, deflectionModel=None, turbulenceModel=None,
                 bounds='limit', smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=FugaDeficit(LUT_path, bounds=bounds,
                                                                 smooth2zero_x=smooth2zero_x,
                                                                 smooth2zero_y=smooth2zero_y,
                                                                 remove_wriggles=remove_wriggles),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)
        DeprecatedModel.__init__(self, 'py_wake.literature.fuga.Ott_2014')


class FugaBlockage(All2AllIterative, DeprecatedModel):
    def __init__(self, LUT_path, site, windTurbines, rotorAvgModel=None,
                 deflectionModel=None, turbulenceModel=None, convergence_tolerance=1e-6, bounds='limit', remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        fuga_deficit = FugaDeficit(LUT_path, bounds=bounds, remove_wriggles=remove_wriggles)
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel=fuga_deficit,
                                  rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                  deflectionModel=deflectionModel, blockage_deficitModel=fuga_deficit,
                                  turbulenceModel=turbulenceModel, convergence_tolerance=convergence_tolerance)
        DeprecatedModel.__init__(self, 'py_wake.literature.fuga.Ott_2014_Blockage')


class FugaMultiLUTDeficit(XRLUTDeficitModel):

    def __init__(self, LUT_path=tfp + 'fuga/2MW/multilut/LUTs_Zeta0=0.00e+00_16_32_*_zi400_z0=0.00001000_z9.8-207.9_UL_nx128_ny128_dx20.0_dy5.0.nc',
                 z_lst=None, TI_ref_height=None, bounds='limit',
                 smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False,
                 rotorAvgModel=None, groundModel=None,
                 use_effective_ti=False):

        fuga_kwargs = dict(variables=['UL', 'UT'], smooth2zero_x=smooth2zero_x, smooth2zero_y=smooth2zero_y,
                           remove_wriggles=remove_wriggles, z_lst=z_lst)
        if isinstance(LUT_path, str) and os.path.isdir(LUT_path):
            f = fuga_utils.dat2netcdf(LUT_path).filename
            da_lst = [FugaXRLUT(f, **fuga_kwargs).dataarray]
        else:
            if isinstance(LUT_path, str):
                lut_path_str = LUT_path
                LUT_path = list(glob.glob(lut_path_str))
                assert len(LUT_path), f"No files found matching {lut_path_str}"

            da_lst = [FugaXRLUT(f, **fuga_kwargs).dataarray for f in LUT_path]

        dims = self.preprocess_luts(da_lst)

        da_lst = [da.assign_coords({'d_h': da.diameter * 1000 + da.hubheight,
                                    **{k: getattr(da, k) for k in dims[1:]}}).expand_dims(dims) for da in da_lst]
        self.TI_ref_height = TI_ref_height

        if z_lst is None:
            z_lst = np.sort(np.unique([z for da in da_lst for z in da.z.values]))
        if not all([(da.x.values.tolist() == da_lst[0].x.values.tolist()) for da in da_lst]):  # pragma: no cover
            warnings.warn("LUTs contains different x coordinates. "
                          "Regenerated LUTs with fixed Nx and dx to reduce memory usage and improve performance")
        x_lst = np.sort(np.unique([da.x for da in da_lst]))
        x_lst = x_lst[(x_lst >= np.max([da.x[0] for da in da_lst])) & (x_lst <= np.min([da.x[-1] for da in da_lst]))]
        if not all([(da.y.values.tolist() == da_lst[0].y.values.tolist()) for da in da_lst]):  # pragma: no cover
            warnings.warn("LUTs contains different y coordinates. "
                          "Regenerated LUTs with fixed Ny and dy to reduce memory usage and improve performance")
        y_lst = np.sort(np.unique([da.y for da in da_lst]))
        y_lst = y_lst[(y_lst >= np.max([da.y[0] for da in da_lst])) & (y_lst <= np.min([da.y[-1] for da in da_lst]))]

        da_lst = [interp_lut_coordinate(da, x=x_lst, y=y_lst, z=z_lst) for da in da_lst]

        # combine_by_coords does not always merge attributes correctly
        attrs = merge_attrs([da.attrs for da in da_lst], combine_attrs='drop_conflicts')
        da = xr.combine_by_coords(da_lst, combine_attrs='drop').squeeze()
        da.attrs = attrs
        self.x, self.y = da.x.values, da.y.values
        self._args4model = {k + "_ilk" for k in ['zeta0', 'zi'] if k in da.dims}

        method = [['linear', 'nearest'][d in ['d_h', 'variable']] for d in da.dims]
        XRLUTDeficitModel.__init__(self, da, get_input=self.get_input, method=method, bounds=bounds,
                                   rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                   use_effective_ws=False, use_effective_ti=use_effective_ti)

    def preprocess_luts(self, da_lst):
        dims = ['d_h', 'zeta0', 'zi', 'z0']
        return dims

    def wake_radius(self, dw_ijlk, D_src_il, h_ilk, TI_ilk, hcw_ijlk=None, z_ijlk=None, **kwargs):
        z_ijlk = h_ilk[:, na]
        lim = self.da.y.max().item()

        def get_mdu(hcw_ijlk):
            hcw_ijlk = np.nan_to_num(np.clip(hcw_ijlk, -lim, lim), nan=lim)
            return self._calc_mdu(D_src_il=D_src_il, dw_ijlk=dw_ijlk,
                                  hcw_ijlk=hcw_ijlk, h_ilk=h_ilk, z_ijlk=z_ijlk, TI_ilk=TI_ilk, **kwargs)
        mdu_target = get_mdu(dw_ijlk * 0) * np.exp(-2)  # corresponding to 2 sigma

        def get_err(hcw):
            return get_mdu(hcw) - mdu_target

        def get_wake_radius(wake_radius_ijlk):
            # Newton Raphson
            for _ in range(4):
                err = get_err(wake_radius_ijlk)
                derr = get_err(wake_radius_ijlk + 1) - err
                wake_radius_ijlk = wake_radius_ijlk - err / derr
            return np.abs(wake_radius_ijlk)
        wake_radius_ijlk = get_wake_radius(D_src_il[:, na, :, na])  # diameter as initial guess

        if np.any(kwargs.get('yaw_ilk', [0])) and 'UT' in self.da.variable_names:
            # mean of positive and negative side
            wake_radius_ijlk = (wake_radius_ijlk + get_wake_radius(-D_src_il[:, na, :, na]))

        return wake_radius_ijlk

    def calc_deficit(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk, z_ijlk, ct_ilk, D_src_il, **kwargs):
        # bypass XRLUTDeficitModel.calc_deficit
        if not self.deficit_initalized:
            self._calc_layout_terms(dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, z_ijlk=z_ijlk, D_src_il=D_src_il, **kwargs)
        return self.mdu_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]

    def _calc_mdu(self, **kwargs):
        # 0 = UL
        variables0 = np.reshape(0, np.ones_like(kwargs['dw_ijlk'].shape).astype(int))
        mdu_ijlk = XRLUTDeficitModel.calc_deficit(self, **kwargs, variables=variables0)
        if 'yaw_ilk' in kwargs:
            theta_yaw_ijlk = gradients.deg2rad(kwargs['yaw_ilk'])[:, na]
            if list(self.da.variable_names) == ['UL', 'UT']:
                # 1 = UT
                mdUT_ijlk = XRLUTDeficitModel.calc_deficit(self, **kwargs, variables=variables0 + 1)
                mdUT_ijlk = np.negative(mdUT_ijlk, out=mdUT_ijlk, where=kwargs['hcw_ijlk'] >= 0)
                mdu_ijlk = mdu_ijlk * np.cos(theta_yaw_ijlk) + np.sin(theta_yaw_ijlk) * mdUT_ijlk
            else:
                mdu_ijlk *= np.cos(theta_yaw_ijlk)
        return mdu_ijlk

    def _calc_layout_terms(self, **kwargs):
        self.mdu_ijlk = self._calc_mdu(**kwargs)
        return self.mdu_ijlk

    def get_input(self, D_src_il, TI_ilk, h_ilk, dw_ijlk, hcw_ijlk, z_ijlk, **kwargs):
        user = {'zeta0': lambda: kwargs['zeta0_ilk'][:, na],
                'zi': lambda: kwargs['zi_ilk'][:, na],
                'variables': lambda: kwargs['variables']}
        interp_kwargs = {'d_h': (D_src_il[:, :, na] * 1000 + h_ilk)[:, na],
                         'z0': fuga_utils.z0(TI_ilk, self.TI_ref_height or h_ilk, zeta0=kwargs.get('zeta0_ilk', 0))[:, na],
                         'z': z_ijlk,
                         'x': dw_ijlk,
                         'y': cabs(hcw_ijlk)}
        interp_kwargs.update({k: v() for k, v in user.items() if k in self.da.dims})
        return [interp_kwargs[k] for k in self.da.dims]

    def get_output(self, output_ijlk, **kwargs):
        return output_ijlk


class FugaYawDeficit(FugaMultiLUTDeficit):
    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc',
                 z_lst=None, TI_ref_height=None, bounds='limit',
                 smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False,
                 rotorAvgModel=None, groundModel=None,
                 use_effective_ti=False):
        FugaMultiLUTDeficit.__init__(self, LUT_path=LUT_path, z_lst=z_lst, TI_ref_height=TI_ref_height, bounds=bounds,
                                     smooth2zero_x=smooth2zero_x, smooth2zero_y=smooth2zero_y, remove_wriggles=remove_wriggles,
                                     rotorAvgModel=rotorAvgModel, groundModel=groundModel, use_effective_ti=use_effective_ti)


class FugaDeficit(FugaMultiLUTDeficit):
    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc',
                 z_lst=None, TI_ref_height=None, bounds='limit',
                 smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False,
                 rotorAvgModel=None, groundModel=None,
                 use_effective_ti=False):
        FugaMultiLUTDeficit.__init__(self, LUT_path=LUT_path, z_lst=z_lst, TI_ref_height=TI_ref_height, bounds=bounds,
                                     smooth2zero_x=smooth2zero_x, smooth2zero_y=smooth2zero_y, remove_wriggles=remove_wriggles,
                                     rotorAvgModel=rotorAvgModel, groundModel=groundModel, use_effective_ti=use_effective_ti)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'

        for wf_model in [Fuga(path, site, windTurbines),
                         FugaBlockage(path, site, windTurbines)]:
            plt.figure()
            print(wf_model)

            # run wind farm simulation
            sim_res = wf_model(x, y)

            # calculate AEP
            aep = sim_res.aep().sum()

            # plot wake map
            flow_map = sim_res.flow_map(wd=30, ws=9.8)
            flow_map.plot_wake_map()
            flow_map.plot_windturbines()
            plt.title('AEP: %.2f GWh' % aep)
        plt.show()


main()
