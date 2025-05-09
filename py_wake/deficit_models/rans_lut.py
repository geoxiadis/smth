import xarray as xr
import sys
import time
from py_wake.deficit_models.deficit_model import XRLUTDeficitModel
from numpy import newaxis as na
from py_wake import np
from scipy.interpolate import interp1d
from py_wake.superposition_models import LinearSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.turbulence_models.rans_lut_turb import RANSLUTTurbulence
from py_wake.utils.rans_lut_utils import fugalut2ranslut, cxzy_to_cxyzg
from py_wake.rotor_avg_models import EllipSysPolygonRotorAvg
from py_wake.wind_farm_models.wind_farm_model import SimulationResult


class RANSLUT(All2AllIterative):
    def __init__(self, path_LUT, site, windTurbines, zRef, Dref, rotorAvgModel=None,
                 power_as_RANS=True, rotorAvgModelPost=EllipSysPolygonRotorAvg(n_r=9, n_theta=32),
                 ADcontrolfiles=None, ADcontrolfiles_write=True, flowmap_maxpoints=10000,
                 ws_cutin=[4.0], ws_cutout=[25.0], Prated=[2e6], dws=1.0):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables including velocity deifict and wake added ti
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        zRef : float
            Reference height to rescale lut file for single wt type luts
        Dref : float
            Reference rotor diameter to rescale lut file for single wt type luts
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        power_as_RANS : bool
            If True, the turbine power is calculated as a post step following
            a RANS approach using CPstar = CPstar * (U_inf / U_AD) ** 3 = func(U_AD),
            where UAD a rotor averaged wind speed including the turbine induction
            In addition, an alternative thrust coefficient is calculated
            CTstar = CT * (U_inf / U_AD) ** 2 = func(U_AD).
        rotorAvgModelPost : object
            rotorAvgModel used to post process turbine output when power_as_RANS=True
        ADcontrolfiles : list
            If not None, a list of filename(s) is expected including
            columns of U_AD, CTstar, CPstar, which could come from RANS.
        """
        self.power_as_RANS = power_as_RANS
        self.rotorAvgModelPost = rotorAvgModelPost
        if not isinstance(ADcontrolfiles, list):
            self.ADcontrolfiles = [ADcontrolfiles]
        else:
            self.ADcontrolfiles = ADcontrolfiles
        self.flowmap_maxpoints = flowmap_maxpoints
        self.ws_cutin = ws_cutin
        self.ws_cutout = ws_cutout
        self.dws = dws
        self.Prated = Prated

        lut = xr.load_dataset(path_LUT)
        if 'type_i' not in lut.coords:
            lut = self.rescale_lut_coords(lut, zRef, Dref)
        lut = self.rename_lut(lut)
        deficit = RANSLUTDeficit(lut=lut.deficits, rotorAvgModel=rotorAvgModel)
        turb = RANSLUTTurbulence(lut=lut.added_ti, rotorAvgModel=rotorAvgModel)

        All2AllIterative.__init__(self, site, windTurbines,
                                  wake_deficitModel=deficit,
                                  blockage_deficitModel=deficit,
                                  turbulenceModel=turb,
                                  superpositionModel=LinearSum())
        if power_as_RANS:
            # CP and CT as function of disk-avereaged wind speed used for post processing
            self.fint_CPstar = []
            self.fint_CTstar = []
            if ADcontrolfiles is None:
                starttime = time.time()
                # Could add a TI dimension interpolation lists
                lw_j = self.site.local_wind(h=zRef)
                Ti = lw_j.TI_ilk[0, 0, 0]
                self.cal_lut(Ti, write=ADcontrolfiles_write)
                endtime = time.time()
                print('cal time', endtime - starttime, 'sec')
            else:
                self.read_ad_controlfiles()

    def rescale_lut_coords(self, lut, zRef, Dref):
        if Dref is not None and zRef is not None:
            # Rescale non-dimensional coord to [m]
            lut['x'] = lut['x'] * Dref
            lut['y'] = lut['y'] * Dref
            lut['z'] = lut['z'] * Dref + zRef
        else:
            if Dref is None:
                print('ERROR: lut coords cannot be rescaled because Dref is not specified')
            elif zRef is None:
                print('ERROR: lut coords cannot be rescaled because zRef is not specified')
            sys.exit()
        return lut

    def rename_lut(self, lut):
        lutvars = ['x', 'y', 'z', 'ct', 'ti', 'yaw', 'type_i']
        renamevars = ['dw_ijlk', 'hcw_ijlk', 'z_ijlk', 'ct_ilk', 'TI_eff_ilk', 'yaw_ilk', 'type_i_ilk']
        d = {}
        for i in range(len(lutvars)):
            if lutvars[i] in lut.coords:
                d[lutvars[i]] = renamevars[i]
        return lut.rename(d)

    def read_ad_controlfiles(self):
        # RANS based calibrated control file for CPstar and CTstar as function of the disk averaged velocity UAD provided by the user
        for ADcontrolfile in self.ADcontrolfiles:
            RANScal = np.genfromtxt(ADcontrolfile, skip_header=True)
            self.fint_CPstar.append(interp1d(RANScal[:, 0], RANScal[:, 2], fill_value=0.0, bounds_error=False))
            self.fint_CTstar.append(interp1d(RANScal[:, 0], RANScal[:, 1], fill_value=0.0, bounds_error=False))

    def cal_lut(self, ti, density=1.225, cmu=0.03, kappa=0.4, write=False):
        # Calculcate CPstar, CTstar = f(UAD) directly from single wake RANS-LUT
        data = self.wake_deficitModel.da.interp(dw_ijlk=0.0)
        weights = self.rotorAvgModelPost.nodes_weight[na, :]
        nr = self.rotorAvgModelPost.n_rnodes
        ntheta = self.rotorAvgModelPost.n_theta
        cxyz = np.zeros((nr - 1, ntheta, 3))
        cxyz[:, :, 0] = 0.5 * self.rotorAvgModelPost.nodes_x.reshape((nr - 1, ntheta))
        cxyz[:, :, 1] = 0.5 * self.rotorAvgModelPost.nodes_y.reshape((nr - 1, ntheta))

        uStar_UH = ti * np.sqrt(1.5 * np.sqrt(cmu))
        # z0 = zRef / (np.exp(kappa / uStar_UH) - 1)

        if 'type_i_ilk' in data.dims:
            types = data.type_i_ilk.values
        else:
            types = [0]

        for type in types:
            if 'type' in self.windTurbines.powerCtFunction.required_inputs:
                _kwargs = {'type': type}
            else:
                _kwargs = {}
            ws = np.arange(self.ws_cutin[type], self.ws_cutout[type] + self.dws, self.dws)
            Dref = self.windTurbines.diameter(type)
            zH = self.windTurbines.hub_height(type)

            cxyzg_i = cxzy_to_cxyzg(cxyz[na, :, :, :] * Dref, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, zH]]))
            y_j = cxyzg_i[:, :, :, 1].flatten()
            h_j = cxyzg_i[:, :, :, 2].flatten()
            if 'type_i_ilk' in data.dims:
                Udata = data.sel(type_i_ilk=type).interp(TI_eff_ilk=ti).interp(hcw_ijlk=('i', y_j), z_ijlk=('i', h_j))
            else:
                Udata = data.interp(TI_eff_ilk=ti).interp(hcw_ijlk=('i', y_j), z_ijlk=('i', h_j))

            cts = self.windTurbines.ct(ws=ws, **_kwargs)
            cps = (self.windTurbines.power(ws=ws, **_kwargs) / (0.125 * density * Dref ** 2 * np.pi * ws ** 3))

            U_CT_CP_AD = np.zeros((len(ws), 3))
            # PWE: z0 is added to h_j and zH
            # logshear = np.log(h_j / z0) / np.log(zRef / z0)
            z0 = zH / (np.exp(kappa / uStar_UH) - 1)
            logshear = np.log(h_j / z0) / np.log(zH / z0)
            U = logshear[na, :] - 1.0 + Udata.interp(ct_ilk=cts, kwargs={"fill_value": "extrapolate"}).values
            UAD = ws * np.sum(np.reshape(U, (len(ws), (ntheta * (nr - 1)))) * weights, axis=1)
            U_CT_CP_AD[:, 0] = UAD
            ratio = ws / UAD
            U_CT_CP_AD[:, 1] = cts * ratio ** 2
            U_CT_CP_AD[:, 2] = cps * ratio ** 3

            if write:
                outfile = 'lutcal_WT%g_Ti%g_%i.dat' % (Dref, ti, type)
                f = open(outfile, 'w')
                f.write('%s %s\n' % (str(ws.size + 2), ' 4'))
                f.write('%16.14f %16.14f %16.14f %16.14f\n' % (0.95 * U_CT_CP_AD[0, 0], U_CT_CP_AD[0, 1], U_CT_CP_AD[0, 2], 0.95 * ws[0]))
                for k in range(len(ws)):
                    f.write('%16.14f %16.14f %16.14f %16.14f\n' % (U_CT_CP_AD[k, 0], U_CT_CP_AD[k, 1], U_CT_CP_AD[k, 2], ws[k]))
                f.write('%16.14f %16.14f %16.14f %16.14f\n' % (ws[-1], 0.0, 0.0, ws[-1]))
                f.close()

            self.fint_CPstar.append(interp1d(U_CT_CP_AD[:, 0], U_CT_CP_AD[:, 2], fill_value=0.0, bounds_error=False))
            self.fint_CTstar.append(interp1d(U_CT_CP_AD[:, 0], U_CT_CP_AD[:, 1], fill_value=0.0, bounds_error=False))

    def calc_wt_interaction(self, x_ilk, y_ilk, h_i=None, type_i=0, wd=None, ws=None, time=False,
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

        h_i, D_i = self.windTurbines.get_defaults(len(x_ilk), type_i, h_i)
        wd, ws = self.site.get_defaults(wd, ws)
        I, L, K, = len(x_ilk), len(wd), (1, len(ws))[time is False]
        kwargs.update(dict(x_ilk=x_ilk, y_ilk=y_ilk, h_ilk=h_i[:, na, na], wd=wd, ws=ws, time=time,
                           type_i=np.zeros_like(D_i) + type_i))

        for inputModifierModel in self.inputModifierModels:
            kwargs.update(inputModifierModel.setup(**kwargs))

        # Find local wind speed, wind direction, turbulence intensity and probability
        lw = self.site.local_wind(x=np.mean(x_ilk, (1, 2)), y=np.mean(y_ilk, (1, 2)), h=h_i,
                                  wd=kwargs['wd'], ws=kwargs['ws'], time=kwargs['time'])
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

        if 'type_i_ilk' not in kwargs:
            kwargs.update({'type_i_ilk': kwargs['type_i'][:, na, na]})
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
        z_ilk = self.site.elevation(kwargs['x_ilk'], kwargs['y_ilk'])
        self.site.distance.setup(
            np.asarray(kwargs['x_ilk']), np.asarray(kwargs['y_ilk']), np.asarray(kwargs['h_ilk']), np.asarray(z_ilk))

        if self.power_as_RANS:
            # Run seperately for every wind direction and calculate power from UAD afterwards
            power_ilk, WS_eff_ilk, ct_ilk, TI_eff_ilk = self.run_and_post_wt_output(kwargs)
            del kwargs['WS_eff_ilk']
            del kwargs['TI_eff_ilk']
        else:
            WS_eff_ilk, TI_eff_ilk, ct_ilk, kwargs = self._calc_wt_interaction(**kwargs)
            power_ilk = self.windTurbines.power(ws=WS_eff_ilk, **self.get_wt_kwargs(TI_eff_ilk, kwargs))

        kwargs.update({'time': time, 'type_i': type_i})
        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, lw, kwargs

    def run_and_post_wt_output(self, kwargs):
        """
        Run lut model and calculate wt power and thrust by the integrating rotor averaged
        wind speed as done in EllipSys. flow_map is used to interpolate the integration points;
        this method requires a lot of memory when many points are extracted at the same time in
        combination with a large wind farm. The variable flowmap_maxpoints can be used to reduce
        the number of points per flow_map call for the cost of computational speed.
        """

        # Integrate UAD from PyWake LUT with the aim of getting the same UAD as PyWakeEllipSys

        type_i = np.asarray(kwargs['type_i'], dtype=int)
        uniquetypes = np.unique(type_i)
        types = []
        for i in range(len(uniquetypes)):
            types.append(np.where(type_i == uniquetypes[i])[0])
        zRefs = self.windTurbines.hub_height(uniquetypes)
        Drefs = self.windTurbines.diameter(uniquetypes)

        # Calculate ad integration parameters
        weights = self.rotorAvgModelPost.nodes_weight[na, :, na]
        nr = self.rotorAvgModelPost.n_rnodes
        ntheta = self.rotorAvgModelPost.n_theta
        I, L, K = kwargs['I'], kwargs['L'], kwargs['K']
        pos = np.zeros((I, 3))
        for i in range(len(uniquetypes)):
            pos[types[i], 2] = zRefs[i]
        o3 = np.zeros((I, 3))
        o3[:, :] = np.array([1.0, 0.0, 0.0])[na, :]
        o2 = np.zeros((I, 3))
        o2[:, :] = np.array([0.0, 0.0, 1.0])[na, :]
        cxyz_i = np.zeros((I, nr - 1, ntheta, 3))
        cxyz_i[:, :, :, 0] = 0.5 * self.rotorAvgModelPost.nodes_x.reshape((nr - 1, ntheta))[na, :, :]
        cxyz_i[:, :, :, 1] = 0.5 * self.rotorAvgModelPost.nodes_y.reshape((nr - 1, ntheta))[na, :, :]
        for i in range(len(uniquetypes)):
            cxyz_i[types[i], :, :, 0:2] = cxyz_i[types[i], :, :, 0:2] * Drefs[i]

        wd = kwargs['wd']
        ws = kwargs['ws']
        shape = (I, L, K)
        power_ilk = np.zeros(shape)
        ct_ilk = np.zeros(shape)
        WS_eff_ilk = np.zeros(shape)
        TI_eff_ilk = np.zeros(shape)
        Cnsts = 0.5 * 1.225 * np.asarray(Drefs) ** 2 * 0.25 * np.pi
        if self.flowmap_maxpoints is not None:
            flowmap_points = (nr - 1) * ntheta * I
            nflowmaps = int(np.ceil(flowmap_points) / self.flowmap_maxpoints)
            i1last = (nflowmaps - 1) * self.flowmap_maxpoints
            WS_eff_jlk = np.zeros((flowmap_points, 1, K))
            TI_eff_jlk = np.zeros((flowmap_points, 1, K))
        for l in range(L):
            print('wd=', wd[l])
            sys.stdout.flush()
            starttime = time.time()
            kwargs2 = kwargs.copy()
            kwargs2['wd'] = [wd[l]]
            kwargs2['L'] = 1
            kwargs2['WD_ilk'][:, :, :] = wd[l]
            WS_eff_ilk, TI_eff_ilk[:, l, :][:, na, :], ct_ilk, kwargs3 = self._calc_wt_interaction(**kwargs2)

            localWind = self.site.local_wind(x=kwargs['x_ilk'][:, l, 0], y=kwargs['y_ilk'][:, l, 0], h=kwargs2['h_ilk'][:, l, 0], wd=wd[l], ws=ws)
            simres = SimulationResult(self, localWind=localWind,
                                      WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk[:, l, :][:, na, :],
                                      power_ilk=power_ilk[:, l, :][:, na, :], ct_ilk=ct_ilk, **kwargs3)

            endtime = time.time()
            print('sim time', endtime - starttime, 'sec')

            starttime = time.time()
            pos[:, 0] = kwargs['x_ilk'][:, l, 0]
            pos[:, 1] = kwargs['y_ilk'][:, l, 0]
            deg = 270.0 - wd[l]
            o3[:, 0] = np.cos(deg * (np.pi / 180.0))
            o3[:, 1] = np.sin(deg * (np.pi / 180.0))
            cxyzg_i = cxzy_to_cxyzg(cxyz_i, o3, o2, pos)
            x_j = cxyzg_i[:, :, :, 0].flatten()
            y_j = cxyzg_i[:, :, :, 1].flatten()
            h_j = cxyzg_i[:, :, :, 2].flatten()
            if self.flowmap_maxpoints is None:
                lw_j, WS_eff_jlk, TI_eff_jlk = self._flow_map(x_j, y_j, h_j, simres)
            else:
                for i in range(nflowmaps - 1):
                    i1 = i * self.flowmap_maxpoints
                    i2 = (i + 1) * self.flowmap_maxpoints
                    lw_j, WS_eff_jlk[i1:i2, :, :], TI_eff_jlk[i1:i2, :, :] = self._flow_map(x_j[i1:i2], y_j[i1:i2], h_j[i1:i2], simres)
                lw_j, WS_eff_jlk[i1last:, :, :], TI_eff_jlk[i1last:, :, :] = self._flow_map(x_j[i1last:], y_j[i1last:], h_j[i1last:], simres)

            # Integrate U over AD
            WS_eff_ilk[:, l, :] = np.sum(np.reshape(WS_eff_jlk, (I, ntheta * (nr - 1), len(ws))) * weights, axis=1)

            # Interpolate CP and CT
            if len(uniquetypes) > 1:
                lut_cpstar = np.zeros(WS_eff_ilk[:, 0, :].shape)
                for i in range(len(uniquetypes)):
                    lut_cpstar[types[i], :] = self.fint_CPstar[i](WS_eff_ilk[types[i], l, :])
                    ct_ilk[types[i], l, :] = self.fint_CTstar[i](WS_eff_ilk[types[i], l, :])
            else:
                lut_cpstar = self.fint_CPstar[0](WS_eff_ilk[:, l, :])
                ct_ilk[:, l, :] = self.fint_CTstar[0](WS_eff_ilk[:, l, :])

            # Calculate power and make sure to stay rated (due to CP interpolation errors)
            if len(uniquetypes) > 1:
                for i in range(len(uniquetypes)):
                    power_ilk[types[i], l, :] = lut_cpstar[types[i], :] * Cnsts[i] * WS_eff_ilk[types[i], l, :] ** 3
                    power_ilk[types[i], l, :][power_ilk[types[i], l, :] > self.Prated[i]] = self.Prated[i]
            else:
                power_ilk[:, l, :] = lut_cpstar * Cnsts[0] * WS_eff_ilk[:, l, :] ** 3
                power_ilk[power_ilk > self.Prated[0]] = self.Prated[0]

            # Set power and ct to zero for wts that were originally shut down
            wt_shutdown = np.where(simres.power_ilk == 0.0)
            ct_ilk[wt_shutdown] = 0.0
            power_ilk[wt_shutdown] = 0.0
            # print('power_ilk', power_ilk, 'WS_eff_ilk', WS_eff_ilk)
            endtime = time.time()
            print('post time:', endtime - starttime, 'sec')
        return power_ilk, WS_eff_ilk, ct_ilk, TI_eff_ilk


# The rans_lut from mmpe does not lead to satisfying results when using the shear correction for a row of wts.
# The shear correction does work well for the power of a single wt.
# This needs to be future investigated
# lut files with multple turbines can be provided using type_i dimension. In this case, the coords need to be in [m]
class RANSLUTDeficit(XRLUTDeficitModel):
    """ Expects LUT velocity deficit xarray with coords in [m] and PyWake variable names """
    def __init__(self, lut=None, rotorAvgModel=None, groundModel=None, use_effective_ws=True, use_effective_ti=False, shearcor=False):
        if lut is None:
            ds = fugalut2ranslut()
            lut = ds.deficits.sel(TI_eff_ilk=0.075)
        if isinstance(lut, str):
            lut = xr.load_dataset(lut).deficits
        lut = (1 - lut)
        XRLUTDeficitModel.__init__(self, lut, bounds='limit', rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                   use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)

    def get_output(self, output_ijlk, WS_eff_ilk, **_):
        return (1.0 - output_ijlk) * WS_eff_ilk[:, na]

    def wake_radius(self, dw_ijlk, **kwargs):
        # Required for PyWake but not needed for RANS LUT model
        wake_radius_ijlk = np.ones(dw_ijlk.shape)
        return wake_radius_ijlk


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
        import matplotlib.pyplot as plt

        site = IEA37Site(16)
        wt = GenericWindTurbine(name='15MW', diameter=236, hub_height=150, power_norm=15000, max_cp=0.45, constant_ct=0.8)
        path_LUT = '/groups/PyWakeEllipSys/LUT/EnergyIsland_WTgen15MW.nc'
        lut = xr.load_dataset(path_LUT)
        zRef = 150.0
        Dref = 236.0
        deficit = RANSLUTDeficit(lut.deficits, zRef, Dref)
        turb = RANSLUTTurbulence(lut.added_ti, zRef, Dref)
        wfm = All2AllIterative(site, wt, wake_deficitModel=deficit,
                               blockage_deficitModel=deficit,
                               turbulenceModel=turb)
        x, y = site.initial_position.T
        sim_res = wfm(x, y, wd=270)
        fm = sim_res.flow_map()
        plt.figure()
        fm.plot_wake_map()
        plt.show()


main()
