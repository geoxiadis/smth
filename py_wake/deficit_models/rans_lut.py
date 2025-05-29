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
from py_wake.utils.rans_lut_utils import convert_ranslut, cxzy_to_cxyzg
from py_wake.rotor_avg_models import EllipSysPolygonRotorAvg
from py_wake.wind_farm_models.wind_farm_model import SimulationResult


class RANSLUT(All2AllIterative):
    def __init__(self, path_LUT, site, windTurbines, zRef, Dref, rotorAvgModel=None,
                 power_as_RANS=True, rotorAvgModelPost=EllipSysPolygonRotorAvg(n_r=9, n_theta=32),
                 flowmap_maxpoints=10000,
                 ADcontrolfiles=None, ADcontrolfiles_write=False, cal_TI=0.06,
                 ws_cutin=[4.0], ws_cutout=[25.0], dws=1.0, Prated=[2e6]):
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
        flowmap_maxpoints: int
            Maximum number of interpolation points during post turbine step
            to reduce memomry usage
        ADcontrolfiles : list
            If not None, a list of filename(s) is expected including
            columns of U_AD, CTstar, CPstar, which could come from RANS.
            If None then a CTstar, CPstar = f(U_AD) relationship is calculated from
            the lut file.
        ADcontrolfiles_write : bool
            If True and ADcontrolfiles is None, then the lut calculated
            CTstar, CPstar = f(U_AD) relationship is written to a file.
        cal_TI : float
            Reference Ti value to calculate shear for CTstar, CPstar = f(U_AD) relationship,
            only used to when ADcontrolfiles is None.
        ws_cutin : list
            List of cutin wind speeds with length equal to number of different wts,
            only used to when ADcontrolfiles is None.
        ws_cutin : list
            List of cutin wind speeds with length equal to number of different wts,
            only used to when ADcontrolfiles is None.
        dws : float
            Wind speed interval for lut calculated CTstar, CPstar = f(U_AD) relationship,
            only used to when ADcontrolfiles is None.
        Prated : list
            List of rated power in [W] with length equal to number of different wts,
        """
        self.power_as_RANS = power_as_RANS
        self.rotorAvgModelPost = rotorAvgModelPost
        self.flowmap_maxpoints = flowmap_maxpoints
        self.ADcontrolfiles = ADcontrolfiles
        self.ADcontrolfiles_write = ADcontrolfiles_write
        self.cal_TI = cal_TI
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

        self.get_CPCTstar_lut()

    def rescale_lut_coords(self, lut, zRef, Dref):
        if Dref is not None and zRef is not None:
            # Rescale non-dimensional coord to [m]
            lut['x'] = lut['x'] * Dref
            lut['y'] = lut['y'] * Dref
            lut['z'] = lut['z'] * Dref + zRef
        else:  # pragma: no cover
            if Dref is None:
                print('ERROR: lut coords cannot be rescaled because Dref is not specified')
            elif zRef is None:
                print('ERROR: lut coords cannot be rescaled because zRef is not specified')
            sys.exit()
        return lut

    def rename_lut(self, lut):
        lutvars = ['x', 'y', 'z', 'ct', 'ti', 'yaw']
        renamevars = ['dw_ijlk', 'hcw_ijlk', 'z_ijlk', 'ct_ilk', 'TI_eff_ilk', 'yaw_ilk']
        d = {lutvars[i]: renamevars[i] for i in range(len(lutvars)) if lutvars[i] in lut.coords}
        return lut.rename(d)

    def get_CPCTstar_lut(self):
        if self.power_as_RANS:
            # CP and CT as function of rotot-averaged wind speed used for post processing
            self.fint_CPstar = []
            self.fint_CTstar = []
            if self.ADcontrolfiles is None:
                self.cal_lut()
            else:
                self.read_ad_controlfiles()

    def read_ad_controlfiles(self):
        # RANS based calibrated control file for CPstar and CTstar as function of
        # the disk averaged velocity UAD provided by the user
        for ADcontrolfile in self.ADcontrolfiles:
            RANScal = np.genfromtxt(ADcontrolfile, skip_header=True)
            self.fint_CPstar.append(interp1d(RANScal[:, 0], RANScal[:, 2], fill_value=0.0, bounds_error=False))
            self.fint_CTstar.append(interp1d(RANScal[:, 0], RANScal[:, 1], fill_value=0.0, bounds_error=False))

    def cal_lut(self, density=1.225, cmu=0.03, kappa=0.4):
        starttime = time.time()
        # Calculcate CPstar, CTstar = f(UAD) directly from single wake RANS-LUT
        data = self.wake_deficitModel.da.interp(dw_ijlk=0.0)
        weights = self.rotorAvgModelPost.nodes_weight[na, :]
        nr = self.rotorAvgModelPost.n_rnodes
        ntheta = self.rotorAvgModelPost.n_theta
        cxyz = np.zeros((nr - 1, ntheta, 3))
        cxyz[:, :, 0] = 0.5 * self.rotorAvgModelPost.nodes_x.reshape((nr - 1, ntheta))
        cxyz[:, :, 1] = 0.5 * self.rotorAvgModelPost.nodes_y.reshape((nr - 1, ntheta))

        uStar_UH = self.cal_TI * np.sqrt(1.5 * np.sqrt(cmu))
        # z0 = zRef / (np.exp(kappa / uStar_UH) - 1)

        if 'type_i' in data.dims:
            types = data.type_i.values
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

            cxyzg_i = cxzy_to_cxyzg(cxyz[na, :, :, :] * Dref, np.array([[1.0, 0.0, 0.0]]),
                                    np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, zH]]))
            y_j = cxyzg_i[:, :, :, 1].flatten()
            h_j = cxyzg_i[:, :, :, 2].flatten()
            if 'type_i' in data.dims:
                Udata = data.sel(type_i=type).interp(TI_eff_ilk=self.cal_TI).interp(hcw_ijlk=('i', y_j), z_ijlk=('i', h_j))
            else:
                Udata = data.interp(TI_eff_ilk=self.cal_TI).interp(hcw_ijlk=('i', y_j), z_ijlk=('i', h_j))

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

            if self.ADcontrolfiles_write:
                outfile = 'lutcal_WT%g_Ti%g_%i.dat' % (Dref, self.cal_TI, type)
                f = open(outfile, 'w')
                f.write('%s %s\n' % (str(ws.size + 2), ' 4'))
                f.write('%16.14f %16.14f %16.14f %16.14f\n' %
                        (0.95 * U_CT_CP_AD[0, 0], U_CT_CP_AD[0, 1], U_CT_CP_AD[0, 2], 0.95 * ws[0]))
                for k in range(len(ws)):
                    f.write('%16.14f %16.14f %16.14f %16.14f\n' %
                            (U_CT_CP_AD[k, 0], U_CT_CP_AD[k, 1], U_CT_CP_AD[k, 2], ws[k]))
                f.write('%16.14f %16.14f %16.14f %16.14f\n' % (ws[-1], 0.0, 0.0, ws[-1]))
                f.close()

            self.fint_CPstar.append(interp1d(U_CT_CP_AD[:, 0], U_CT_CP_AD[:, 2], fill_value=0.0, bounds_error=False))
            self.fint_CTstar.append(interp1d(U_CT_CP_AD[:, 0], U_CT_CP_AD[:, 1], fill_value=0.0, bounds_error=False))
            endtime = time.time()
            print('cal time', endtime - starttime, 'sec')

    def get_wt_output(self, WS_eff_ilk_in, TI_eff_ilk, ct_ilk_in, **kwargs):
        """
        Recalculate wt power and thrust by the integrating rotor averaged wind speed
        including induction and interpolation with a CPstar, CTstar = f(U_AD) relation,
        as done in EllipSys. flow_map is used to interpolate the integration points;
        this method requires a lot of memory when many points are extracted at the same time in
        combination with a large wind farm. The variable flowmap_maxpoints can be used to reduce
        the number of points per flow_map call for the cost of computational speed.
        """
        power_ilk_org = self.windTurbines.power(ws=WS_eff_ilk_in, **self.get_wt_kwargs(TI_eff_ilk, kwargs))

        if not self.power_as_RANS:
            return WS_eff_ilk_in, TI_eff_ilk, power_ilk_org, ct_ilk_in

        starttime = time.time()
        # type_i = np.asarray(kwargs['type_i_ilk'][:, 0, 0], dtype=int)
        type_i = np.asarray(kwargs['type_i'], dtype=int)
        uniquetypes = np.unique(type_i)
        types = []
        for i in range(len(uniquetypes)):
            types.append(np.where(type_i == uniquetypes[i])[0])

        # Calculate ad integration parameters
        weights = self.rotorAvgModelPost.nodes_weight[na, :, na]
        nr = self.rotorAvgModelPost.n_rnodes
        ntheta = self.rotorAvgModelPost.n_theta
        shape = WS_eff_ilk_in.shape
        I, L, K = shape
        pos = np.zeros((I, 3))
        pos[:, 0] = kwargs['x_ilk'][:, 0, 0]
        pos[:, 1] = kwargs['y_ilk'][:, 0, 0]
        pos[:, 2] = kwargs['h_ilk'][:, 0, 0]
        deg = 270.0 - kwargs['wd']
        o3 = np.zeros((I, L, 3))
        o3[:, :, :] = np.array([1.0, 0.0, 0.0])[na, :]
        o3[:, :, 0] = np.cos(deg * (np.pi / 180.0))[na, :]
        o3[:, :, 1] = np.sin(deg * (np.pi / 180.0))[na, :]
        o2 = np.zeros((I, 3))
        o2[:, :] = np.array([0.0, 0.0, 1.0])[na, :]
        cxyz_i = np.zeros((I, nr - 1, ntheta, 3))
        cxyz_i[:, :, :, 0] = 0.5 * self.rotorAvgModelPost.nodes_x.reshape((nr - 1, ntheta))[na, :, :]
        cxyz_i[:, :, :, 1] = 0.5 * self.rotorAvgModelPost.nodes_y.reshape((nr - 1, ntheta))[na, :, :]
        cxyz_i[:, :, :, 0:2] = cxyz_i[:, :, :, 0:2] * kwargs['D_i'][:, na, na, na]
        power_ilk = np.zeros(shape)
        ct_ilk = np.zeros(shape)
        WS_eff_ilk = np.zeros(shape)
        Cnsts = 0.5 * 1.225 * np.asarray(kwargs['D_i']) ** 2 * 0.25 * np.pi
        if self.flowmap_maxpoints is not None:
            flowmap_points = (nr - 1) * ntheta * I
            nflowmaps = int(np.ceil(flowmap_points) / self.flowmap_maxpoints)
            i1last = (nflowmaps - 1) * self.flowmap_maxpoints
            WS_eff_jlk = np.zeros((flowmap_points, 1, K))
            TI_eff_jlk = np.zeros((flowmap_points, 1, K))
        # Loop over wind directions because flow_map cannot include interpolation points (x_j, y_j, h_j) vectorized over wds
        # since they change orientation per wind directoin
        for l in range(L):
            # print('wd=', kwargs['wd'][l])
            # sys.stdout.flush()
            localWind = self.site.local_wind(
                x=kwargs['x_ilk'][:, 0, 0], y=kwargs['y_ilk'][:, 0, 0], h=kwargs['h_ilk'][:, 0, 0], wd=kwargs['wd'][l], ws=kwargs['ws'])
            kwargs2 = kwargs.copy()
            kwargs2['wd'] = [kwargs['wd'][l]]
            kwargs2['L'] = 1
            kwargs2['WD_ilk'] = np.array([[kwargs['wd'][l]]])
            simres = SimulationResult(self, localWind=localWind,
                                      WS_eff_ilk=WS_eff_ilk_in[:, l, :][:, na, :], TI_eff_ilk=TI_eff_ilk[:, l, :][:, na, :],
                                      power_ilk=power_ilk[:, l, :][:, na, :], ct_ilk=ct_ilk_in[:, l, :][:, na, :], **kwargs2)
            cxyzg_i = cxzy_to_cxyzg(cxyz_i, o3[:, l, :], o2, pos)
            x_j = cxyzg_i[:, :, :, 0].flatten()
            y_j = cxyzg_i[:, :, :, 1].flatten()
            h_j = cxyzg_i[:, :, :, 2].flatten()
            if self.flowmap_maxpoints is None:
                lw_j, WS_eff_jlk, TI_eff_jlk = self._flow_map(x_j, y_j, h_j, simres)
            else:
                for i in range(nflowmaps - 1):
                    i1 = i * self.flowmap_maxpoints
                    i2 = (i + 1) * self.flowmap_maxpoints
                    lw_j, WS_eff_jlk[i1:i2, :, :], TI_eff_jlk[i1:i2, :, :] = self._flow_map(
                        x_j[i1:i2], y_j[i1:i2], h_j[i1:i2], simres)
                lw_j, WS_eff_jlk[i1last:, :, :], TI_eff_jlk[i1last:, :, :] = self._flow_map(
                    x_j[i1last:], y_j[i1last:], h_j[i1last:], simres)

            # Integrate U over AD
            WS_eff_ilk[:, l, :] = np.sum(np.reshape(WS_eff_jlk, (I, ntheta * (nr - 1), K)) * weights, axis=1)

        # Interpolate CP and CT and calculate power
        lut_cpstar = np.zeros(WS_eff_ilk[:, :, :].shape)
        for i in range(len(uniquetypes)):
            lut_cpstar[types[i], :, :] = self.fint_CPstar[i](WS_eff_ilk[types[i], :, :])
            ct_ilk[types[i], :, :] = self.fint_CTstar[i](WS_eff_ilk[types[i], :, :])
            power_ilk[types[i], :, :] = lut_cpstar[types[i], :, :] * \
                Cnsts[types[i]][0] * WS_eff_ilk[types[i], :, :] ** 3
            # Make sure to stay rated (due to CP interpolation errors)
            power_ilk_i = power_ilk[types[i], :, :]
            power_ilk_i[power_ilk_i > self.Prated[i]] = self.Prated[i]
            power_ilk[types[i], :, :] = power_ilk_i

        # Set power and ct to zero for wts that were originally shut down
        wt_shutdown = np.where(power_ilk_org == 0.0)
        ct_ilk[wt_shutdown] = 0.0
        power_ilk[wt_shutdown] = 0.0
        # print('power_ilk', power_ilk, 'WS_eff_ilk', WS_eff_ilk)
        endtime = time.time()
        print('post time', endtime - starttime, 'sec')
        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk


class RANSLUTDeficit(XRLUTDeficitModel):
    """ Expects LUT velocity deficit xarray with coords in [m] and PyWake variable names """

    def __init__(self, lut=None, rotorAvgModel=None, groundModel=None,
                 use_effective_ws=True, use_effective_ti=False):
        if lut is None:
            # Load default RANS LUT file based on a demo V80 LUT
            ds = convert_ranslut()
            lut = ds.deficits.sel(TI_eff_ilk=0.06)
        lut = (1 - lut)
        XRLUTDeficitModel.__init__(self, lut, bounds='limit', rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                   use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)

    def get_output(self, output_ijlk, WS_eff_ilk, **_):
        return (1.0 - output_ijlk) * WS_eff_ilk[:, na]

    def wake_radius(self, dw_ijlk, **kwargs):
        # Required for PyWake but not needed for RANS LUT model
        wake_radius_ijlk = np.ones(dw_ijlk.shape)
        return wake_radius_ijlk
