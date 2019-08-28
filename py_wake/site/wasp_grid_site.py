from py_wake.site._site import Site, UniformWeibullSite
import numpy as np
import xarray as xr
import pickle
import os
import glob
from _collections import defaultdict
import re
from scipy.interpolate.interpolate import RegularGridInterpolator
import copy
from numpy import newaxis as na


class WaspGridSite(UniformWeibullSite):
    def __init__(self, ds, k_star=0.075, r_i=None, distance_type='direct', turning=False, calc_all=False, terrain_step=5):
        '''
        k_star : float, wake decay parameter
        r_i :
        '''
#        self.z0 = z0
#        self.h_ref = h_ref
        self.calc_all = calc_all
        self.terrain_step = terrain_step
        self.k_star = k_star
        self.r_i = r_i
        self.distance_type = distance_type
        self.turning = turning
        self._ds = ds
        self.interp_funcs = None
        self.interp_funcs_initialization()
        self.elevation_interpolator = EqDistRegGrid2DInterpolator(self._ds.coords['x'].data,
                                                                  self._ds.coords['y'].data,
                                                                  self._ds['elev'].data)
        self.TI_data_exist = 'tke' in self.interp_funcs.keys()
        super().__init__(p_wd=np.nanmean(self._ds['f'].data, (0, 1, 2)),
                         a=np.nanmean(self._ds['A'].data, (0, 1, 2)),
                         k=np.nanmean(self._ds['k'].data, (0, 1, 2)),
                         ti=0)

    def local_wind(self, x_i, y_i, h_i, wd=None, ws=None, wd_bin_size=None, ws_bin_size=None):
        if wd is None:
            wd = self.default_wd
        if ws is None:
            ws = self.default_ws
        wd, ws = np.asarray(wd), np.asarray(ws)
        self.wd = wd
        h_i = np.asarray(h_i)
        x_il, y_il, h_il = [np.repeat([v], len(wd), 0).T for v in [x_i, y_i, h_i]]

        ws_bin_size = self.ws_bin_size(ws, ws_bin_size)
        wd_bin_size = self.wd_bin_size(wd, wd_bin_size)

        wd_il = np.repeat([wd], len(x_i), 0)
        speed_up_il, turning_il, wind_shear_il = \
            [self.interp_funcs[n]((x_il, y_il, h_il, wd_il)) for n in
             ['spd', 'orog_trn', 'wind_shear']]

        # TODO: if speed_up_il does not vary with height the below version of WS_ilk should be used. Else remove z0 and h_ref from the input.
#        term_denominator = np.log(self.h_ref / self.z0)

#        WS_ilk = (ws[na, na, :] * np.log(h_i / self.z0)[:, na, na] /
#                  term_denominator) * speed_up_il[:, :, na]
        WS_ilk = ws[na, na, :] * speed_up_il[:, :, na]

        WD_ilk = (wd_il + turning_il)[..., na]

        if self.TI_data_exist:
            TI_il = self.interp_funcs['tke']((x_il, y_il, h_il, wd_il))
            TI_ilk = (TI_il[:, :, na] * (0.75 + 3.8 / WS_ilk))

        WD_lk, WS_lk = np.meshgrid(wd, ws, indexing='ij')
        P_ilk = self.probability(x_i, y_i, h_i, WD_lk, WS_lk, wd_bin_size, ws_bin_size)
#        P_ilk = self.probability(x_i, y_i, h_i, WD_lk, WS_ilk, wd_bin_size, ws_bin_size)
        return WD_ilk, WS_ilk, TI_ilk, P_ilk

#    def probability(self, x_i, y_i, h_i, WD_lk, WS_ilk, wd_bin_size, ws_bin_size):
    def probability(self, x_i, y_i, h_i, WD_lk, WS_lk, wd_bin_size, ws_bin_size):
        """See Site.probability
        """
        x_il, y_il, h_il = [np.repeat([v], WD_lk.shape[0], 0).T for v in [x_i, y_i, h_i]]
        wd_il = np.repeat([WD_lk.mean(1)], len(x_i), 0)
        Weibull_A_il, Weibull_k_il, freq_il = \
            [self.interp_funcs[n]((x_il, y_il, h_il, wd_il)) for n in
             ['A', 'k', 'f']]
        P_wd_il = freq_il * wd_bin_size
        WS_ilk = WS_lk[na]
        # TODO: The below probability does not sum to 1 due to the speed ups. New calculation of the weibull weights are needed.
        P_ilk = self.weibull_weight(WS_ilk, Weibull_A_il[:, :, na],
                                    Weibull_k_il[:, :, na], ws_bin_size) * P_wd_il[:, :, na]
#        P_ilk = self.weibull_weight(WS_ilk, Weibull_A_il[:, :, na],
#                                    Weibull_k_il[:, :, na], ws_bin_size) * P_wd_il[:, :, na]
        self.freq_il = freq_il
        self.pdf_ilk = self.weibull_weight(WS_ilk, Weibull_A_il[:, :, na],
                                           Weibull_k_il[:, :, na], ws_bin_size)
        self.Weibull_k = Weibull_k_il[:, :, na]
        self.Weibull_A = Weibull_A_il[:, :, na]
        self.Weibull_ws = WS_ilk
        return P_ilk

    def distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il):
        if self.distance_type == 'terrain_following':
            if not src_x_i.shape == dst_x_j.shape or not np.allclose(src_x_i, dst_x_j):
                raise NotImplementedError('Different source and destination postions are not yet implemented for the terrain following distance calculation')
            return self.cal_dist_terrain_following(src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il, self.terrain_step, self.calc_all)
        else:
            return UniformWeibullSite.distances(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il)

    def cal_dist_terrain_following(self, src_x_i, src_y_i, src_h_i, dst_x_j, dst_y_j, dst_h_j, wd_il, step, calc_all):
        """ Calculate downwind and crosswind distances between a set of turbine
        sites, for a range of inflow wind directions. This version assumes the
        flow follows terrain at the same height above ground, and calculate the
        terrain following downwind/crosswind distances.

        Parameters
        ----------
        x_i: array:float
            x coordinates [m]

        y_i: array:float
            y coordinates [m]

        H_i: array:float
            hub-heights [m]

        wd: array:float
            local inflow wind direction [deg] (N = 0, E = 90, S = 180, W = 270)

        Note: wd can be a 1D array, which denotes the binned far field inflow
        wind direction, or it can be a num_sites by num_wds 2D array, which
        denotes the local wind direction for each sites. The 2D array version
        is used when the wind farm is at complex terrain, and the differences
        between local wind direction don't want to be neglected.

        elev_interp_func: any:'scipy.interpolate.interpolate.RegularGridInterpolator'
            interperating function to get elevation [m] at any site. It is a
            RegularGridInterpolator based function provided by site_condition.
            Its usage is like: elev = elev_interp_func(x, y), for sites
            outside the legal area, it will return nan.

        step: float
            stepsize when integrating to calculate terrain following distances
            [m], default: 5.0

        calc_all: bool
            If False (default) only distances to sites within wake are corrected
            for effects of terrain elevation

        Returns
        -------
        dist_down: array:float
            downwind distances between turbine sites for different far field
            inflow wind direction [m/s]

        dist_cross: array:float
            crosswind distances between turbine sites for all wd [m/s]

        downwind_order: array:integer
            downwind orders of turbine sites for different far field inflow
            wind direction [-]

        Note: dist_down[i, j, l] denotes downwind distance from site i to site
        j under lth inflow wd. downwind_order[:, l] denotes the downwind order
        of sites under lth inflow wd.
        """

        """ Calculate terrain following downwind distances from (x_start,
        y_starts) to a set of points at (x_points, y_points), assuming
        the wind blows along x direction. elev_interp_func is used to get elevation values,
        cos_rotate_back and sin_rotate_back are used to tranform the
        coordinates to real coordinates (to be used in elev_interp_func).
        Suffixs:
            - i: starting points
            - l: wind direction
            - s: destination points
        """
        if wd_il.ndim == 2 and wd_il.shape[0] == 1:
            wd_il = wd_il[0]
        elev_interp_func = self.elevation_interpolator
        x_i = src_x_i
        y_i = src_y_i
        H_i = src_h_i
        if not self.turning and hasattr(self, 'wd'):
            wd = self.wd
        else:
            wd = wd_il
        dist_down_straight_iil, dist_cross_iil, downwind_order_il, x_rotated_il, y_rotated_il, cos_wd_il, sin_wd_il, hcw_iil, dh_iil = self._cal_dist(x_i, y_i, H_i, dst_x_j, dst_y_j, dst_h_j, wd)

        dist_down_isl = dist_down_straight_iil
        dist_cross_isl = dist_cross_iil
        dist_hcw_isl = hcw_iil
        dist_dh_isl = dh_iil
        x_start_il = x_rotated_il
        y_start_il = y_rotated_il
        x_points_sl = x_rotated_il
        cos_rotate_back_il = cos_wd_il
        sin_rotate_back_il = -sin_wd_il

        I, L = x_start_il.shape
        i_wd_l = np.arange(L)
        na = np.newaxis
        if calc_all or not np.array(self.r_i).any():
            mask_isl = np.ones_like(dist_down_isl, dtype=np.bool)
        else:
            r_i = self.r_i
            # Only consider if: r0 + k_star * dist_down < dist_cross-r1 and straight distance > 2*r
            # where r1 is radius of downwind turbine.
            # dist_down is expected to be less than dist_straight + 20%
            expected_wake_size = 1.2 * dist_down_isl * self.k_star * np.ones(len(r_i))[:, na, na] + r_i[:, na, na] + r_i[na, :, na]
            mask_isl = (expected_wake_size > dist_cross_isl) & (dist_down_isl > (2 * r_i.max()))

        any_sites_in_wake = np.any(mask_isl, 1)
        for l in i_wd_l:
            for i in range(I):
                x_start = x_start_il[i, l]
                y_start = y_start_il[i, l]

                x_points_s = x_points_sl[:, l]

                if not any_sites_in_wake[i, l]:
                    continue

                # find most downstream relevant site
                i_last_point = downwind_order_il[np.where(mask_isl[i, downwind_order_il[:, l], l])[0][-1], l]
                x_last_point = x_points_sl[i_last_point, l]

                cos_rotate_back = cos_rotate_back_il[i, l]
                sin_rotate_back = sin_rotate_back_il[i, l]

                # 2. if starting point is not already the most downwind point
                if x_start < x_last_point:

                    # 3. For points downwind of the starting point, do integration
                    # along a line for starting points to the final point
                    x_integ_line = np.arange(x_start, x_last_point + step, step)
                    x_integ_line[-1] = x_last_point

                    # note these coordinates need to be rotated back to get elevation
                    x_integ_line_back = (x_integ_line * cos_rotate_back +
                                         y_start * sin_rotate_back)
                    y_integ_line_back = (y_start * cos_rotate_back -
                                         x_integ_line * sin_rotate_back)
                    x_max_inds = x_integ_line_back > elev_interp_func.x.max()
                    y_max_inds = y_integ_line_back > elev_interp_func.y.max()
                    x_min_inds = x_integ_line_back < elev_interp_func.x.min()
                    y_min_inds = y_integ_line_back < elev_interp_func.y.min()
                    x_integ_line_back[x_max_inds] = elev_interp_func.x.max()
                    y_integ_line_back[y_max_inds] = elev_interp_func.y.max()
                    x_integ_line_back[x_min_inds] = elev_interp_func.x.min()
                    y_integ_line_back[y_min_inds] = elev_interp_func.y.min()
                    z_integ_line = elev_interp_func(x_integ_line_back,
                                                    y_integ_line_back,
                                                    mode='extrapolate')

                    # integration along the line
                    dx = np.diff(x_integ_line)
                    dz = np.diff(z_integ_line)
                    ds = np.sqrt(dx**2 + dz**2)
                    dist_line = np.cumsum(ds)

                    downwind = x_start < x_points_s
                    within_wake = mask_isl[i, :, l]
                    update = downwind & within_wake
                    dist_down_isl[i, update, l] = np.interp(x_points_s[update], x_integ_line[1:], dist_line)
#                    if np.isnan(dist_down_isl).any():
#                        print(dist_down_isl)
        self.dist_down, self.dist_cross, self.downwind_order = dist_down_isl, dist_hcw_isl, downwind_order_il.T

        return dist_down_isl, dist_hcw_isl, dist_dh_isl, downwind_order_il.T

    def _cal_dist(self, x_i, y_i, H_i, dst_x_j, dst_y_j, dst_h_j, wd):
        num_sites = len(x_i)

        if len(wd.shape) == 2:
            num_wds = wd.shape[1]
            wd_mean_l = np.mean(wd, 0)
            complex_flag = True
        else:
            complex_flag = False
            num_wds = wd.shape[0]
            wd_mean_l = wd

        I, L = num_sites, num_wds
        na = np.newaxis

        # rotate the coordinate so that wind blows along x axis
        rotate_angle_mean_l = (270 - wd_mean_l) * np.pi / 180.0

        dx_ii, dy_ii, dH_ii = [np.subtract(*np.meshgrid(v, v)) for v in [x_i, y_i, H_i]]
        if complex_flag:
            rotate_angle_il = (270 - wd) * np.pi / 180.0
            cos_wd_il = np.cos(rotate_angle_il)
            sin_wd_il = np.sin(rotate_angle_il)
        else:
            cos_wd_il = np.broadcast_to(np.cos(rotate_angle_mean_l), (I, L))
            sin_wd_il = np.broadcast_to(np.sin(rotate_angle_mean_l), (I, L))

        # using local wind direction to rotate
        x_rotated_il = x_i[:, na] * cos_wd_il + y_i[:, na] * sin_wd_il
        y_rotated_il = y_i[:, na] * cos_wd_il - x_i[:, na] * sin_wd_il

        downwind_order_il = np.argsort(x_rotated_il, 0).astype(np.int)

        dist_down_iil = dx_ii[:, :, na] * cos_wd_il[:, na, :] + dy_ii[:, :, na] * sin_wd_il[:, na, :]
        dy_rotated_iil = dy_ii[:, :, na] * cos_wd_il[:, na, :] - dx_ii[:, :, na] * sin_wd_il[:, na, :]

        # treat crosswind distances as usual
        dist_cross_iil = np.sqrt(dy_rotated_iil**2 + dH_ii[:, :, na]**2)
        hcw_iil = dy_rotated_iil
        dh_iil = np.repeat(dH_ii[:, :, na], 360, axis=-1)

        return dist_down_iil, dist_cross_iil, downwind_order_il, x_rotated_il, y_rotated_il, cos_wd_il, sin_wd_il, hcw_iil, dh_iil

    def elevation(self, x_i, y_i):
        return self.elevation_interpolator(x_i, y_i)

    @classmethod
    def from_pickle(cls, file_name):
        with open(file_name, 'rb') as f:
            ds = pickle.load(f)
        return cls(ds)

    def to_pickle(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self._ds, f, protocol=-1)

    def interp_funcs_initialization(self,
                                    interp_keys=['A', 'k', 'f', 'tke', 'spd', 'orog_trn', 'elev',
                                                 'wind_shear']):
        """ Initialize interpolating functions using RegularGridInterpolator
        for specified variables defined in interp_keys.
        """

        interp_funcs = {}

        for key in interp_keys:
            try:
                dr = self._ds.data_vars[key]
            except KeyError:
                print('Warning! {0} is not included in the current'.format(key) +
                      ' WindResourceGrid object.\n')
                continue

            coords = []

            data = dr.data
            for dim in dr.dims:
                # change sector index into wind direction (deg)
                if dim == 'sec':
                    num_sec = len(dr.coords[dim].data)   # number of sectors
                    coords.append(
                        np.linspace(0, 360, num_sec + 1))
                    data = np.concatenate((data, data[:, :, :, :1]), axis=3)
                elif dim == 'z' and len(dr.coords[dim].data) == 1:
                    # special treatment for only one layer of data
                    height = dr.coords[dim].data[0]
                    coords.append(np.array([height - 1.0, height + 1.0]))
                    data = np.concatenate((data, data), axis=2)
                else:
                    coords.append(dr.coords[dim].data)

            interp_funcs[key] = RegularGridInterpolator(
                coords,
                data, bounds_error=False)
        self.interp_funcs = interp_funcs

    @classmethod
    def from_wasp_grd(cls, path, globstr='*.grd', speedup_using_pickle=True):
        '''
        Reader for WAsP .grd resource grid files.

        Parameters
        ----------
        path: str
            path to file or directory containing goldwind excel files

        globstr: str
            string that is used to glob files if path is a directory.

        Returns
        -------
        WindResourceGrid: :any:`WindResourceGrid`:

        Examples
        --------
        >>> from mowflot.wind_resource import WindResourceGrid
        >>> path = '../mowflot/tests/data/WAsP_grd/'
        >>> wrg = WindResourceGrid.from_wasp_grd(path)
        >>> print(wrg)
            <xarray.Dataset>
            Dimensions:            (sec: 12, x: 20, y: 20, z: 3)
            Coordinates:
              * sec                (sec) int64 1 2 3 4 5 6 7 8 9 10 11 12
              * x                  (x) float64 5.347e+05 5.348e+05 5.349e+05 5.35e+05 ...
              * y                  (y) float64 6.149e+06 6.149e+06 6.149e+06 6.149e+06 ...
              * z                  (z) float64 10.0 40.0 80.0
            Data variables:
                flow_inc           (x, y, z, sec) float64 1.701e+38 1.701e+38 1.701e+38 ...
                ws_mean            (x, y, z, sec) float64 3.824 3.489 5.137 5.287 5.271 ...
                meso_rgh           (x, y, z, sec) float64 0.06429 0.03008 0.003926 ...
                obst_spd           (x, y, z, sec) float64 1.701e+38 1.701e+38 1.701e+38 ...
                orog_spd           (x, y, z, sec) float64 1.035 1.039 1.049 1.069 1.078 ...
                orog_trn           (x, y, z, sec) float64 -0.1285 0.6421 0.7579 0.5855 ...
                power_density      (x, y, z, sec) float64 77.98 76.96 193.5 201.5 183.9 ...
                rix                (x, y, z, sec) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
                rgh_change         (x, y, z, sec) float64 6.0 10.0 10.0 10.0 6.0 4.0 0.0 ...
                rgh_spd            (x, y, z, sec) float64 1.008 0.9452 0.8578 0.9037 ...
                f                  (x, y, z, sec) float64 0.04021 0.04215 0.06284 ...
                tke                (x, y, z, sec) float64 1.701e+38 1.701e+38 1.701e+38 ...
                A                  (x, y, z, sec) float64 4.287 3.837 5.752 5.934 5.937 ...
                k                  (x, y, z, sec) float64 1.709 1.42 1.678 1.74 1.869 ...
                flow_inc_tot       (x, y, z) float64 1.701e+38 1.701e+38 1.701e+38 ...
                ws_mean_tot        (x, y, z) float64 5.16 6.876 7.788 5.069 6.85 7.785 ...
                power_density_tot  (x, y, z) float64 189.5 408.1 547.8 178.7 402.2 546.6 ...
                rix_tot            (x, y, z) float64 0.0 0.0 0.0 9.904e-05 9.904e-05 ...
                tke_tot            (x, y, z) float64 1.701e+38 1.701e+38 1.701e+38 ...
                A_tot              (x, y, z) float64 5.788 7.745 8.789 5.688 7.716 8.786 ...
                k_tot              (x, y, z) float64 1.725 1.869 2.018 1.732 1.877 2.018 ...
                elev               (x, y) float64 37.81 37.42 37.99 37.75 37.46 37.06 ...

        '''

        var_name_dict = {
            'Flow inclination': 'flow_inc',
            'Mean speed': 'ws_mean',
            'Meso roughness': 'meso_rgh',
            'Obstacles speed': 'obst_spd',
            'Orographic speed': 'orog_spd',
            'Orographic turn': 'orog_trn',
            'Power density': 'power_density',
            'RIX': 'rix',
            'Roughness changes': 'rgh_change',
            'Roughness speed': 'rgh_spd',
            'Sector frequency': 'f',
            'Turbulence intensity': 'tke',
            'Weibull-A': 'A',
            'Weibull-k': 'k',
            'Elevation': 'elev',
            'AEP': 'aep'}

        def _read_grd(filename):

            def _parse_line_floats(f):
                return [float(i) for i in f.readline().strip().split()]

            def _parse_line_ints(f):
                return [int(i) for i in f.readline().strip().split()]

            with open(filename, 'rb') as f:
                file_id = f.readline().strip().decode()
                nx, ny = _parse_line_ints(f)
                xl, xu = _parse_line_floats(f)
                yl, yu = _parse_line_floats(f)
                zl, zu = _parse_line_floats(f)
                values = np.array([l.split() for l in f.readlines() if l.strip() != b""],
                                  dtype=np.float)  # around 8 times faster

            xarr = np.linspace(xl, xu, nx)
            yarr = np.linspace(yl, yu, ny)

            # note that the indexing of WAsP grd file is 'xy' type, i.e.,
            # values.shape == (xarr.shape[0], yarr.shape[0])
            # we need to transpose values to match the 'ij' indexing
            values = values.T
            #############
            # note WAsP denotes unavailable values using very large numbers, here
            # we change them into np.nan, to avoid strange results.
            values[values > 1e20] = np.nan

            return xarr, yarr, values

        if speedup_using_pickle:
            if os.path.isdir(path):
                pkl_fn = os.path.join(path, os.path.split(os.path.dirname(path))[1] + '.pkl')
                if os.path.isfile(pkl_fn):
                    return WaspGridSite.from_pickle(pkl_fn)
                else:
                    site_conditions = WaspGridSite.from_wasp_grd(path, globstr, speedup_using_pickle=False)
                    site_conditions.to_pickle(pkl_fn)
                    return site_conditions
            else:
                raise NotImplementedError

        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, globstr)))
        else:
            raise Exception('Path was not a directory...')

        file_height_dict = defaultdict(list)

        pattern = re.compile(r'Sector (\w+|\d+) \s+ Height (\d+)m \s+ ([a-zA-Z0-9- ]+)')
        for f in files:
            sector, height, var_name = re.findall(pattern, f)[0]
            # print(sector, height, var_name)
            name = var_name_dict.get(var_name, var_name)
            file_height_dict[height].append((f, sector, name))

        elev_avail = False
        first = True
        for height, files_subset in file_height_dict.items():

            first_at_height = True
            for file, sector, var_name in files_subset:
                xarr, yarr, values = _read_grd(file)

                if sector == 'All':

                    # Only 'All' sector has the elevation files.
                    # So here we make sure that, when the elevation file
                    # is read, it gets the right (x,y) coords/dims.
                    if var_name == 'elev':
                        elev_avail = True
                        elev_vals = values
                        elev_coords = {'x': xarr,
                                       'y': yarr}
                        elev_dims = ('x', 'y')
                        continue

                    else:
                        var_name += '_tot'

                        coords = {'x': xarr,
                                  'y': yarr,
                                  'z': np.array([float(height)])}

                        dims = ('x', 'y', 'z')

                        da = xr.DataArray(values[..., np.newaxis],
                                          coords=coords,
                                          dims=dims)

                else:

                    coords = {'x': xarr,
                              'y': yarr,
                              'z': np.array([float(height)]),
                              'sec': np.array([int(sector)])}

                    dims = ('x', 'y', 'z', 'sec')

                    da = xr.DataArray(values[..., np.newaxis, np.newaxis],
                                      coords=coords,
                                      dims=dims)

                if first_at_height:
                    ds_tmp = xr.Dataset({var_name: da})
                    first_at_height = False
                else:
                    ds_tmp = xr.merge([ds_tmp, xr.Dataset({var_name: da})])

            if first:
                ds = ds_tmp
                first = False
            else:
                ds = xr.concat([ds, ds_tmp], dim='z')

        if elev_avail:
            ds['elev'] = xr.DataArray(elev_vals,
                                      coords=elev_coords,
                                      dims=elev_dims)
        ############
        # Calculate the compund speed-up factor based on orog_spd, rgh_spd
        # and obst_spd
        spd = 1
        for dr in ds.data_vars:
            if dr in ['orog_spd', 'obst_spd', 'rgh_spd']:
                # spd *= np.where(ds.data_vars[dr].data > 1e20, 1, ds.data_vars[dr].data)
                spd *= np.where(np.isnan(ds.data_vars[dr].data), 1, ds.data_vars[dr].data)

        ds['spd'] = copy.deepcopy(ds['orog_spd'])
        ds['spd'].data = spd

        #############
        # change the frequency from per sector to per deg
        ds['f'].data = ds['f'].data * len(ds['f']['sec'].data) / 360.0

#         #############
#         # note WAsP denotes unavailable values using very large numbers, here
#         # we change them into np.nan, to avoid strange results.
#         for var in ds.data_vars:
#             ds[var].data = np.where(ds[var].data > 1e20, np.nan, ds[var].data)

        # make sure coords along z is asending order
        ds = ds.loc[{'z': sorted(ds.coords['z'].values)}]

        ######################################################################
        # Adding wind shear coefficient based on speed-up factor at different
        # height and a reference far field wind shear coefficient (alpha_far)
        def _add_wind_shear(ds, alpha_far=0.143):
            ds['wind_shear'] = copy.deepcopy(ds['spd'])

            heights = ds['wind_shear'].coords['z'].data

            # if there is only one layer, assign default value
            if len(heights) == 1:

                ds['wind_shear'].data = (np.zeros_like(ds['wind_shear'].data) +
                                         alpha_far)

                print('Note there is only one layer of wind resource data, ' +
                      'wind shear are assumed as uniform, i.e., {0}'.format(
                          alpha_far))
            else:
                ds['wind_shear'].data[:, :, 0, :] = (alpha_far +
                                                     np.log(ds['spd'].data[:, :, 0, :] / ds['spd'].data[:, :, 1, :]) /
                                                     np.log(heights[0] / heights[1]))

                for h in range(1, len(heights)):
                    ds['wind_shear'].data[:, :, h, :] = (
                        alpha_far +
                        np.log(ds['spd'].data[:, :, h, :] / ds['spd'].data[:, :, h - 1, :]) /
                        np.log(heights[h] / heights[h - 1]))

            return ds

        ds = _add_wind_shear(ds)

        return cls(ds)


class EqDistRegGrid2DInterpolator():
    def __init__(self, x, y, Z):
        self.x = x
        self.y = y
        self.Z = Z
        self.dx, self.dy = [xy[1] - xy[0] for xy in [x, y]]
        self.x0 = x[0]
        self.y0 = y[0]

    def __call__(self, x, y, mode='valid'):
        xp, yp = x, y

        xi = (xp - self.x0) / self.dx
        xif, xi0 = np.modf(xi)
        xi0 = xi0.astype(np.int)
        xi1 = xi0 + 1

        yi = (yp - self.y0) / self.dy
        yif, yi0 = np.modf(yi)
        yi0 = yi0.astype(np.int)
        yi1 = yi0 + 1

        valid = (xif >= 0) & (yif >= 0) & (xi1 < len(self.x)) & (yi1 < len(self.y))
        z = np.empty_like(xp) + np.nan
        xi0, xi1, xif, yi0, yi1, yif = [v[valid] for v in [xi0, xi1, xif, yi0, yi1, yif]]
        z00 = self.Z[xi0, yi0]
        z10 = self.Z[xi1, yi0]
        z01 = self.Z[xi0, yi1]
        z11 = self.Z[xi1, yi1]
        z0 = z00 + (z10 - z00) * xif
        z1 = z01 + (z11 - z01) * xif
        z[valid] = z0 + (z1 - z0) * yif
        if mode == 'extrapolate':
            valid = valid & ~np.isnan(z)
            if (valid[0] == False) | (valid[-1] == False):  # noqa
                nonnan_index = np.where(~np.isnan(z))[0]
                if valid[0] == False:  # noqa
                    first_valid = nonnan_index[0]
                    z[:first_valid] = z[first_valid]
                if valid[-1] == False:  # noqa
                    last_valid = nonnan_index[-1]
                    z[last_valid + 1:] = z[last_valid]
        return z


def main():
    from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
    import matplotlib.pyplot as plt
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=False)
    x, y = site._ds.coords['x'].data, site._ds.coords['y'].data,
    Y, X = np.meshgrid(y, x)
    Z = site._ds['elev'].data
    if 1:
        Z = site.elevation(X.flatten(), Y.flatten()).reshape(X.shape)
        c = plt.contourf(X, Y, Z, 100)
        plt.colorbar(c)
        i, j = 15, 15
        plt.plot(X[i], Y[i], 'b')

    plt.plot(X[:, j], Y[:, j], 'r')
    plt.axis('equal')
    plt.figure()
    Z = site.elevation_interpolator(X[:, j], Y[:, j], mode='extrapolate')
    plt.plot(X[:, j], Z, 'r')

    plt.figure()
    Z = site.elevation_interpolator(X[i], Y[i], mode='extrapolate')
    plt.plot(Y[i], Z, 'b')

    z = np.arange(35, 200, 1)
    u_z = site.local_wind([x[i]] * len(z), y_i=[y[i]] * len(z),
                          h_i=z, wd=[0], ws=[10])[1][:, 0, 0]
    plt.figure()
    Z = site.elevation_interpolator(X[i], Y[i], mode='extrapolate')
    plt.plot(Y[i], Z, 'b')
    plt.figure()
    plt.plot(u_z, z)
    plt.show()


if __name__ == '__main__':
    main()
