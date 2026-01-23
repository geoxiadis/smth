import os
import numpy as np
from numpy import newaxis as na
import xarray as xr
from scipy.interpolate import griddata, RegularGridInterpolator
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtXr
from py_wake.utils.grid_interpolator import GridInterpolator


def sph2cart(lat_rad, lon_rad):
    c = np.cos(lat_rad)
    return np.stack((c * np.cos(lon_rad), c * np.sin(lon_rad), np.sin(lat_rad)), -1)

def sound_power_level_interp(ds=None, ws=None, mode=None, freq=None, yaw=None, **_):
        """
        Docstring for sound_power_level
        
        :param ws: Description
        :param mode: Description
        :param freq: Description
        :param yaw: Description
        :param _: Description
        """
                    
        ws = np.atleast_1d(ds.ws.values if ws is None else ws)
        mode = np.atleast_1d(ds.mode.values if mode is None else 
                             np.zeros_like(ws) + mode)
        freqs = np.atleast_1d(ds.freq.values if freq is None else freq)
        yaws = np.atleast_1d(ds.yaw.values if yaw is None else yaw)
    
        mode_grid, freq_grid, ws_grid, yaw_grid = np.meshgrid(
            mode, freqs, ws, yaws, indexing="ij"
        )
    
        points = np.stack(
            [mode_grid.ravel(), freq_grid.ravel(), ws_grid.ravel(), yaw_grid.ravel()],
            axis=-1,
        )
        
        # TODO:--------remove yaw in Power from data set? and remove this lines--------
        if "yaw" in ds["P"].dims:
            ds2 = ds.copy()
            ds2 = ds2.assign(
                              power=ds["P"].isel(yaw=1),
                              ct=ds["Ct"].isel(yaw=1),
                            )
        else:
            ds2 = ds.copy()    
            ds2 = ds2.assign(
                              power=ds["P"],
                              ct=ds["Ct"],
                            )        
        
        ip = GridInterpolator(
            [ds.mode.values, ds.freq.values, ds.ws.values, ds.yaw.values],
            ds.swl.values,
            method=["linear", "linear", "linear", "linear"]
        )

        swl = ip(points).reshape(len(mode), len(freqs), len(ws), len(yaws))
        
        # print("In sound_power_level:")
        # print("mode:", mode)
        # print("freqs:", freqs)
        # print("ws:", ws)
        # print("yaws:", yaws)
        # print("swl shape:", swl.shape)
        # exit()
        # which wind speeds and yaw to consider? why is frederik considering only first ws?
        return freqs, swl       

def fast_spherical_interpolate_directivity(
    ds,
    mode,
    freq,
    ws,
    yaw,
    lat_deg,
    lon_deg,
    max_dist=np.pi / 2,
    power=3.0,
):
    freq = np.atleast_1d(freq)
    ws = np.atleast_1d(ws)
    yaw = np.atleast_1d(yaw)
    mode = np.atleast_1d(mode)
    lat_deg = np.atleast_1d(lat_deg)
    lon_deg = np.atleast_1d(lon_deg)
    ws, yaw, mode = np.broadcast_arrays(ws, yaw, mode)
    n_cases = ws.size
    n_obs = lat_deg.size
    n_freq = freq.size
    n_data_obs = ds.sizes["obs"]
    
    # cartesian vectors of all the observer positions
    obs_xyz_data = sph2cart(
        np.radians(ds["lat"].values), np.radians(ds["lon"].values)
    )  # (D,3)
    
    # provides the vectors in cartesian coordiantes to the target
    obs_xyz_target = sph2cart(np.radians(lat_deg), np.radians(lon_deg))  # (O,3)
    
    # dot product of the target with all observers, we get the angle between
    # the two directions signifying how aligned the two directions are
    dist = np.arccos(np.clip(obs_xyz_target @ obs_xyz_data.T, -1.0, 1.0))  # (O,D)
    
    # TODO: what is happening here?
    # doing this to get the relevant observers to our target
    # are these a weighing function? the further a meas. observer is from  
    # the target observer, i.e. the larger the angle and smaller
    # the weight
    w = np.where(dist < max_dist, 1.0 / (dist + 1e-10) ** power, 0.0)
    w_sum = w.sum(axis=1, keepdims=True)
    w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum != 0)  # (O,D)
    
    # build the grid interpolator
    delta_swl_interpolator = GridInterpolator(
        [
            ds["mode"].values,
            ds["freq"].values,
            np.arange(n_data_obs),
            ds["ws"].values,
            ds["yaw"].values,
        ],
        ds["delta_swl"].values,
        # method="linear",
        method=["linear", "linear", "linear", "linear", "linear"],
    )
    
    # our relevant meas. observers are those that are < 90 degrees angle
    # from the target
    relevant_obs_mask = (dist < max_dist).any(axis=0)  # Shape: (n_data_obs,)
    relevant_obs_indices = np.where(relevant_obs_mask)[0]
    n_relevant = len(relevant_obs_indices)
    if n_relevant == 0:
        return np.zeros((n_cases, n_obs, 1, 1, n_freq))
    
    # get the weights for the relevant observers
    w_relevant = w[:, relevant_obs_indices]  # Shape: (n_obs, n_relevant)
    case_grid, freq_grid, obs_grid = np.meshgrid(
        np.arange(n_cases), freq, relevant_obs_indices, indexing="ij"
    )
            
    pts = np.column_stack(
        [
            mode[case_grid.ravel()],
            freq_grid.ravel(),
            obs_grid.ravel(),
            ws[case_grid.ravel()],
            yaw[case_grid.ravel()],
        ]
    )
    interp_result = delta_swl_interpolator(pts)
    interp_vals = interp_result.reshape(n_cases, n_freq, n_relevant)
    result = np.einsum(
        "cfr,or->cof", interp_vals, w_relevant
    )  # Apply weights: (cases, freq, relevant) @ (obs, relevant).T -> (cases, obs, freq)
    
    print("spherical_interpolate_delta_swl result shape:")
    print(result.shape)
    print(f"result[:, :, None, None, :]  = {result[:, :, None, None, :].shape}")
    # exit()
    return result[:, :, None, None, :]  # (n_cases, n_obs, 1, 1, n_freq)

def interpolate_directivity(ds, mode, freq, ws, yaw, lat_deg, lon_deg):
    """
    Docstring for interpolate_directivity
    
    :param ds: Description
    :param mode: Description
    :param freq: Description
    :param ws: Description
    :param yaw: Description
    :param lat_deg: Description
    :param lon_deg: Description
    """
    #----------step 1: ------------
    # hawc2 directivity data from netcdf
    # radians to degrees of observed lat. and lon.
    obs_lat_deg = np.rad2deg(ds["lat"].values)
    obs_lon_deg = np.rad2deg(ds["lon"].values)

    # observed direcitivity values
    # Shape: (mode, freq, obs, ws, yaw)
    obs_vals = ds["delta_swl"].values

    # we reshape the values to (obs, mode, freq, ws, yaw)
    # for vectorized interpolation
    obs_vals_transposed = obs_vals.transpose(2, 0, 1, 3 ,4)

    # Flatten all non-spatial dimensions into one "feature vector" 
    # per observer. We do this by stacking the mode, freq, ws, yaw 
    # dimensions into a single dimension.
    n_obs = obs_vals_transposed.shape[0]
    other_dims = obs_vals_transposed.shape[1:] # (mode, freq, ws, yaw)
    n_features = np.prod(other_dims)
    obs_vals_flat = obs_vals_transposed.reshape(n_obs, n_features)

    # ----------step 2: ------------
    # 4 deg resolution
    n_grid_lat = 45
    n_grid_lon = 90
    # 10 deg resolution
    # n_grid_lat = 19
    # n_grid_lon = 36
        
    # lat/lon gridding spaces for smooth interpolation
    grid_lat_ax = np.linspace(-90, 90, n_grid_lat)
    grid_lon_ax = np.linspace(-180, 180, n_grid_lon)
        
    # generate the latitude and longitude regular grid
    gridded_lat, gridded_lon = np.meshgrid(grid_lat_ax, grid_lon_ax, indexing='ij')
        
    print(f" {n_obs} data points --> {n_grid_lat*n_grid_lon} regular grid")

    # interpolate the values on the regular grid
    grid_obs_vals_flat = griddata(points=(obs_lat_deg, obs_lon_deg),
                                  values=obs_vals_flat,
                                  xi=(gridded_lat, gridded_lon),
                                  method='linear',
                                  fill_value=0
                                  )
    # unflatten the gridded observed values back to original shape
    # shape = (n_grid_lat, n_grid_lon) + other_dims
    # Reshape back to (lat, lon, mode, freq, ws, yaw)
    shape = (n_grid_lat, n_grid_lon) + other_dims
    grid_obs_vals_ND = grid_obs_vals_flat.reshape(shape)

    # build the grid interpolator
    # Dim 0: Lat
    # Dim 1: Lon
    # Dim 2: Mode
    # Dim 3: Freq
    # Dim 4: WS
    # Dim 5: Yaw
    coords = [
        grid_lat_ax,    
        grid_lon_ax,
        ds['mode'].values,   
        ds['freq'].values,   
        ds['ws'].values,     
        ds['yaw'].values     
    ]
    # we have a interpolator mapped on a regular lat/lon grid
    interp = RegularGridInterpolator(coords,
                             grid_obs_vals_ND,
                             method='linear',
                             bounds_error=False, 
                             fill_value=0
                             )

    # ----------step 3: Query points-----------
    freq_inp = np.atleast_1d(freq)
    ws_inp = np.atleast_1d(ws)
    yaw_inp = np.atleast_1d(yaw)
    mode_inp = np.atleast_1d(mode)
    lat_inp = np.atleast_1d(lat_deg)
    lon_inp = np.atleast_1d(lon_deg)
    # Using numpy new axis to create empty dimensions for broadcasting
    # Latitutde and Longitude are coupled (dimension: 0)
    lat_b = lat_inp[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    lon_b = lon_inp[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    mode_b = mode_inp[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    freq_b = freq_inp[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    ws_b = ws_inp[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    yaw_b = yaw_inp[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    
    query_points = np.stack(np.broadcast_arrays(lat_b, lon_b, mode_b,
                                                freq_b, ws_b, yaw_b),
                                                axis=-1)
    n_query_points = np.prod(query_points.shape[:-1])
    # print(f"Total query points: {n_query_points}")  
    # print(f"query_points.shape: {query_points.shape}")
    result = interp(query_points)
        
    # print(f"result.shape = {result.shape}")
    result = result.transpose(3, 0, 1, 4, 2)
    result = result[:, :, 0, 0, :]

    # print(f"result.shape = {result.shape}")
    # print(f"result[:, :, None, None, :]  = {result[:, :, None, None, :].shape}")

    return result[:, :, None, None, :]

def ground_eff(freqs, freqs_init, ground_distance_ijlk, src_h, rec_h, ground_type):
    """_summary_
    Parameters
    ----------
    ground_distance_ijlk : _type_
        _description_
    distance_ijlk : _type_
        _description_
    ground_type : _type_
        _description_
    Returns
    -------
    _type_
        _description_
    """
    # freqs_init = self.freqs_init
    # Ground effects ISO
    src_h_ilk = np.expand_dims(src_h, tuple(range(len(np.shape(src_h)), 3))) + .0
    src_h_ijlk = src_h_ilk[:, na]
    rec_h_ijlk = rec_h[na, :, na, na]
    hei_check_ijlk = 30.0 * (src_h_ijlk + rec_h_ijlk)
    q_ijlk = np.where(ground_distance_ijlk <= hei_check_ijlk, 0, 1 - hei_check_ijlk / ground_distance_ijlk)
    Gs = Gr = Gm = ground_type
    # for f=63Hz Am =-3*q, for other frequencies Am = -3 * q * (1-Gm)
    fak = np.where(freqs_init == 63.0, 1, (1 - Gm))
    Am_fijlk = -3.0 * q_ijlk[na] * fak[:, na, na, na, na]
    # 125 Hz
    a_source_ijlk = 1.5 + 3.0 * np.exp(-0.12 * (src_h_ijlk - 5.0)**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0)) + \
        5.7 * np.exp(-0.09 * src_h_ijlk**2) * \
        (1.0 - np.exp(-2.8 * (10.0**(-6.0)) * (ground_distance_ijlk**2)))
    a_rec_ijlk = 1.5 + 3.0 * np.exp(-0.12 * (rec_h_ijlk - 5.0)**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0)) + \
        5.7 * np.exp(-0.09 * rec_h_ijlk**2) * (1.0 - np.exp(-2.8 * (10.0**(-6.0)) * (ground_distance_ijlk**2)))
    # 250 Hz
    b_source_ijlk = 1.5 + 8.6 * np.exp(-0.09 * src_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
    b_rec_ijlk = 1.5 + 8.6 * np.exp(-0.09 * rec_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
    # 500 Hz
    c_source_ijlk = 1.5 + 14.0 * np.exp(-0.46 * src_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
    c_rec_ijlk = 1.5 + 14.0 * np.exp(-0.46 * rec_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
    # 1000 Hz
    d_source_ijlk = 1.5 + 5.0 * np.exp(-0.9 * src_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
    d_rec_ijlk = 1.5 + 5.0 * np.exp(-0.9 * rec_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
    zeros = np.zeros_like(ground_distance_ijlk)
    As_fijlk = np.array([zeros - 1.5,  # 63Hz
                         -1.5 + Gs * a_source_ijlk,  # 125Hz
                         -1.5 + Gs * b_source_ijlk,  # 250Hz
                         -1.5 + Gs * c_source_ijlk,  # 500Hz
                         -1.5 + Gs * d_source_ijlk,  # 1000Hz
                         zeros - 1.5 * (1.0 - Gs),  # 2000Hz
                         zeros - 1.5 * (1.0 - Gs),  # 4000Hz
                         zeros - 1.5 * (1.0 - Gs)  # 8000Hz
                         ])
    Ar_fijlk = np.array([zeros - 1.5,  # 63 Hz
                         -1.5 + Gr * a_rec_ijlk,  # 125 Hz
                         -1.5 + Gr * b_rec_ijlk,  # 250 Hz
                         -1.5 + Gr * c_rec_ijlk,  # 500 Hz
                         -1.5 + Gr * d_rec_ijlk,  # 1000 Hz
                         zeros - 1.5 * (1.0 - Gr),  # 2000 Hz
                         zeros - 1.5 * (1.0 - Gr),  # 4000 Hz
                         zeros - 1.5 * (1.0 - Gr)  # 8000 Hz
                         ])
    # sum up As+Ar+Am
    Agr_fijlk = np.sum([As_fijlk, Ar_fijlk, Am_fijlk], 0)
    # Interpolation to other frequencies are actually not following the standard completely.
    ip = GridInterpolator([freqs_init], Agr_fijlk)
    Agr_ijlkf = np.moveaxis(ip(np.array([freqs]).T), 0, -1)
    # # TODO: move geometric spreading out in to transmission loss! Should be -ve
    # # Area of sphere: 4pi R^2
    # # 10*log_10(4pi) ~ 11
    # # 10 * log_10(R^2) = 20 * log_10(R) =
    # # reference area = 1.0 m^2
    # Adiv_ijlk = 20.0 * np.log10(distance_ijlk / 1.0) + 11.0  # The geometrical spreading (divergence)
    # ISO_ground_ijlkf = - Adiv_ijlk[:, :, :, :, na] - Agr_ijlkf
    # return ISO_ground_ijlkf
    return Agr_ijlkf

def atmab_alpha(freqs, T0, RH0, p_s):
    """ Atmospheric absorption
    Source: H. E. Bass, L. C. Sutherland, A. J. Zuckerwar, D. T. Blackstock, D. M. Hester;
    Atmospheric absorption of sound: Further developments. J. Acoust. Soc. Am. 1 January 1995; 97 (1): 680–683. https://doi.org/10.1121/1.412989
    Note that an erratum was published to correct eq. 3
    # inputs
    distance_ij : distance between source and observer in meter
    T0 : atmospheric temperature in deg C
    RH0 : relative humidity in pct
    p_s : atmospheric pressure in atm
    """
    T_0 = 293.15  # reference temperature in K
    T_01 = 273.16  # triple point isotherm temperature in K
    T = T0 + 273.15  # convert temperature from deg to K
    p_s0 = 1.0  # reference atmospheric pressure in atm
    #
    log10psatps0 = -6.8346 * (T_01 / T)**1.261 + 4.6151
    psatps0 = 10**log10psatps0  # saturation pressure of water vapor normalised by reference atmospheric pressure
    #
    h = RH0 * p_s0 / p_s * psatps0  # absolute humidity in pct
    #
    F = freqs / p_s  # frequency normalised by atmospheric pressure
    F2 = F**2
    #
    # relaxation frequency of oxygen normalised by atmospheric pressure
    F_rO = 1.0 / p_s0 * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    F_rN = 1.0 / p_s0 * (T_0 / T)**(0.5) * (9.0 + 2.8e2 * h *
                                            np.exp(-4.17 * ((T_0 / T)**(1.0 / 3.0) - 1.0)))  # relaxation frequency of nitrogen normalised by atmospheric pressure
    #
    alpha = F2 * (1.84e-11 * (T / T_0)**(0.5) * p_s0 + (T / T_0)**(-5.0 / 2.0) *
                  (1.275e-2 * np.exp(-2239.1 / T) / (F_rO + F2 / F_rO) +
                   0.1068 * np.exp(-3352.0 / T) / (F_rN + F2 / F_rN)))  # sound absorption coefficient in nepers normalised by distance and atmospheric pressure
    #
    alpha_ps = alpha * p_s  # sound absorption coefficient in nepers normalised by distance
    # TODO: remove distance_ij out of the function
    # ISO_alpha = alpha_ps[na, na] * 20.0 / np.log(10.0) * distance_ij[:, :, na]  # sound absorption coefficient in dB
    ISO_alpha = alpha_ps[na, na] * 20.0 / np.log(10.0)

    return ISO_alpha

def iso_transmission_loss(freqs_init, freqs, dx_ijlk, dy_ijlk, dh_ijlk,
                          src_h, rec_h, ground_type, patm, Temp, RHum):
    """
    Docstring for transmission_loss
    
    :param self: Description
    :param freqs_init: Description
    :param freqs: Description
    :param dx_ijlk: Description
    :param dy_ijlk: Description
    :param dh_ijlk: Description
    :param src_h: Description
    :param rec_h: Description
    :param ground_type: Description
    :param patm: Description
    :param Temp: Description
    :param RHum: Description
    """
    # convert atmospheric pressure from Pa to atm
    patm /= 101325.0 
    # transmission loss = ground effects + atmospheric absorption
    # rec_h = np.zeros_like(rec_x) + rec_h
    # dx_ijlk, dy_ijlk, dh_ijlk = self.distance(self.src_x, self.src_y, self.src_h, wd_l=np.array([270]),
    #                                           dst_xyh_jlk=[rec_x, rec_y, rec_h])
    ground_distance_ijlk = np.sqrt(dx_ijlk**2 + dy_ijlk**2)
    distance_ijlk = np.sqrt(dx_ijlk**2 + dy_ijlk**2 + dh_ijlk**2)
    
    # TODO: multiply the distance here and exmpand the dimentsions
    # The atmospheric absorption term
    atm_abs_ijlkf = -1 * atmab_alpha(freqs, T0=Temp, RH0=RHum, p_s=patm) * distance_ijlk[:, :, na]  
    # ground effect term
    ground_eff_ijlkf = -1 * ground_eff(freqs, freqs_init, ground_distance_ijlk, src_h, rec_h, ground_type)
    # # The geometrical spreading (divergence)
    # Area of sphere: 4pi R^2
    # 10*log_10(4pi) ~ 11
    # 10 * log_10(R^2) = 20 * log_10(R) =
    # reference area = 1.0 m^2
    geom_div_ijlk = -1 * (20.0 * np.log10(distance_ijlk / 1.0) + 11.0)  
    #  total transmission loss in sound pressure level
    delta_spl_ijlkf = ground_eff_ijlkf + atm_abs_ijlkf + geom_div_ijlk[:, :, :, :, na]
    
    return delta_spl_ijlkf  # Delta_SPL