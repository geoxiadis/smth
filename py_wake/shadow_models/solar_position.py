from datetime import datetime

import numpy as np
import pandas as pd
from pvlib.solarposition import get_solarposition


VECTORIZED_METHODS = {"nrel_numpy", "ephemeris"}


def _solar_to_arrays(solar, shape):
    elevation_name = (
        "apparent_elevation"
        if "apparent_elevation" in solar
        else "elevation"
    )
    elevation = np.radians(
        solar[elevation_name].to_numpy(dtype=float)).reshape(shape)
    azimuth = np.radians(
        solar["azimuth"].to_numpy(dtype=float)).reshape(shape)

    sun_vectors = np.empty(shape + (3,), dtype=float)
    sun_vectors[..., 0] = np.cos(elevation) * np.sin(azimuth)
    sun_vectors[..., 1] = np.cos(elevation) * np.cos(azimuth)
    sun_vectors[..., 2] = np.sin(elevation)

    celestial_coord = np.empty(shape + (2,), dtype=float)
    celestial_coord[..., 0] = azimuth
    celestial_coord[..., 1] = elevation
    return sun_vectors, celestial_coord


def solar_position(dates, lat, lon, method="nrel_numpy", **kwargs):
    """Calculate sun vectors and azimuth/elevation angles with pvlib.

    Parameters
    ----------
    dates : datetime or array-like
        Timestamps. Naive timestamps are interpreted as UTC.
    lat, lon : float or array-like
        Latitude and longitude in degrees.
    method : str, optional
        pvlib solar position method, by default ``"nrel_numpy"``.
        Methods known to support vector latitude/longitude inputs are called
        once with flattened time/location arrays. Other methods use pvlib's
        scalar-location path.
    **kwargs
        Additional keyword arguments passed to
        ``pvlib.solarposition.get_solarposition``.

    Returns
    -------
    sun_vectors : ndarray
        Unit vectors with shape ``(n_locations, n_times, 3)``.
    celestial_coord : ndarray
        Azimuth and elevation angles in radians with shape
        ``(n_locations, n_times, 2)``.
    """
    if isinstance(dates, (pd.Timestamp, datetime)):
        dates = pd.DatetimeIndex([dates])
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(dates))

    if dates.tz is None:
        dates = dates.tz_localize("UTC")

    lat_array = np.asarray(lat, dtype=float).reshape(-1)
    lon_array = np.asarray(lon, dtype=float).reshape(-1)

    if lat_array.shape != lon_array.shape:
        raise ValueError(
            "Latitude and longitude inputs must have the same number of elements. "
            f"Got shapes: {lat_array.shape} and {lon_array.shape}"
        )

    if np.any((lat_array < -90) | (lat_array > 90)):
        invalid_lats = lat_array[(lat_array < -90) | (lat_array > 90)]
        raise ValueError(
            f"Invalid latitude(s) found outside [-90, 90]: {invalid_lats}"
        )

    if np.any((lon_array < -180) | (lon_array > 180)):
        invalid_lons = lon_array[(lon_array < -180) | (lon_array > 180)]
        raise ValueError(
            f"Invalid longitude(s) found outside [-180, 180]: {invalid_lons}"
        )

    n_locations = lat_array.size
    n_times = len(dates)

    if method not in VECTORIZED_METHODS:
        sun_vectors = np.empty((n_locations, n_times, 3), dtype=float)
        celestial_coord = np.empty((n_locations, n_times, 2), dtype=float)
        for i, (latitude, longitude) in enumerate(zip(lat_array, lon_array)):
            solar = get_solarposition(
                dates, latitude, longitude, method=method, **kwargs)
            sun_vectors[i], celestial_coord[i] = _solar_to_arrays(
                solar, (n_times,))
        return sun_vectors, celestial_coord

    flat_dates = dates.repeat(n_locations)
    flat_lat = np.tile(lat_array, n_times)
    flat_lon = np.tile(lon_array, n_times)
    solar = get_solarposition(
        flat_dates, flat_lat, flat_lon, method=method, **kwargs)
    sun_vectors, celestial_coord = _solar_to_arrays(
        solar, (n_times, n_locations))
    return np.swapaxes(sun_vectors, 0, 1), np.swapaxes(celestial_coord, 0, 1)
