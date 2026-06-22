import itertools
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from tqdm import tqdm

from py_wake.flow_map import Grid, HorizontalGrid, Points
from py_wake.shadow_models.shadow_base import ShadowMap, ShadowResult
from py_wake.shadow_models.solar_position import solar_position
from py_wake.shadow_models.wind_direction import (
    from_turbine_direction_vector,
    resolve_wind_direction,
    to_turbine_direction_vector,
)

from numba import njit, prange


WGS84 = CRS.from_epsg(4326)


class ShadowModel:
    def __init__(
        self,
        src_x,
        src_y,
        src_h,
        turbine_diameter,
        tower_diameter=None,
        time=None,
        wind_direction=None,
        rotor_offset=None,
        src_crs="EPSG:4326",
        calculation_crs="auto",
        solar_position_method="nrel_numpy",
        solar_position_kwargs=None,
        elevation_function=lambda x, y: np.zeros(np.shape(x)),
        min_sun_elevation=0.0,
        max_distance=None,
    ):
        """
        Model for calculating shadow flicker from wind turbines.

        Calculates rotor and tower shadows based on turbine positions,
        sun position and wind direction.

        Parameters
        ----------
        src_x : array-like
            X coordinates of turbines
        src_y : array-like
            Y coordinates of turbines
        src_h : array-like
            Hub heights of turbines
        src_crs : str
            Coordinate reference system of turbine positions
        calculation_crs : str or pyproj.CRS or "auto", optional
            Projected CRS used for shadow geometry. If "auto", a local
            azimuthal equidistant CRS centered on the source/receptor
            coordinates is used.
        solar_position_method : str, optional
            pvlib solar position method. Default is "nrel_numpy".
        solar_position_kwargs : dict or None, optional
            Additional keyword arguments passed to pvlib.
        turbine_diameter : float or array-like
            Rotor diameter(s) of turbines
        tower_diameter : float or array-like or None
            Tower diameter(s) of turbines. Required for tower shadows.
        time : DatetimeIndex or array-like
            Timestamps for shadow calculation
        wind_direction : float, array-like, "sun_facing", or dict
            Wind direction strategy. Scalars are uniform directions. 1D arrays
            are time series. 2D arrays are per-turbine time series with shape
            (n_src, n_time). Use "sun_facing" for per-turbine sun-facing
            directions, or {"type": "wind_rose", "wd": wd, "f": f} to sample
            a wind rose.
        rotor_offset : float or array-like or None, optional
            Distance between tower center and rotor center in meters.
            If None, no rotor offset is applied.
        elevation_function : callable, optional
            Function to calculate terrain elevation, by default returns zero
            elevation
        min_sun_elevation : float, optional
            Minimum sun elevation angle (in degrees) required for shadow calculation.
            Sun positions below this elevation will be ignored. Default is 0.0
            (horizon).
        max_distance : float or None, optional
            Maximum distance (in meters) from turbine center for shadow calculation.
            Receptors beyond this distance will not have shadows calculated.
            If None, all receptors will be processed. Default is None.
        """
        self.elevation_function = elevation_function
        self.min_sun_elevation = np.radians(min_sun_elevation)
        self.max_distance = max_distance
        self.src_x = np.atleast_1d(np.asarray(src_x, dtype=float))
        self.src_y = np.atleast_1d(np.asarray(src_y, dtype=float))
        self.src_crs = CRS.from_user_input(src_crs)
        self.src_crs_input = src_crs
        self.calculation_crs = calculation_crs
        self.solar_position_method = solar_position_method
        self.solar_position_kwargs = dict(solar_position_kwargs or {})

        self.turbine_diameter = np.zeros_like(
            self.src_x, dtype=float) + turbine_diameter
        self.tower_diameter = (
            None if tower_diameter is None
            else np.zeros_like(self.src_x, dtype=float) + tower_diameter
        )
        if rotor_offset is None:
            self.rotor_offset = np.zeros_like(self.src_x, dtype=float)
        else:
            self.rotor_offset = np.zeros_like(
                self.src_x, dtype=float) + rotor_offset

        if time is None:
            raise ValueError("time must be specified")
        if wind_direction is None:
            raise ValueError("wind_direction must be specified")

        self.src_h = np.zeros_like(self.src_x, dtype=float) + src_h
        self.time = pd.DatetimeIndex(time)

        if self.time.tz is None:
            self.time = self.time.tz_localize("UTC")

        if len(self.time) > 1:
            time_diffs = np.diff(self.time)
            # All samples are consistent
            if not np.all(time_diffs == time_diffs[0]):
                raise ValueError(
                    "Inconsistent sampling rate detected in time series")
            try:
                seconds = time_diffs[0].total_seconds()
            except AttributeError:
                seconds = time_diffs[0].astype("timedelta64[s]").astype(float)

            self.freq = 3600 / seconds
        else:
            self.freq = np.inf
        self.wind_direction_input = wind_direction
        delayed_wind_direction = (
            isinstance(wind_direction, str) and
            wind_direction.lower() == "sun_facing"
        ) or (
            isinstance(wind_direction, dict) and
            str(wind_direction.get("type", "")).lower() in ("sun_facing", "wind_rose")
        )
        if delayed_wind_direction:
            self.wind_direction_angles = None
            if isinstance(wind_direction, str):
                self.wind_direction_mode = wind_direction.lower()
            else:
                self.wind_direction_mode = str(wind_direction.get("type")).lower()
            self.wind_dir_rad = None
            self.turbine_dir_vector = None
        else:
            wind_direction_angles, wind_direction_mode = resolve_wind_direction(
                wind_direction, self.time, len(self.src_x))
            self._set_wind_direction(wind_direction_angles, wind_direction_mode)
        self.src_z = elevation_function(self.src_x, self.src_y) + self.src_h
        self.src_lon, self.src_lat = self._transform_xy(
            self.src_x, self.src_y, self.src_crs, WGS84)

        if np.any(self.src_h < self.turbine_diameter / 2):
            raise ValueError(
                "Rotor hub height cannot be less than half the rotor diameter!"
            )

        self.n_src = len(self.src_x)
        self.memory_warning_limit = 2  # GB, maybe should not be hardcoded

    def _set_wind_direction(self, wind_direction_angles, wind_direction_mode):
        self.wind_direction_angles = np.asarray(
            wind_direction_angles, dtype=float)
        self.wind_direction_mode = wind_direction_mode
        self.wind_dir_rad = np.radians(self.wind_direction_angles)
        self.turbine_dir_vector = to_turbine_direction_vector(
            self.wind_direction_angles)

    @staticmethod
    def _transform_xy(x, y, src_crs, dst_crs):
        transformer = Transformer.from_crs(
            src_crs, dst_crs, always_xy=True)
        return transformer.transform(np.asarray(x, dtype=float),
                                     np.asarray(y, dtype=float))

    @staticmethod
    def _lon_centroid(lon):
        lon_rad = np.radians(lon)
        return np.degrees(np.arctan2(np.mean(np.sin(lon_rad)),
                                     np.mean(np.cos(lon_rad))))

    @staticmethod
    def _xy_unit_factor(crs):
        factors = []
        for axis in crs.axis_info[:2]:
            factor = getattr(axis, "unit_conversion_factor", None)
            if factor is None:
                unit_name = getattr(axis, "unit_name", "").lower()
                if unit_name in ("metre", "meter", "m"):
                    factor = 1.0
                else:
                    raise ValueError(
                        "calculation_crs x/y axes must use linear units")
            factors.append(float(factor))
        if len(factors) < 2:
            raise ValueError("calculation_crs must have x/y axes")
        if not np.allclose(factors[0], factors[1]):
            raise ValueError(
                "calculation_crs x/y axes must use the same linear unit")
        return factors[0]

    def _auto_calculation_crs(self, lon, lat):
        lon_0 = self._lon_centroid(np.asarray(lon, dtype=float))
        lat_0 = float(np.mean(np.asarray(lat, dtype=float)))
        return CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat_0:.12f} +lon_0={lon_0:.12f} "
            "+datum=WGS84 +units=m +no_defs"
        )

    def _get_calculation_crs(self, rec_lon=None, rec_lat=None):
        if self.calculation_crs == "auto":
            lon = self.src_lon if rec_lon is None else np.r_[self.src_lon, rec_lon]
            lat = self.src_lat if rec_lat is None else np.r_[self.src_lat, rec_lat]
            crs = self._auto_calculation_crs(lon, lat)
        else:
            crs = CRS.from_user_input(self.calculation_crs)

        if not crs.is_projected:
            raise ValueError("calculation_crs must be a projected CRS")
        unit_factor = self._xy_unit_factor(crs)
        return crs, unit_factor

    def _project_to_calculation_crs(self, rec_x, rec_y, rec_crs):
        rec_crs = CRS.from_user_input(rec_crs)
        rec_lon, rec_lat = self._transform_xy(rec_x, rec_y, rec_crs, WGS84)
        calculation_crs, unit_factor = self._get_calculation_crs(
            rec_lon, rec_lat)

        src_x, src_y = self._transform_xy(
            self.src_x, self.src_y, self.src_crs, calculation_crs)
        rec_x_calc, rec_y_calc = self._transform_xy(
            rec_x, rec_y, rec_crs, calculation_crs)

        return (
            np.asarray(src_x, dtype=float) * unit_factor,
            np.asarray(src_y, dtype=float) * unit_factor,
            np.asarray(rec_x_calc, dtype=float) * unit_factor,
            np.asarray(rec_y_calc, dtype=float) * unit_factor,
            np.asarray(rec_lon, dtype=float),
            np.asarray(rec_lat, dtype=float),
            calculation_crs,
        )

    def _default_source_calculation_xy(self):
        calculation_crs, unit_factor = self._get_calculation_crs()
        src_x, src_y = self._transform_xy(
            self.src_x, self.src_y, self.src_crs, calculation_crs)
        return np.asarray(src_x) * unit_factor, np.asarray(src_y) * unit_factor

    @staticmethod
    def _normalize_mode(mode):
        mode = mode.lower()
        if mode not in ("rotor", "tower", "both", "combined"):
            raise ValueError(
                'mode must be "rotor", "tower", "both", or "combined"')
        return mode

    def _validate_tower_mode(self, mode):
        if mode in ("tower", "both", "combined") and self.tower_diameter is None:
            raise ValueError(
                "tower_diameter is required when calculating tower shadows")

    @staticmethod
    @njit(parallel=True,
          fastmath=True,
          cache=True,
          nogil=True,
          boundscheck=False)
    def _calculate_rotor_shadow_from_coords(
        src_coords, rec_coords, wind_dir_batch, sun_vectors_batch,
        turbine_diameters, rotor_offsets, wind_dir_rad_batch,
        min_sun_elevation=0.0
    ):
        n_src = src_coords.shape[0]
        n_rec = rec_coords.shape[0]
        n_times = min(wind_dir_batch.shape[1], sun_vectors_batch.shape[1])
        n_times = min(n_times, wind_dir_rad_batch.shape[1])

        result = np.zeros((n_src, n_rec, n_times), dtype=np.bool_)
        rotor_radii_sq = (turbine_diameters / 2.0) ** 2

        for t in prange(n_times):
            for s in range(n_src):
                wind_x_t = wind_dir_batch[s, t, 0]
                wind_y_t = wind_dir_batch[s, t, 1]
                wind_z_t = wind_dir_batch[s, t, 2]
                wind_dir_rad_t = wind_dir_rad_batch[s, t]
                sun_x_t = sun_vectors_batch[s, t, 0]
                sun_y_t = sun_vectors_batch[s, t, 1]
                sun_z_t = sun_vectors_batch[s, t, 2]
                if sun_z_t <= min_sun_elevation:
                    continue

                sun_dot_wind = (
                    sun_x_t * wind_x_t +
                    sun_y_t * wind_y_t +
                    sun_z_t * wind_z_t
                )
                if np.abs(sun_dot_wind) < 1e-10:
                    sun_dot_wind = 1e-10 if sun_dot_wind >= 0.0 else -1e-10

                offset_x = rotor_offsets[s] * np.sin(wind_dir_rad_t)
                offset_y = rotor_offsets[s] * np.cos(wind_dir_rad_t)
                rotor_x = src_coords[s, 0] + offset_x
                rotor_y = src_coords[s, 1] + offset_y
                rotor_z = src_coords[s, 2]
                radius_sq = rotor_radii_sq[s]

                for r in range(n_rec):
                    line_x = rec_coords[r, 0] - rotor_x
                    line_y = rec_coords[r, 1] - rotor_y
                    line_z = rec_coords[r, 2] - rotor_z

                    wind_dot_line = (
                        wind_x_t * line_x +
                        wind_y_t * line_y +
                        wind_z_t * line_z
                    )
                    factor = wind_dot_line / sun_dot_wind
                    if factor > 0.0:
                        continue

                    inter_x = line_x - factor * sun_x_t
                    inter_y = line_y - factor * sun_y_t
                    inter_z = line_z - factor * sun_z_t

                    distance_sq = (
                        inter_x * inter_x +
                        inter_y * inter_y +
                        inter_z * inter_z
                    )
                    result[s, r, t] = distance_sq <= radius_sq

        return result

    @staticmethod
    @njit(parallel=True,
          fastmath=True,
          cache=True,
          nogil=True,
          boundscheck=False)
    def _calculate_tower_shadow_from_coords(
        src_coords, rec_coords, sun_vectors_batch, tower_diameters,
        tower_heights, min_sun_elevation=0.0
    ):
        n_src = src_coords.shape[0]
        n_rec = rec_coords.shape[0]
        n_times = sun_vectors_batch.shape[1]

        result = np.zeros((n_src, n_rec, n_times), dtype=np.bool_)

        for t in prange(n_times):
            for s in range(n_src):
                sun_x_t = sun_vectors_batch[s, t, 0]
                sun_y_t = sun_vectors_batch[s, t, 1]
                sun_z_t = sun_vectors_batch[s, t, 2]
                if sun_z_t <= min_sun_elevation:
                    continue

                tower_radius = tower_diameters[s] / 2.0
                radius_sq = tower_radius * tower_radius
                hub_z = src_coords[s, 2]
                base_z = hub_z - tower_heights[s]
                center_x = src_coords[s, 0]
                center_y = src_coords[s, 1]

                for r in range(n_rec):
                    rel_x = rec_coords[r, 0] - center_x
                    rel_y = rec_coords[r, 1] - center_y
                    rec_z = rec_coords[r, 2]

                    if (
                        rel_x * rel_x + rel_y * rel_y <= radius_sq and
                        base_z <= rec_z <= hub_z
                    ):
                        result[s, r, t] = True
                        continue

                    a = sun_x_t * sun_x_t + sun_y_t * sun_y_t
                    if a < 1e-20:
                        continue

                    b = 2.0 * (rel_x * sun_x_t + rel_y * sun_y_t)
                    c = rel_x * rel_x + rel_y * rel_y - radius_sq
                    discriminant = b * b - 4.0 * a * c
                    if discriminant < 0.0:
                        continue

                    sqrt_discriminant = np.sqrt(discriminant)
                    lambda_1 = (-b - sqrt_discriminant) / (2.0 * a)
                    lambda_2 = (-b + sqrt_discriminant) / (2.0 * a)
                    z_1 = rec_z + lambda_1 * sun_z_t
                    z_2 = rec_z + lambda_2 * sun_z_t

                    result[s, r, t] = (
                        (lambda_1 >= 0.0 and base_z <= z_1 <= hub_z) or
                        (lambda_2 >= 0.0 and base_z <= z_2 <= hub_z)
                    )

        return result

    def shadow_tracing(
        self,
        rec_x,
        rec_y,
        rec_z,
        wind_dir,
        sun_vectors,
        mode: str = "both",
        verbose: bool = True,
        time_batch_size: int = 2000,
        receptor_batch_size: int = 10000,
        turbine_batch_size: int = 20,
        src_x=None,
        src_y=None,
        src_z=None,
        wind_dir_rad=None,
    ):
        """
        Calculate shadow presence with efficient multi-dimensional batching.

        Parameters
        ----------
        rec_x, rec_y, rec_z : array-like
            Receptor coordinates
        wind_dir : array-like
            Wind directions for each time step
        sun_vectors : array-like
            Sun vectors for each time step
        mode : str, optional
            'rotor', 'tower', or 'both'
        time_batch_size : int, optional
            Number of time steps to process in each batch
        receptor_batch_size : int, optional
            Number of receptors to process in each batch
        turbine_batch_size : int, optional
            Number of turbines to process in each batch

        Returns
        -------
        rotor_shadow, tower_shadow : tuple of ndarray or None
            Boolean arrays indicating shadow presence
        """

        mode = self._normalize_mode(mode)
        self._validate_tower_mode(mode)

        original_shape = rec_x.shape
        rec_x, rec_y, rec_z = map(np.ravel, (rec_x, rec_y, rec_z))

        if src_x is None or src_y is None:
            src_x, src_y = self._default_source_calculation_xy()
        if src_z is None:
            src_z = self.src_z

        n_turbines = len(src_x)
        sun_vectors = np.asarray(sun_vectors)
        if sun_vectors.shape[-1] != 3:
            raise ValueError(
                "sun_vectors must have last dimension of size 3 (x, y, z)")
        if sun_vectors.ndim == 2:
            # (n_times, 3) -> broadcast to (n_src, n_times, 3)
            n_times = sun_vectors.shape[0]
            sun_vectors = np.broadcast_to(
                sun_vectors[np.newaxis, :, :], (n_turbines, n_times, 3)
            )
        elif sun_vectors.ndim == 3:
            # (n_src, n_times, 3)
            n_times = sun_vectors.shape[1]
        else:
            raise ValueError(
                "sun_vectors must be 2D (n_times,3) or 3D (n_src,n_times,3)")

        wind_dir = np.asarray(wind_dir, dtype=float)
        if wind_dir.shape[-1] != 3:
            raise ValueError(
                "wind_dir must have last dimension of size 3 (x, y, z)")
        if wind_dir.ndim == 2:
            if wind_dir.shape[0] == 1 and n_times != 1:
                wind_dir = np.broadcast_to(wind_dir, (n_times, 3))
            elif wind_dir.shape[0] > n_times:
                wind_dir = wind_dir[:n_times]
            elif wind_dir.shape[0] != n_times:
                raise ValueError("wind_dir time dimension must match sun_vectors")
            wind_dir = np.broadcast_to(
                wind_dir[np.newaxis, :, :], (n_turbines, n_times, 3))
        elif wind_dir.ndim == 3:
            if wind_dir.shape[:2] != (n_turbines, n_times):
                raise ValueError(
                    "3D wind_dir must have shape (n_src, n_times, 3)")
        else:
            raise ValueError(
                "wind_dir must be 2D (n_times,3) or 3D (n_src,n_times,3)")

        if wind_dir_rad is None:
            stored_wind_dir_rad = self.wind_dir_rad
            if stored_wind_dir_rad is not None and (
                np.shape(stored_wind_dir_rad) == (n_times,) or
                np.shape(stored_wind_dir_rad) == (n_turbines, n_times)
            ):
                wind_dir_rad = stored_wind_dir_rad
            else:
                wind_dir_rad = np.radians(
                    from_turbine_direction_vector(wind_dir))
        wind_dir_rad = np.asarray(wind_dir_rad, dtype=float)
        if wind_dir_rad.ndim == 1:
            if wind_dir_rad.shape[0] == 1 and n_times != 1:
                wind_dir_rad = np.full(n_times, wind_dir_rad[0])
            elif wind_dir_rad.shape[0] > n_times:
                wind_dir_rad = wind_dir_rad[:n_times]
            elif wind_dir_rad.shape[0] != n_times:
                raise ValueError(
                    "wind_dir_rad time dimension must match sun_vectors")
            wind_dir_rad = np.broadcast_to(
                wind_dir_rad[np.newaxis, :], (n_turbines, n_times))
        elif wind_dir_rad.ndim == 2:
            if wind_dir_rad.shape != (n_turbines, n_times):
                raise ValueError(
                    "2D wind_dir_rad must have shape (n_src, n_times)")
        else:
            raise ValueError(
                "wind_dir_rad must be 1D (n_times,) or 2D (n_src,n_times)")
        n_receptors = len(rec_x)

        self.result_size_check(n_times, n_turbines, n_receptors, mode)

        rotor_shadow = None
        tower_shadow = None
        if mode in ["rotor", "both", "combined"]:
            rotor_shadow = np.zeros(
                (n_turbines, n_receptors, n_times), dtype=np.bool_)
        if mode in ["tower", "both", "combined"]:
            tower_shadow = np.zeros(
                (n_turbines, n_receptors, n_times), dtype=np.bool_)

        all_rec_coords = np.column_stack((rec_x, rec_y, rec_z))
        all_src_coords = np.column_stack((src_x, src_y, src_z))

        if self.max_distance is not None:
            max_distance_sq = self.max_distance * self.max_distance
            distances_sq = (
                (rec_x[:, None] - src_x[None, :])**2 +
                (rec_y[:, None] - src_y[None, :])**2
            )
            within_distance = np.any(distances_sq <= max_distance_sq, axis=1)
            receptor_mask = within_distance
            distance_mask = distances_sq <= max_distance_sq
            original_indices = np.arange(n_receptors)[receptor_mask]
            filtered_rec_coords = all_rec_coords[receptor_mask]
            filtered_n_receptors = len(filtered_rec_coords)
        else:
            receptor_mask = np.ones(n_receptors, dtype=bool)
            distance_mask = None
            original_indices = np.arange(n_receptors)
            filtered_rec_coords = all_rec_coords
            filtered_n_receptors = n_receptors

        time_batches = [
            (t, min(t + time_batch_size, n_times))
            for t in range(0, n_times, time_batch_size)
        ]
        receptor_batches = [
            (r, min(r + receptor_batch_size, filtered_n_receptors))
            for r in range(0, filtered_n_receptors, receptor_batch_size)
        ]
        turbine_batches = [
            (s, min(s + turbine_batch_size, n_turbines))
            for s in range(0, n_turbines, turbine_batch_size)
        ]

        total_batches = len(time_batches) * \
            len(receptor_batches) * len(turbine_batches)

        with tqdm(
            total=total_batches, desc="Processing shadow batches", disable=not verbose
        ) as pbar:
            for (
                (t_start, t_end),
                (r_start, r_end),
                (s_start, s_end),
            ) in itertools.product(time_batches, receptor_batches, turbine_batches):

                time_slice = slice(t_start, t_end)
                r_slice = slice(r_start, r_end)
                s_slice = slice(s_start, s_end)

                batch_wind_dir = wind_dir[s_slice, time_slice, :]
                # (n_src_batch, n_times_batch, 3)
                batch_sun_vectors_per_src = sun_vectors[s_slice, time_slice, :]
                batch_rec_coords = filtered_rec_coords[r_slice]
                batch_src_coords = all_src_coords[s_slice]
                original_receptor_indices = original_indices[r_slice]
                source_indices = np.arange(s_start, s_end)
                time_indices = np.arange(t_start, t_end)
                if distance_mask is None:
                    pair_distance_mask = None
                else:
                    pair_distance_mask = distance_mask[
                        np.ix_(original_receptor_indices, source_indices)].T

                if mode in ("rotor", "both", "combined") and rotor_shadow is not None:
                    result = self._calculate_rotor_shadow_from_coords(
                        batch_src_coords,
                        batch_rec_coords,
                        batch_wind_dir,
                        batch_sun_vectors_per_src,
                        self.turbine_diameter[s_slice],
                        self.rotor_offset[s_slice],
                        wind_dir_rad[s_slice, time_slice],
                        self.min_sun_elevation,
                    )
                    if pair_distance_mask is not None:
                        result &= pair_distance_mask[:, :, np.newaxis]
                    rotor_shadow[np.ix_(
                        source_indices,
                        original_receptor_indices,
                        time_indices,
                    )] = result

                if mode in ("tower", "both", "combined") and tower_shadow is not None:
                    result_t = self._calculate_tower_shadow_from_coords(
                        batch_src_coords,
                        batch_rec_coords,
                        batch_sun_vectors_per_src,
                        self.tower_diameter[s_slice],
                        self.src_h[s_slice],
                        self.min_sun_elevation,
                    )
                    if pair_distance_mask is not None:
                        result_t &= pair_distance_mask[:, :, np.newaxis]
                    tower_shadow[np.ix_(
                        source_indices,
                        original_receptor_indices,
                        time_indices,
                    )] = result_t

                pbar.update(1)
                desc = (
                    f"Processing: time {t_start + 1}-{t_end}/{n_times}, "
                    f"receptors {r_start + 1}-{r_end}/{filtered_n_receptors}, "
                    f"turbines {s_start + 1}-{s_end}/{n_turbines}"
                )
                pbar.set_description(desc)

        new_shape = (n_turbines,) + original_shape + (n_times,)
        if rotor_shadow is not None:
            rotor_shadow = rotor_shadow.reshape(new_shape)
        if tower_shadow is not None:
            tower_shadow = tower_shadow.reshape(new_shape)

        return rotor_shadow, tower_shadow

    def result_size_check(self, n_times, n_turbines, n_receptors, mode):
        array_size = n_turbines * n_receptors * \
            n_times * np.dtype(np.bool_).itemsize
        n_arrays = 2 if mode in ("both", "combined") else 1
        size_gb = array_size / (1024**3) * n_arrays
        if size_gb > self.memory_warning_limit:  # Warning threshold of 1GB
            warnings.warn(
                f"Large shadow arrays detected! The shadow array will use {
                    size_gb:.2f} GB of memory.",
                RuntimeWarning,
            )

    @staticmethod
    def _add_wind_direction_data_var(data_vars, wind_direction):
        wind_direction = np.asarray(wind_direction, dtype=float)
        if wind_direction.ndim == 1:
            data_vars["wind_direction"] = ("time", wind_direction)
        elif wind_direction.ndim == 2:
            data_vars["wind_direction"] = (
                ("wt", "time"), wind_direction)
        else:
            raise ValueError("Resolved wind_direction must be 1D or 2D")

    def _get_grid(self, grid):
        if grid is None:

            def f(x, N=20, ext=0.1):
                ext *= np.max([1, (np.max(x) - np.min(x))])
                return np.linspace(np.min(x) - ext, np.max(x) + ext, N)

            grid = HorizontalGrid(f(self.src_x), f(self.src_y), h=0)
            return grid(self.src_x, self.src_y, 0) + (None,)
        if isinstance(grid, Grid):
            plane = grid.plane
            grid = grid(x_i=grid.x, y_i=grid.y, h_i=grid.h)
        else:
            raise NotImplementedError(
                "The grid must be instance of Grid or None")
        return grid + (plane,)

    def run(self, grid=None, rec_crs="EPSG:4326", mode="both", verbose=True):
        """
        Calculate shadow time series for a grid or discrete points

        This function determines whether to return a ShadowMap (for grid-based
        calculations) or a ShadowResult (for receptor-based calculations) based on
        the provided grid type.

        Parameters
        ----------
        grid : HorizontalGrid, Points, or None
            - If HorizontalGrid: performs grid-based calculation (returns ShadowMap)
            - If Points: performs receptor-based calculation (returns ShadowResult)
            - If None: creates default HorizontalGrid with h=0 (returns ShadowMap)
        rec_crs : str, default "EPSG:4326"
            Coordinate reference system of coordinates
        mode : str, default "both"
            Shadow type to calculate: "rotor", "tower", or "both"
        verbose : bool, default True
            If True, prints progress information

        Returns
        -------
        ShadowMap or ShadowResult
            Shadow calculation results in appropriate format
        """
        # if grid is None:
        #     grid = HorizontalGrid()

        mode = self._normalize_mode(mode)
        self._validate_tower_mode(mode)

        X, Y, x_j, y_j, h_j, _ = self._get_grid(grid)

        is_points = isinstance(grid, Points)

        rotor_shadow, tower_shadow, run_info = self._call_with_info(
            x_j, y_j, h_j, rec_crs, mode, verbose
        )

        n_turbines = len(self.src_x)

        attrs = {
            "created": np.datetime64("now").astype(str),
            "model": "ShadowModel",
            "tz": str(self.time.tz),
            "freq": self.freq,
            "src_crs": self.src_crs.to_string(),
            "rec_crs": CRS.from_user_input(rec_crs).to_string(),
            "calculation_crs": run_info["calculation_crs"].to_string(),
            "solar_position_method": self.solar_position_method,
            "wind_direction_mode": run_info["wind_direction_mode"],
            "mode": mode,
        }

        if is_points:
            n_receptors = len(x_j)

            data_vars = {
                "rec_x": ("rec", x_j),
                "rec_y": ("rec", y_j),
                "rec_h": ("rec", h_j),
                "src_x": ("wt", self.src_x),
                "src_y": ("wt", self.src_y),
                "src_h": ("wt", self.src_h),
                "turbine_diameter": ("wt", self.turbine_diameter),
                "rec_lon": ("rec", run_info["rec_lon"]),
                "rec_lat": ("rec", run_info["rec_lat"]),
            }
            if self.tower_diameter is not None:
                data_vars["tower_diameter"] = ("wt", self.tower_diameter)
            self._add_wind_direction_data_var(
                data_vars, run_info["wind_direction"])

            coords = {
                "wt": np.arange(n_turbines),
                "rec": np.arange(n_receptors),
                "time": self.time,
            }

            # Determine original shape from grid
            if hasattr(grid, 'x') and hasattr(grid, 'y'):
                if hasattr(grid.x, 'shape'):
                    original_shape = grid.x.shape
                else:
                    original_shape = (len(grid.x),)
            else:
                original_shape = (len(x_j),)

            rec_x_reshaped = np.array(x_j).reshape(original_shape)
            rec_y_reshaped = np.array(y_j).reshape(original_shape)

            if mode == "combined":
                data_vars = {
                    "Combined shadow": (("wt", "rec", "time"), rotor_shadow | tower_shadow),
                    "src_x": ("wt", self.src_x),
                    "src_y": ("wt", self.src_y),
                    "src_h": ("wt", self.src_h),
                    "rec_x": rec_x_reshaped,
                    "rec_y": rec_y_reshaped,
                    "rec_lon": ("rec", run_info["rec_lon"]),
                    "rec_lat": ("rec", run_info["rec_lat"]),
                    "turbine_diameter": ("wt", self.turbine_diameter),
                }
                if self.tower_diameter is not None:
                    data_vars["tower_diameter"] = ("wt", self.tower_diameter)
                self._add_wind_direction_data_var(
                    data_vars, run_info["wind_direction"])
            else:
                data_vars = {
                    "src_x": ("wt", self.src_x),
                    "src_y": ("wt", self.src_y),
                    "src_h": ("wt", self.src_h),
                    "rec_x": rec_x_reshaped,
                    "rec_y": rec_y_reshaped,
                    "rec_lon": ("rec", run_info["rec_lon"]),
                    "rec_lat": ("rec", run_info["rec_lat"]),
                    "turbine_diameter": ("wt", self.turbine_diameter),
                }
                if self.tower_diameter is not None:
                    data_vars["tower_diameter"] = ("wt", self.tower_diameter)
                self._add_wind_direction_data_var(
                    data_vars, run_info["wind_direction"])
                if mode in ["rotor", "both"]:
                    data_vars["Rotor shadow"] = (
                        ("wt", "rec", "time"), rotor_shadow)
                if mode in ["tower", "both"]:
                    data_vars["Tower shadow"] = (
                        ("wt", "rec", "time"), tower_shadow)

            attrs["result_type"] = "ShadowResult"
            dataset = xr.Dataset(data_vars=data_vars,
                                 coords=coords, attrs=attrs)

            return ShadowResult(dataset)
        else:
            data_vars = {
                "src_x": ("wt", self.src_x),
                "src_y": ("wt", self.src_y),
                "src_h": ("wt", self.src_h),
                "turbine_diameter": ("wt", self.turbine_diameter),
            }
            if self.tower_diameter is not None:
                data_vars["tower_diameter"] = ("wt", self.tower_diameter)
            self._add_wind_direction_data_var(
                data_vars, run_info["wind_direction"])

            if mode == "combined":
                combined_shadow = rotor_shadow | tower_shadow
                data_vars["Combined shadow"] = (
                    ("wt", "y", "x", "time"),
                    combined_shadow.reshape(
                        (n_turbines,) + X.shape + (combined_shadow.shape[2],)
                    ),
                )
            else:
                if mode in ["rotor", "both"]:
                    data_vars["Rotor shadow"] = (
                        ("wt", "y", "x", "time"),
                        rotor_shadow.reshape(
                            (n_turbines,) + X.shape + (rotor_shadow.shape[2],)
                        ),
                    )

                if mode in ["tower", "both"]:
                    data_vars["Tower shadow"] = (
                        ("wt", "y", "x", "time"),
                        tower_shadow.reshape(
                            (n_turbines,) + X.shape + (tower_shadow.shape[2],)
                        ),
                    )

            coords = {
                "wt": np.arange(n_turbines),
                "x": X[0],
                "y": Y[:, 0],
                "time": self.time,
            }
            attrs["result_type"] = "ShadowMap"
            dataset = xr.Dataset(data_vars=data_vars,
                                 coords=coords, attrs=attrs)

            return ShadowMap(dataset)

    def __call__(
            self,
            rec_x,
            rec_y,
            rec_h,
            rec_crs,
            mode: str = "both",
            verbose: bool = True):
        rotor_shadow, tower_shadow, _ = self._call_with_info(
            rec_x, rec_y, rec_h, rec_crs, mode, verbose)
        return rotor_shadow, tower_shadow

    def _call_with_info(
            self,
            rec_x,
            rec_y,
            rec_h,
            rec_crs,
            mode: str = "both",
            verbose: bool = True):
        mode = self._normalize_mode(mode)
        self._validate_tower_mode(mode)

        rec_x = np.atleast_1d(rec_x)
        rec_y = np.atleast_1d(rec_y)
        rec_h = np.ones_like(rec_x) * rec_h
        rec_z = self.elevation_function(rec_x, rec_y) + rec_h
        src_x, src_y, rec_calc_x, rec_calc_y, rec_lon, rec_lat, calculation_crs = (
            self._project_to_calculation_crs(rec_x, rec_y, rec_crs)
        )

        sun_vectors, _ = solar_position(
            self.time, self.src_lat, self.src_lon,
            method=self.solar_position_method,
            **self.solar_position_kwargs
        )
        wind_direction_angles, wind_direction_mode = resolve_wind_direction(
            self.wind_direction_input, self.time, self.n_src, sun_vectors)
        self._set_wind_direction(wind_direction_angles, wind_direction_mode)

        rotor_shadow, tower_shadow = self.shadow_tracing(
            rec_calc_x, rec_calc_y, rec_z, self.turbine_dir_vector,
            sun_vectors, mode, verbose, src_x=src_x, src_y=src_y,
            src_z=self.src_z, wind_dir_rad=self.wind_dir_rad)
        return rotor_shadow, tower_shadow, {
            "rec_lon": rec_lon,
            "rec_lat": rec_lat,
            "calculation_crs": calculation_crs,
            "wind_direction": self.wind_direction_angles,
            "wind_direction_mode": self.wind_direction_mode,
        }
