import numpy as np


def to_turbine_direction_vector(wd):
    """Convert PyWake wind-direction angles to turbine direction vectors."""
    wd_rad = np.radians(wd)
    return np.stack(
        (
            -np.sin(wd_rad),
            -np.cos(wd_rad),
            np.zeros_like(wd_rad, dtype=float),
        ),
        axis=-1,
    )


def from_turbine_direction_vector(vectors):
    """Convert turbine direction vectors to PyWake wind-direction angles."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape[-1] < 2:
        raise ValueError("vectors must have at least two components")
    return (np.degrees(np.arctan2(-vectors[..., 0], -vectors[..., 1])) + 360) % 360


def wind_rose(wd, f, n, seed=None, jitter=True):
    """Sample a timestamped wind-direction series from a wind rose."""
    wd = np.asarray(wd, dtype=float).reshape(-1)
    f = np.asarray(f, dtype=float).reshape(-1)

    if wd.size == 0:
        raise ValueError("wd must contain at least one sector")
    if wd.shape != f.shape:
        raise ValueError("wd and f must have the same length")
    if not np.all(np.isfinite(wd)) or not np.all(np.isfinite(f)):
        raise ValueError("wd and f must contain finite values")
    if np.any(f < 0):
        raise ValueError("f must be non-negative")

    total = np.sum(f)
    if total <= 0:
        raise ValueError("f must have a positive sum")

    if wd.size > 1:
        sector_widths = np.diff(np.r_[wd, wd[0] + 360])
        if np.any(sector_widths <= 0):
            raise ValueError("wd sectors must be strictly increasing")
        if not np.allclose(sector_widths, sector_widths[0]):
            raise ValueError("wd sectors must be equally spaced")
        sector_width = sector_widths[0]
    else:
        sector_width = 360.0

    rng = np.random.default_rng(seed)
    sampled = rng.choice(wd, size=int(n), p=f / total)
    if jitter:
        sampled = sampled + rng.uniform(-sector_width / 2, sector_width / 2, int(n))
    return sampled % 360


def sun_facing_wind_direction(sun_vectors):
    """Return wind directions that orient each rotor normal toward the sun."""
    sun_vectors = np.asarray(sun_vectors, dtype=float)
    if sun_vectors.ndim not in (2, 3) or sun_vectors.shape[-1] != 3:
        raise ValueError(
            "sun_vectors must be 2D (n_times, 3) or 3D (n_src, n_times, 3)")

    return from_turbine_direction_vector(sun_vectors)


def _resolve_dict_wind_direction(wind_direction, n_time, n_src, sun_vectors):
    direction_type = wind_direction.get("type")
    if direction_type is None:
        raise ValueError('wind_direction dictionaries must include a "type" key')

    direction_type = str(direction_type).lower()
    if direction_type == "sun_facing":
        if sun_vectors is None:
            raise ValueError("sun_vectors must be supplied for sun_facing wind direction")
        wd = sun_facing_wind_direction(sun_vectors)
        if wd.ndim == 1:
            wd = np.broadcast_to(wd[np.newaxis, :], (n_src, n_time))
        return wd, "sun_facing"

    if direction_type == "wind_rose":
        try:
            wd = wind_direction["wd"]
            f = wind_direction["f"]
        except KeyError as e:
            raise ValueError("wind_rose wind_direction requires wd and f") from e
        sampled = wind_rose(
            wd,
            f,
            n_time,
            seed=wind_direction.get("seed"),
            jitter=wind_direction.get("jitter", True),
        )
        return sampled, "wind_rose"

    raise ValueError(
        'wind_direction type must be "sun_facing" or "wind_rose"')


def resolve_wind_direction(wind_direction, time, n_src, sun_vectors=None):
    """Resolve scalar, series, wind-rose, and sun-facing wind-direction inputs."""
    n_time = len(time)

    if isinstance(wind_direction, str):
        if wind_direction.lower() != "sun_facing":
            raise ValueError('wind_direction string must be "sun_facing"')
        wind_direction = {"type": "sun_facing"}

    if isinstance(wind_direction, dict):
        return _resolve_dict_wind_direction(
            wind_direction, n_time, n_src, sun_vectors)

    wd = np.asarray(wind_direction, dtype=float)
    if wd.ndim == 0:
        return np.full(n_time, float(wd)), "uniform"
    if wd.ndim == 1:
        if wd.size == 1 and n_time != 1:
            return np.full(n_time, float(wd[0]) % 360), "uniform"
        if wd.size != n_time:
            raise ValueError(
                "1D wind_direction must have the same length as time")
        return wd % 360, "time_series"
    if wd.ndim == 2:
        if wd.shape != (n_src, n_time):
            raise ValueError(
                "2D wind_direction must have shape (n_src, n_time)")
        return wd % 360, "per_turbine"

    raise ValueError(
        "wind_direction must be a scalar, 1D time series, 2D per-turbine "
        "time series, 'sun_facing', or a wind_direction strategy dictionary")
