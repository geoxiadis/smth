import numpy as np
import pandas as pd
import pytest
from pvlib.solarposition import get_solarposition

from py_wake.shadow_models.solar_position import solar_position


def _solar_position_loop(dates, latitudes, longitudes, method="nrel_numpy"):
    vectors = []
    coordinates = []
    for latitude, longitude in zip(latitudes, longitudes):
        solar = get_solarposition(
            dates, latitude, longitude, method=method)
        elevation_name = (
            "apparent_elevation"
            if "apparent_elevation" in solar
            else "elevation"
        )
        elevation = np.radians(solar[elevation_name].to_numpy(dtype=float))
        azimuth = np.radians(solar["azimuth"].to_numpy(dtype=float))
        vectors.append(
            np.column_stack((
                np.cos(elevation) * np.sin(azimuth),
                np.cos(elevation) * np.cos(azimuth),
                np.sin(elevation),
            ))
        )
        coordinates.append(np.column_stack((azimuth, elevation)))
    return np.asarray(vectors), np.asarray(coordinates)


def test_solar_position_basic():
    """Test basic solar position calculation"""
    latitude, longitude = 55.0, 12.0
    time_index = pd.DatetimeIndex(['2024-06-21 12:00:00'])
    sun_vectors, solar_elevation = solar_position(
        time_index, latitude, longitude, method="nrel_numpy")

    assert sun_vectors.shape == (1, 1, 3)
    assert solar_elevation.shape == (1, 1, 2)
    assert np.all(np.abs(sun_vectors) <= 1)
    assert np.all((solar_elevation >= -90) & (solar_elevation <= 90))


def test_solar_position_multiple_times():
    """Test solar position for multiple timestamps"""
    latitude, longitude = 55.0, 12.0
    times = pd.date_range(
        '2024-06-21 00:00:00', '2024-06-21 23:59:59', freq='1h', tz='Europe/Copenhagen')
    sun_vectors, solar_elevation = solar_position(
        times, latitude, longitude, method="nrel_numpy")

    assert sun_vectors.shape == (1, 24, 3)
    assert solar_elevation.shape == (1, 24, 2)


def test_solar_position_array_inputs():
    """Test solar position with array inputs for latitude/longitude"""
    latitudes = np.array([55.0, 56.0])
    longitudes = np.array([12.0, 13.0])
    time = pd.Timestamp('2024-06-21 12:00:00', tz='UTC')

    sun_vectors, solar_elevation = solar_position(
        time, latitudes, longitudes, method="nrel_numpy")

    assert sun_vectors.shape == (2, 1, 3)
    assert solar_elevation.shape == (2, 1, 2)


def test_solar_position_matches_per_location_loop():
    times = pd.date_range(
        "2024-06-21 00:00:00", periods=6, freq="4h", tz="UTC")
    latitudes = np.array([55.0, 56.0, 57.0])
    longitudes = np.array([12.0, 13.0, 14.0])

    vectors, coords = solar_position(
        times, latitudes, longitudes, method="nrel_numpy")
    loop_vectors, loop_coords = _solar_position_loop(
        times, latitudes, longitudes)

    np.testing.assert_allclose(vectors, loop_vectors, rtol=0, atol=0)
    np.testing.assert_allclose(coords, loop_coords, rtol=0, atol=0)


def test_solar_position_method_selection():
    time = pd.DatetimeIndex(['2024-06-21 12:00:00'], tz='UTC')
    vectors_numpy, coords_numpy = solar_position(
        time, 55.0, 12.0, method="nrel_numpy")
    vectors_ephemeris, coords_ephemeris = solar_position(
        time, 55.0, 12.0, method="ephemeris")

    assert vectors_numpy.shape == vectors_ephemeris.shape == (1, 1, 3)
    assert coords_numpy.shape == coords_ephemeris.shape == (1, 1, 2)
    assert np.all(np.isfinite(vectors_numpy))
    assert np.all(np.isfinite(vectors_ephemeris))


def test_solar_position_edge_cases():
    """Test solar position at edge cases"""
    times = pd.DatetimeIndex(['2024-06-21 12:00:00'], tz='UTC')

    sun_vectors_north, solar_elevation_north = solar_position(
        times, 90.0, 0.0, method="nrel_numpy")
    sun_vectors_south, solar_elevation_south = solar_position(
        times, -90.0, 0.0, method="nrel_numpy")

    sun_vectors_east, solar_elevation_east = solar_position(
        times, 0.0, 180.0, method="nrel_numpy")
    sun_vectors_west, solar_elevation_west = solar_position(
        times, 0.0, -180.0, method="nrel_numpy")

    assert np.all(np.isfinite(sun_vectors_north))
    assert np.all(np.isfinite(sun_vectors_south))
    assert np.all(np.isfinite(sun_vectors_east))
    assert np.all(np.isfinite(sun_vectors_west))
    np.testing.assert_allclose(np.linalg.norm(
        sun_vectors_north, axis=-1), 1, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(
        sun_vectors_south, axis=-1), 1, rtol=1e-12)
    assert solar_elevation_north[0, 0, 1] > solar_elevation_south[0, 0, 1]
    np.testing.assert_allclose(sun_vectors_east, sun_vectors_west, atol=1e-10)


def test_solar_position_input_validation():
    """Test input validation and error handling"""
    latitude, longitude = 55.0, 12.0

    with pytest.raises(ValueError):
        solar_position(pd.Timestamp(
            '2024-06-21 12:00:00', tz='UTC'), [55.0, 55.1], longitude)

    with pytest.raises(ValueError):
        solar_position(pd.Timestamp(
            '2024-06-21 12:00:00', tz='UTC'), 91.0, longitude)

    with pytest.raises(ValueError):
        solar_position(pd.Timestamp(
            '2024-06-21 12:00:00', tz='UTC'), latitude, 181.0)
