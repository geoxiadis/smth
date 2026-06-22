import numpy as np
import pandas as pd
import pytest

from py_wake.shadow_models.wind_direction import (
    from_turbine_direction_vector,
    resolve_wind_direction,
    sun_facing_wind_direction,
    to_turbine_direction_vector,
    wind_rose,
)


@pytest.fixture
def time_series():
    return pd.date_range("2024-01-01", periods=5, freq="h")


def test_turbine_direction_vector_round_trip():
    wd = np.array([0.0, 90.0, 180.0, 270.0, 45.0])
    vectors = to_turbine_direction_vector(wd)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0],
    ])
    np.testing.assert_allclose(vectors, expected, atol=1e-12)
    np.testing.assert_allclose(from_turbine_direction_vector(vectors), wd)


def test_from_turbine_direction_vector_validation():
    with pytest.raises(ValueError, match="at least two components"):
        from_turbine_direction_vector(np.array([1.0]))


def test_resolve_scalar_uniform(time_series):
    wd, mode = resolve_wind_direction(270, time_series, n_src=2)
    assert mode == "uniform"
    assert wd.shape == (len(time_series),)
    np.testing.assert_allclose(wd, 270)


def test_resolve_length_one_series_as_uniform(time_series):
    wd, mode = resolve_wind_direction([270], time_series, n_src=2)
    assert mode == "uniform"
    np.testing.assert_allclose(wd, 270)


def test_resolve_time_series(time_series):
    expected = np.arange(len(time_series)) * 10.0
    wd, mode = resolve_wind_direction(expected, time_series, n_src=2)
    assert mode == "time_series"
    np.testing.assert_allclose(wd, expected)


def test_resolve_per_turbine_time_series(time_series):
    expected = np.vstack([
        np.arange(len(time_series)),
        np.arange(len(time_series)) + 180,
    ])
    wd, mode = resolve_wind_direction(expected, time_series, n_src=2)
    assert mode == "per_turbine"
    assert wd.shape == (2, len(time_series))
    np.testing.assert_allclose(wd, expected)


def test_wind_rose_reproducible_and_normalizes_frequencies():
    wd = np.array([0.0, 90.0, 180.0, 270.0])
    f = np.array([1.0, 4.0, 1.0, 4.0])
    sampled = wind_rose(wd, f, 20, seed=42)
    sampled_again = wind_rose(wd, f / f.sum(), 20, seed=42)
    np.testing.assert_allclose(sampled, sampled_again)
    assert sampled.shape == (20,)
    assert np.all((sampled >= 0) & (sampled < 360))


def test_wind_rose_without_jitter_returns_sector_centers():
    wd = np.array([0.0, 120.0, 240.0])
    f = np.array([0.5, 0.25, 0.25])
    sampled = wind_rose(wd, f, 20, seed=1, jitter=False)
    assert set(np.unique(sampled)).issubset(set(wd))


@pytest.mark.parametrize(
    "wd,f,match",
    [
        ([0, 90], [1], "same length"),
        ([0, 90], [0, 0], "positive sum"),
        ([0, 90], [1, -1], "non-negative"),
        ([0, 100, 270], [1, 1, 1], "equally spaced"),
        ([0, 270, 180], [1, 1, 1], "strictly increasing"),
    ],
)
def test_wind_rose_validation(wd, f, match):
    with pytest.raises(ValueError, match=match):
        wind_rose(wd, f, 5, seed=1)


def test_wind_rose_additional_validation_and_single_sector():
    with pytest.raises(ValueError, match="at least one sector"):
        wind_rose([], [], 5, seed=1)
    with pytest.raises(ValueError, match="finite"):
        wind_rose([0, np.nan], [1, 1], 5, seed=1)

    sampled = wind_rose([180], [1], 5, seed=1)
    assert sampled.shape == (5,)
    assert np.all((sampled >= 0) & (sampled < 360))


def test_resolve_wind_rose_strategy(time_series):
    wd, mode = resolve_wind_direction(
        {"type": "wind_rose", "wd": [0, 180], "f": [1, 1], "seed": 3},
        time_series,
        n_src=2,
    )
    assert mode == "wind_rose"
    assert wd.shape == (len(time_series),)


def test_sun_facing_wind_direction_per_turbine(time_series):
    sun_vectors = np.array([
        [[0.0, -1.0, 0.5], [-1.0, 0.0, 0.5], [0.0, 1.0, 0.5],
         [1.0, 0.0, 0.5], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 0.0, 0.5],
         [0.0, -1.0, 0.5], [0.0, 0.0, 1.0]],
    ])
    wd = sun_facing_wind_direction(sun_vectors)
    assert wd.shape == (2, len(time_series))
    vectors = to_turbine_direction_vector(wd)
    mask = np.linalg.norm(sun_vectors[..., :2], axis=-1) > 0
    np.testing.assert_allclose(
        vectors[..., :2][mask], sun_vectors[..., :2][mask], atol=1e-12)


def test_sun_facing_wind_direction_validation():
    with pytest.raises(ValueError, match="sun_vectors"):
        sun_facing_wind_direction(np.zeros((2, 2)))


def test_resolve_sun_facing_requires_sun_vectors(time_series):
    with pytest.raises(ValueError, match="sun_vectors"):
        resolve_wind_direction("sun_facing", time_series, n_src=2)


def test_resolve_sun_facing_strategy(time_series):
    sun_vectors = np.zeros((2, len(time_series), 3))
    sun_vectors[:, :, 1] = -1
    wd, mode = resolve_wind_direction(
        {"type": "sun_facing"}, time_series, n_src=2, sun_vectors=sun_vectors)
    assert mode == "sun_facing"
    assert wd.shape == (2, len(time_series))
    np.testing.assert_allclose(wd, 0)


def test_resolve_sun_facing_broadcasts_common_series(time_series):
    sun_vectors = np.zeros((len(time_series), 3))
    sun_vectors[:, 1] = -1
    wd, mode = resolve_wind_direction(
        "sun_facing", time_series, n_src=2, sun_vectors=sun_vectors)
    assert mode == "sun_facing"
    assert wd.shape == (2, len(time_series))
    np.testing.assert_allclose(wd, 0)


def test_resolve_dict_wind_direction_validation(time_series):
    with pytest.raises(ValueError, match='"type"'):
        resolve_wind_direction({}, time_series, n_src=1)
    with pytest.raises(ValueError, match="requires wd and f"):
        resolve_wind_direction(
            {"type": "wind_rose", "wd": [0]}, time_series, n_src=1)
    with pytest.raises(ValueError, match="type must be"):
        resolve_wind_direction({"type": "invalid"}, time_series, n_src=1)


def test_resolve_wind_direction_validation(time_series):
    with pytest.raises(ValueError, match="same length as time"):
        resolve_wind_direction([0, 1], time_series, n_src=1)
    with pytest.raises(ValueError, match="shape"):
        resolve_wind_direction(np.zeros((3, len(time_series))), time_series, n_src=2)
    with pytest.raises(ValueError, match="sun_facing"):
        resolve_wind_direction("invalid", time_series, n_src=1)
    with pytest.raises(ValueError, match="wind_direction must be"):
        resolve_wind_direction(np.zeros((1, 1, len(time_series))), time_series, n_src=1)
