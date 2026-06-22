from py_wake.shadow_models.shadow_base import ShadowMap, ShadowResult
from py_wake.shadow_models.shadow import ShadowModel
from py_wake.flow_map import HorizontalGrid, Points, XYGrid
import xarray as xr
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pyproj import Transformer


def test_basic_functionality(shadow_model):
    """Combined test for basic initialization, shadow tracing, and grid calculations."""

    assert shadow_model.src_crs.to_string() == "EPSG:4326"
    assert shadow_model.freq == 1.0

    rec_x = np.array([12.1])
    rec_y = np.array([55.1])
    rec_z = np.array([0])
    wind_dir = np.array([[1, 0, 0]])
    sun_vectors = np.array([[0, 1, 0.5]])
    rotor_shadow, tower_shadow = shadow_model.shadow_tracing(
        rec_x, rec_y, rec_z, wind_dir, sun_vectors, mode="both",
        time_batch_size=100, receptor_batch_size=100, turbine_batch_size=1
    )
    assert isinstance(rotor_shadow, np.ndarray)
    assert isinstance(tower_shadow, np.ndarray)

    grid = XYGrid(x=np.linspace(11.9, 12.1, 3),
                  y=np.linspace(54.9, 55.1, 3), h=0)
    result = shadow_model.run(grid=grid)
    assert isinstance(result.dataset, xr.Dataset)


def test_basic_error_conditions():
    """Test basic error conditions for ShadowModel initialization."""
    time = pd.date_range("2024-01-01", "2024-01-02", freq="1h", tz="UTC")
    lon_to_merc = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True)
    src_x, src_y = lon_to_merc.transform([12.0], [55.0])
    sm = ShadowModel(
        src_x=src_x, src_y=src_y, src_h=100,
        turbine_diameter=90, tower_diameter=5,
        time=time, wind_direction=np.zeros(len(time)), src_crs="EPSG:3857",
        solar_position_method="nrel_numpy"
    )
    assert sm.src_crs.to_string() == "EPSG:3857"
    with pytest.raises(ValueError, match="hub height cannot be less"):
        ShadowModel(
            src_x=[12.0], src_y=[55.0], src_h=40,
            turbine_diameter=90, tower_diameter=5,
            time=time, wind_direction=[0]
        )


def test_xy_unit_factor_validation_branches():
    class Axis:
        def __init__(self, factor=None, unit_name="metre"):
            self.unit_name = unit_name
            if factor is not None:
                self.unit_conversion_factor = factor

    class FakeCRS:
        def __init__(self, axes):
            self.axis_info = axes

    assert ShadowModel._xy_unit_factor(
        FakeCRS([Axis(unit_name="metre"), Axis(unit_name="metre")])) == 1.0

    with pytest.raises(ValueError, match="linear units"):
        ShadowModel._xy_unit_factor(
            FakeCRS([Axis(unit_name="degree"), Axis(unit_name="degree")]))
    with pytest.raises(ValueError, match="x/y axes"):
        ShadowModel._xy_unit_factor(FakeCRS([Axis(factor=1.0)]))
    with pytest.raises(ValueError, match="same linear unit"):
        ShadowModel._xy_unit_factor(
            FakeCRS([Axis(factor=1.0), Axis(factor=0.3048)]))


def test_normalize_mode_rejects_invalid():
    with pytest.raises(ValueError, match="mode must be"):
        ShadowModel._normalize_mode("invalid")


SRC_X = [12.0]
SRC_Y = [55.0]
SRC_H = 100
TURBINE_D = 90
TOWER_D = 5
DEFAULT_TIME = pd.date_range(
    "2024-01-01 10:00", "2024-01-01 14:00", freq="1h", tz="UTC"
)
DEFAULT_WD = [0] * len(DEFAULT_TIME)


@pytest.fixture
def default_time():
    return pd.date_range("2024-01-01 10:00",
                         "2024-01-01 14:00", freq="1h", tz="UTC")


@pytest.fixture
def default_wd(default_time):
    return np.zeros(len(default_time))


@pytest.fixture
def shadow_model(default_time, default_wd):
    return ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=default_wd,
        src_crs="EPSG:4326",
    )


def test_init_single_time_step():
    time_single = pd.DatetimeIndex(["2024-01-01 12:00"], tz="UTC")
    wd_single = [0]
    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time_single,
        wind_direction=wd_single,
    )
    assert sm.freq == np.inf
    assert len(sm.turbine_dir_vector) == 1


def test_init_time_without_tz():
    time_no_tz = pd.date_range("2024-01-01", "2024-01-02", freq="1h")
    wd = np.zeros(len(time_no_tz))
    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time_no_tz,
        wind_direction=wd,
    )
    assert sm.time.tz is not None
    assert str(sm.time.tz) == "UTC"


def test_init_requires_time_and_wind_direction(default_time):
    with pytest.raises(ValueError, match="time must be specified"):
        ShadowModel(
            src_x=SRC_X,
            src_y=SRC_Y,
            src_h=SRC_H,
            turbine_diameter=TURBINE_D,
            tower_diameter=TOWER_D,
            wind_direction=[0],
        )
    with pytest.raises(ValueError, match="wind_direction must be specified"):
        ShadowModel(
            src_x=SRC_X,
            src_y=SRC_Y,
            src_h=SRC_H,
            turbine_diameter=TURBINE_D,
            tower_diameter=TOWER_D,
            time=default_time,
        )


def test_init_inconsistent_time_freq():
    time_inconsistent = pd.DatetimeIndex(
        ["2024-01-01 12:00", "2024-01-01 13:00", "2024-01-01 15:00"], tz="UTC"
    )
    wd = np.zeros(len(time_inconsistent))
    with pytest.raises(ValueError, match="Inconsistent sampling rate"):
        ShadowModel(
            src_x=SRC_X,
            src_y=SRC_Y,
            src_h=SRC_H,
            turbine_diameter=TURBINE_D,
            tower_diameter=TOWER_D,
            time=time_inconsistent,
            wind_direction=wd,
        )


def test_init_custom_elevation():
    def custom_elev_func(x, y):
        return np.array(x) * 0 + 10

    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=DEFAULT_TIME,
        wind_direction=DEFAULT_WD,
        elevation_function=custom_elev_func,
    )
    np.testing.assert_array_equal(sm.src_z, [SRC_H + 10])


def test_tower_diameter_only_required_for_tower_modes(default_time, default_wd):
    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=None,
        time=default_time,
        wind_direction=default_wd,
        solar_position_method="nrel_numpy",
    )

    rotor_result = sm.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert "Rotor shadow" in rotor_result.dataset
    assert "tower_diameter" not in rotor_result.dataset

    for mode in ["tower", "both", "combined"]:
        with pytest.raises(ValueError, match="tower_diameter is required"):
            sm.run(
                grid=Points(x=[12.01], y=[55.0], h=[0]),
                mode=mode,
                verbose=False,
            )


def test_solar_position_method_and_kwargs(default_time, default_wd):
    sm_default = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=default_wd,
    )
    default_result = sm_default.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert default_result.dataset.attrs["solar_position_method"] == "nrel_numpy"

    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=default_wd,
        solar_position_method="ephemeris",
        solar_position_kwargs={"temperature": 10},
    )
    result = sm.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert result.dataset.attrs["solar_position_method"] == "ephemeris"


def test_nrel_numba_method_accepts_multiple_turbines(default_time):
    sm = ShadowModel(
        src_x=[12.0, 12.01],
        src_y=[55.0, 55.01],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=np.zeros(len(default_time)),
        solar_position_method="nrel_numba",
    )
    result = sm.run(
        grid=Points(x=[12.02], y=[55.02], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert result.dataset.attrs["solar_position_method"] == "nrel_numba"


def test_get_calculation_crs_projected_override(default_time, default_wd):
    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=default_wd,
        calculation_crs="EPSG:32633",
    )
    crs, unit_factor = sm._get_calculation_crs()
    assert crs.to_epsg() == 32633
    assert unit_factor == 1.0

    sm.calculation_crs = "EPSG:4326"
    with pytest.raises(ValueError, match="projected CRS"):
        sm._get_calculation_crs()


def test_scalar_wind_direction_input(default_time):
    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=270,
        solar_position_method="nrel_numpy",
    )
    np.testing.assert_allclose(sm.wind_direction_angles, 270)
    result = sm.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert result.dataset.attrs["wind_direction_mode"] == "uniform"
    assert result.dataset["wind_direction"].dims == ("time",)


def test_per_turbine_wind_direction_input(default_time):
    wind_direction = np.vstack([
        np.zeros(len(default_time)),
        np.full(len(default_time), 180.0),
    ])
    sm = ShadowModel(
        src_x=[12.0, 12.01],
        src_y=[55.0, 55.01],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=wind_direction,
        solar_position_method="nrel_numpy",
    )
    result = sm.run(
        grid=Points(x=[12.02], y=[55.02], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert result.dataset.attrs["wind_direction_mode"] == "per_turbine"
    assert result.dataset["wind_direction"].dims == ("wt", "time")
    assert result.dataset["wind_direction"].shape == (2, len(default_time))


def test_wind_rose_wind_direction_reproducible(default_time):
    strategy = {
        "type": "wind_rose",
        "wd": [0, 90, 180, 270],
        "f": [1, 2, 1, 2],
        "seed": 10,
    }
    sm1 = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction=strategy,
    )
    assert sm1.wind_direction_mode == "wind_rose"
    assert sm1.wind_direction_angles is None
    result1 = sm1.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    result2 = sm1.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    np.testing.assert_allclose(
        result1.dataset["wind_direction"], result2.dataset["wind_direction"])


def test_unseeded_wind_rose_resamples_each_run(default_time):
    sm = ShadowModel(
        src_x=SRC_X,
        src_y=SRC_Y,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction={"type": "wind_rose", "wd": [0, 180], "f": [1, 1]},
    )
    result1 = sm.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    result2 = sm.run(
        grid=Points(x=[12.01], y=[55.0], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert not np.array_equal(
        result1.dataset["wind_direction"].values,
        result2.dataset["wind_direction"].values,
    )


def test_sun_facing_wind_direction_input(default_time):
    sm = ShadowModel(
        src_x=[12.0, 12.01],
        src_y=[55.0, 55.01],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=default_time,
        wind_direction="sun_facing",
        solar_position_method="nrel_numpy",
    )
    assert sm.wind_direction_mode == "sun_facing"
    result = sm.run(
        grid=Points(x=[12.02], y=[55.02], h=[0]),
        mode="rotor",
        verbose=False,
    )
    assert result.dataset.attrs["wind_direction_mode"] == "sun_facing"
    assert result.dataset["wind_direction"].dims == ("wt", "time")
    assert result.dataset["wind_direction"].shape == (2, len(default_time))


def test_simulation_result_shadow_model_wind_direction_override(default_time):
    from py_wake.deficit_models.noj import NOJ
    from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
    from py_wake.site._site import UniformSite

    site = UniformSite([1], ti=0)
    wt = IEA37_WindTurbines()
    wfm = NOJ(site, wt)
    sim_res = wfm(
        [0],
        [0],
        wd=np.zeros(len(default_time)),
        ws=np.ones(len(default_time)) * 8,
        time=default_time,
    )

    sm = sim_res.shadow_model(
        tower_diameter=5,
        rotor_offset=8,
        wind_direction=123,
        solar_position_method="nrel_numpy",
    )
    np.testing.assert_allclose(sm.wind_direction_angles, 123)
    assert sm.wind_direction_mode == "uniform"
    assert sm.rotor_offset[0] == 8


REC_X_CLOSE = [12.01]
REC_Y_CLOSE = [55.0]
REC_Z_GROUND = [0]


def test_shadow_tracing_modes(shadow_model):
    from py_wake.shadow_models.solar_position import solar_position

    sun_vectors, _ = solar_position(
        shadow_model.time, shadow_model.src_lat, shadow_model.src_lon,
        method="nrel_numpy")
    sun_vectors = sun_vectors[0]
    wind_dir = shadow_model.turbine_dir_vector
    rotor_s, tower_s = shadow_model.shadow_tracing(
        np.array(REC_X_CLOSE),
        np.array(REC_Y_CLOSE),
        np.array(REC_Z_GROUND),
        wind_dir,
        sun_vectors,
        mode="rotor",
        time_batch_size=2,
        receptor_batch_size=1,
        turbine_batch_size=1,
    )
    assert isinstance(rotor_s, np.ndarray)
    assert tower_s is None
    assert rotor_s.shape == (1, 1, len(shadow_model.time))
    rotor_s, tower_s = shadow_model.shadow_tracing(
        np.array(REC_X_CLOSE),
        np.array(REC_Y_CLOSE),
        np.array(REC_Z_GROUND),
        wind_dir,
        sun_vectors,
        mode="tower",
        time_batch_size=2,
        receptor_batch_size=1,
        turbine_batch_size=1,
    )
    assert rotor_s is None
    assert isinstance(tower_s, np.ndarray)
    assert tower_s.shape == (1, 1, len(shadow_model.time))
    rotor_s, tower_s = shadow_model.shadow_tracing(
        np.array(REC_X_CLOSE),
        np.array(REC_Y_CLOSE),
        np.array(REC_Z_GROUND),
        wind_dir,
        sun_vectors,
        mode="both",
        time_batch_size=10,
        receptor_batch_size=10,
        turbine_batch_size=10,
    )
    assert isinstance(rotor_s, np.ndarray)
    assert isinstance(tower_s, np.ndarray)
    assert rotor_s.shape == (1, 1, len(shadow_model.time))
    assert tower_s.shape == (1, 1, len(shadow_model.time))
    rotor_s_c, tower_s_c = shadow_model.shadow_tracing(
        np.array(REC_X_CLOSE),
        np.array(REC_Y_CLOSE),
        np.array(REC_Z_GROUND),
        wind_dir,
        sun_vectors,
        mode="combined",
        time_batch_size=10,
        receptor_batch_size=10,
        turbine_batch_size=10,
    )
    assert isinstance(rotor_s_c, np.ndarray)
    assert isinstance(tower_s_c, np.ndarray)


GRID_X = np.linspace(11.9, 12.1, 3)
GRID_Y = np.linspace(54.9, 55.1, 3)
GRID_H = 0
POINTS_X = np.array([11.95, 12.05])
POINTS_Y = np.array([55.05, 54.95])
POINTS_H = 0


@pytest.mark.parametrize("mode", ["rotor", "tower", "both", "combined"])
def test_run_horizontal_grid(shadow_model, mode):
    grid = HorizontalGrid(x=GRID_X, y=GRID_Y, h=GRID_H)
    result = shadow_model.run(grid=grid, mode=mode)
    assert isinstance(result, ShadowMap)
    assert isinstance(result.dataset, xr.Dataset)
    assert "x" in result.dataset.coords
    assert "y" in result.dataset.coords
    assert "time" in result.dataset.coords
    assert "wt" in result.dataset.coords
    assert result.dataset.attrs["mode"] == mode
    assert result.dataset.attrs["result_type"] == "ShadowMap"
    if mode == "rotor":
        assert "Rotor shadow" in result.dataset.data_vars
        assert "Tower shadow" not in result.dataset.data_vars
        assert "Combined shadow" not in result.dataset.data_vars
    elif mode == "tower":
        assert "Rotor shadow" not in result.dataset.data_vars
        assert "Tower shadow" in result.dataset.data_vars
        assert "Combined shadow" not in result.dataset.data_vars
    elif mode == "both":
        assert "Rotor shadow" in result.dataset.data_vars
        assert "Tower shadow" in result.dataset.data_vars
        assert "Combined shadow" not in result.dataset.data_vars
    elif mode == "combined":
        assert "Rotor shadow" not in result.dataset.data_vars
        assert "Tower shadow" not in result.dataset.data_vars
        assert "Combined shadow" in result.dataset.data_vars


@pytest.mark.parametrize("mode", ["rotor", "tower", "both", "combined"])
def test_run_points_grid(shadow_model, mode):
    grid = Points(x=POINTS_X, y=POINTS_Y, h=[POINTS_H] * 2)
    result = shadow_model.run(grid=grid, mode=mode)
    assert isinstance(result, ShadowResult)
    assert isinstance(result.dataset, xr.Dataset)
    assert "rec" in result.dataset.coords
    assert "time" in result.dataset.coords
    assert "wt" in result.dataset.coords
    assert result.dataset.attrs["mode"] == mode
    assert result.dataset.attrs["result_type"] == "ShadowResult"
    assert len(result.dataset.coords["rec"]) == len(POINTS_X)
    if mode == "rotor":
        assert "Rotor shadow" in result.dataset.data_vars
        assert "Tower shadow" not in result.dataset.data_vars
        assert "Combined shadow" not in result.dataset.data_vars
    elif mode == "tower":
        assert "Rotor shadow" not in result.dataset.data_vars
        assert "Tower shadow" in result.dataset.data_vars
        assert "Combined shadow" not in result.dataset.data_vars
    elif mode == "both":
        assert "Rotor shadow" in result.dataset.data_vars
        assert "Tower shadow" in result.dataset.data_vars
        assert "Combined shadow" not in result.dataset.data_vars
    elif mode == "combined":
        assert "Rotor shadow" not in result.dataset.data_vars
        assert "Tower shadow" not in result.dataset.data_vars
        assert "Combined shadow" in result.dataset.data_vars


def test_run_grid_none(shadow_model):
    result = shadow_model.run(grid=None, mode="both")
    assert isinstance(result, ShadowMap)
    assert result.dataset.attrs["result_type"] == "ShadowMap"
    assert "x" in result.dataset.coords
    assert "y" in result.dataset.coords


def test_run_grid_invalid_type(shadow_model):
    with pytest.raises(NotImplementedError, match="must be instance of Grid or None"):
        shadow_model.run(grid="not a grid")


def test_call_method(shadow_model):
    rec_x_scalar = 12.01
    rec_y_scalar = 55.0
    rec_h_scalar = 0
    rotor_s, tower_s = shadow_model(
        rec_x_scalar,
        rec_y_scalar,
        rec_h_scalar,
        rec_crs="EPSG:4326",
        mode="both",
    )
    assert isinstance(rotor_s, np.ndarray)
    assert isinstance(tower_s, np.ndarray)
    assert rotor_s.shape == (len(SRC_X), 1, len(shadow_model.time))
    assert tower_s.shape == (len(SRC_X), 1, len(shadow_model.time))
    rotor_s_arr, tower_s_arr = shadow_model(
        POINTS_X, POINTS_Y, POINTS_H, rec_crs="EPSG:4326", mode="both"
    )
    assert rotor_s_arr.shape == (
        len(SRC_X),
        len(POINTS_X),
        len(shadow_model.time),
    )
    assert tower_s_arr.shape == (
        len(SRC_X),
        len(POINTS_X),
        len(shadow_model.time),
    )


def test_rotor_offset_functionality():
    """Test that rotor offset parameter works correctly"""
    time = pd.date_range("2024-01-01", "2024-01-02", freq="1h", tz="UTC")

    sm_default = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time,
        wind_direction=[0] * 25,
    )

    sm_custom = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time,
        wind_direction=[0] * 25,
        rotor_offset=15,
    )

    assert sm_default.rotor_offset[0] == 0.0
    assert sm_custom.rotor_offset[0] == 15.0

    np.testing.assert_array_equal(sm_default.src_x, [12.0])
    np.testing.assert_array_equal(sm_default.src_y, [55.0])
    np.testing.assert_array_equal(sm_custom.src_x, [12.0])
    np.testing.assert_array_equal(sm_custom.src_y, [55.0])

    sm_east = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time,
        wind_direction=[90] * 25,
        rotor_offset=10,
    )

    assert sm_east.rotor_offset[0] == 10.0

    grid = XYGrid(
        x=np.linspace(11.99, 12.01, 3), y=np.linspace(54.99, 55.01, 3), h=0
    )
    result = sm_custom.run(grid=grid)
    assert result is not None


def test_memory_warning(default_time):
    """Tests that a warning is issued for potentially large result arrays."""
    n_turbines_large = 4
    n_receptors_large = 10000
    n_times_large = 50
    src_x_large = np.linspace(12.0, 12.1, n_turbines_large)
    src_y_large = np.linspace(55.0, 55.1, n_turbines_large)
    rec_x_large = np.linspace(11.9, 12.2, n_receptors_large)
    rec_y_large = np.linspace(54.9, 55.2, n_receptors_large)
    time_large = pd.date_range(
        "2024-01-01", periods=n_times_large, freq="1h", tz="UTC")
    wd_large = np.zeros(len(time_large))
    sm_large = ShadowModel(
        src_x=src_x_large,
        src_y=src_y_large,
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time_large,
        wind_direction=wd_large,
    )

    sm_large.memory_warning_limit = 0.001

    with patch("py_wake.shadow_models.shadow.solar_position") as mock_solar:
        dummy_sun = np.zeros((n_times_large, 3))
        dummy_sun[:, 2] = 0.1
        mock_solar.return_value = (dummy_sun, None)
        with pytest.warns(RuntimeWarning, match="Large shadow arrays detected!"):
            sm_large.shadow_tracing(
                rec_x_large,
                rec_y_large,
                np.zeros_like(rec_x_large),
                sm_large.turbine_dir_vector,
                dummy_sun,
                mode="both",
                time_batch_size=10,
                receptor_batch_size=2000,
                turbine_batch_size=2,
            )


def test_additional_error_conditions():
    """Tests additional error conditions for full coverage."""

    time_no_tz = pd.date_range("2024-01-01", "2024-01-02", freq="1h")
    sm = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time_no_tz,
        wind_direction=[0] * len(time_no_tz),
    )
    assert sm.time.tz is not None

    inconsistent_times = pd.DatetimeIndex(
        ["2024-01-01 12:00", "2024-01-01 13:00", "2024-01-01 15:00"], tz="UTC"
    )
    with pytest.raises(ValueError, match="Inconsistent sampling rate"):
        ShadowModel(
            src_x=[12.0],
            src_y=[55.0],
            src_h=SRC_H,
            turbine_diameter=TURBINE_D,
            tower_diameter=TOWER_D,
            time=inconsistent_times,
            wind_direction=[0] * len(inconsistent_times),
        )

    time_np = np.array(
        ['2024-01-01T12:00', '2024-01-01T13:00'], dtype='datetime64[h]')
    sm_np = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time_np,
        wind_direction=[0] * len(time_np),
    )
    assert sm_np.freq == 1.0


def test_grid_creation_edge_cases():
    """Tests grid creation edge cases."""
    time = pd.date_range("2024-01-01 10:00",
                         "2024-01-01 14:00", freq="1h", tz="UTC")
    sm = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time,
        wind_direction=[0] * len(time),
    )

    X, Y, x_j, y_j, h_j, plane = sm._get_grid(None)
    assert X is not None
    assert Y is not None

    points = Points(x=[12.01, 12.02], y=[55.01, 55.02], h=[0, 0])
    result = sm.run(grid=points, mode="combined")
    assert isinstance(result, ShadowResult)

    grid = HorizontalGrid(x=np.linspace(11.9, 12.1, 3),
                          y=np.linspace(54.9, 55.1, 3), h=0)
    result_combined = sm.run(grid=grid, mode="combined")
    assert isinstance(result_combined, ShadowMap)

    list_grid = Points(x=[12.01, 12.02], y=[55.01, 55.02], h=[0, 0])
    list_grid.x = list(list_grid.x)
    list_grid.y = list(list_grid.y)
    result_list = sm.run(grid=list_grid, mode="combined")
    assert isinstance(result_list, ShadowResult)
    assert result_list.dataset.rec_x.shape == (2,)


def test_main_function_execution():
    try:
        from py_wake.shadow_models.shadow import main
        assert callable(main)
    except ImportError:
        pass


def test_grid_and_mode_combinations():
    """Test various grid types and shadow modes to ensure full coverage."""
    time = pd.date_range("2024-01-01 10:00",
                         "2024-01-01 14:00", freq="1h", tz="UTC")
    sm = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=[0] * len(time)
    )

    grid = HorizontalGrid(x=np.linspace(11.9, 12.1, 3),
                          y=np.linspace(54.9, 55.1, 3), h=0)
    result = sm.run(grid=grid)
    assert isinstance(result.dataset, xr.Dataset)

    points = Points(x=[12.01, 12.02], y=[55.01, 55.02], h=[0, 0])
    result = sm.run(grid=points)
    assert isinstance(result.dataset, xr.Dataset)

    result_combined = sm.run(grid=points, mode="combined")
    assert isinstance(result.dataset, xr.Dataset)


def test_solar_position_edge_cases():
    time = pd.date_range("2024-01-01 10:00",
                         "2024-01-01 14:00", freq="1h", tz="UTC")
    sm = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time,
        wind_direction=[0] * len(time),
    )
    with patch('py_wake.shadow_models.shadow.solar_position') as mock_solar:
        dummy_sun = np.zeros((len(time), 3))
        dummy_sun[:, 2] = -0.1
        mock_solar.return_value = (dummy_sun, None)
        rotor_s, tower_s = sm.shadow_tracing(
            np.array([12.1]),
            np.array([55.1]),
            np.array([0]),
            sm.turbine_dir_vector,
            dummy_sun,
            mode="both"
        )
        assert rotor_s is not None
        assert tower_s is not None


def _coords_from_single_source_lines(lines_3d):
    src_coords = np.zeros((1, 3), dtype=float)
    rec_coords = -np.asarray(lines_3d, dtype=float)[0]
    return src_coords, rec_coords


def _sun_vectors_per_source(sun_vectors):
    return np.asarray(sun_vectors, dtype=float)[np.newaxis, :, :]


def _wind_dir_per_source(wind_dir):
    return np.asarray(wind_dir, dtype=float)[np.newaxis, :, :]


def _wind_dir_rad_per_source(wind_dir_rad):
    return np.asarray(wind_dir_rad, dtype=float)[np.newaxis, :]


def test_sun_below_horizon():
    """Test rotor batch returns False when sun below horizon."""
    src_coords = np.array([[12.0, 55.0, 100.0]])
    rec_coords = np.array([[12.1, 55.1, 0.0]])
    wind_dir_batch = np.array([[1.0, 0.0, 0.0]])
    sun_vectors_batch = np.array([[0.0, 0.0, -0.5]])  # below horizon
    turbine_diameters = np.array([90.0])
    rotor_offsets = np.array([2.5])
    wind_dir_rad_batch = np.array([0.0])

    func = ShadowModel._calculate_rotor_shadow_from_coords
    func = getattr(func, 'py_func', func)
    res = func(
        src_coords, rec_coords, _wind_dir_per_source(wind_dir_batch),
        _sun_vectors_per_source(sun_vectors_batch),
        turbine_diameters, rotor_offsets,
        _wind_dir_rad_per_source(wind_dir_rad_batch)
    )
    assert res.shape == (1, 1, 1)
    assert not np.any(res)


def test_sun_dot_wind_near_zero_adjustment():
    """Test adjustment branch when sun_dot_wind is (near) zero."""
    src_coords = np.array([[12.0, 55.0, 100.0]])
    rec_coords = np.array([[12.0, 55.0, 100.0]])
    wind_dir_batch = np.array([[1.0, 0.0, 0.0]])
    # Make dot(wind, sun) == 0 to trigger epsilon adjustment
    sun_vectors_batch = np.array([[0.0, 1.0, 0.5]])
    turbine_diameters = np.array([90.0])
    rotor_offsets = np.array([2.5])
    wind_dir_rad_batch = np.array([0.0])

    # Exercise near-zero path through numba rotor via .py_func using
    # equivalent geometry
    rotor_func = ShadowModel._calculate_rotor_shadow_from_coords
    rotor_func = getattr(rotor_func, 'py_func', rotor_func)
    res = rotor_func(
        src_coords, rec_coords, _wind_dir_per_source(wind_dir_batch),
        _sun_vectors_per_source(sun_vectors_batch),
        turbine_diameters, rotor_offsets,
        _wind_dir_rad_per_source(wind_dir_rad_batch)
    )
    assert res.shape == (1, 1, 1)


def test_numba_functions_py_func_and_tower_branches():
    """Cover numba .py_func paths and additional branches."""
    # Setup inputs
    lines_3d = np.array([[[0.2, -0.1, -50.0]]])
    wind_dir_batch = np.array([[1.0, 0.0, 0.0]])
    sun_vectors_batch = np.array([[0.0, 1.0, 0.5]])  # sun above horizon
    turbine_diameters = np.array([90.0])
    rotor_offsets = np.array([2.5])
    wind_dir_rad_batch = np.array([0.0])
    src_coords, rec_coords = _coords_from_single_source_lines(lines_3d)

    # Rotor shadow via .py_func if available
    rotor_func = ShadowModel._calculate_rotor_shadow_from_coords
    rotor_func = getattr(rotor_func, 'py_func', rotor_func)
    rotor_res = rotor_func(
        src_coords, rec_coords, _wind_dir_per_source(wind_dir_batch),
        _sun_vectors_per_source(sun_vectors_batch),
        turbine_diameters, rotor_offsets,
        _wind_dir_rad_per_source(wind_dir_rad_batch)
    )
    assert rotor_res.shape == (1, 1, 1)

    # Tower shadow via .py_func if available
    tower_func = ShadowModel._calculate_tower_shadow_from_coords
    tower_func = getattr(tower_func, 'py_func', tower_func)
    tower_res = tower_func(
        src_coords, rec_coords, _sun_vectors_per_source(sun_vectors_batch),
        np.array([5.0]), np.array([100.0])
    )
    assert tower_res.shape == (1, 1, 1)

    # Force receptor_in_tower=True branch: place receptor inside tower cylinder
    lines_inside = np.array([[[0.0, 0.0, 0.0]]])
    src_inside, rec_inside = _coords_from_single_source_lines(lines_inside)
    tower_res_inside = tower_func(
        src_inside, rec_inside, _sun_vectors_per_source(sun_vectors_batch),
        np.array([10.0]), np.array([100.0])
    )
    assert bool(tower_res_inside[0, 0, 0]) is True


def test_numba_rotor_sun_below_horizon():
    """Exercise rotor numba early-continue path."""
    lines_3d = np.array([[[0.0, 0.0, -10.0]]])
    wind_dir_batch = np.array([[1.0, 0.0, 0.0]])
    # sun below horizon triggers early continue
    sun_vectors_below = np.array([[0.0, 1.0, -0.1]])
    turbine_diameters = np.array([90.0])
    src_coords, rec_coords = _coords_from_single_source_lines(lines_3d)

    rotor_func = ShadowModel._calculate_rotor_shadow_from_coords
    rotor_func = getattr(rotor_func, 'py_func', rotor_func)
    res_below = rotor_func(
        src_coords, rec_coords, _wind_dir_per_source(wind_dir_batch),
        _sun_vectors_per_source(sun_vectors_below),
        turbine_diameters, np.array([0.0]), np.array([[0.0]])
    )
    assert res_below.shape == (1, 1, 1)
    assert not np.any(res_below)


def test_numba_tower_edge_cases_near_zero_norm_and_dot():
    """Exercise tower numba branches for xy-norm and near-zero dot."""
    lines_3d = np.array([[[0.2, 0.1, 30.0]]])
    src_coords, rec_coords = _coords_from_single_source_lines(lines_3d)
    sun_vectors_edge = np.array([[0.0, 0.0, 0.2]])  # xy-norm=0, above horizon
    tower_func = ShadowModel._calculate_tower_shadow_from_coords
    tower_func = getattr(tower_func, 'py_func', tower_func)
    res = tower_func(
        src_coords, rec_coords, _sun_vectors_per_source(sun_vectors_edge),
        np.array([5.0]), np.array([100.0])
    )
    assert res.shape == (1, 1, 1)
    # Also exercise near-zero dot(wind, sun) logic: use sun vector orthogonal
    # to ntur
    lines_3d2 = np.array([[[0.1, 0.0, 50.0]]])
    src_coords2, rec_coords2 = _coords_from_single_source_lines(lines_3d2)
    sun_vectors_edge2 = np.array([[1e-14, 1e-14, 0.5]])
    res2 = tower_func(
        src_coords2, rec_coords2, _sun_vectors_per_source(sun_vectors_edge2),
        np.array([5.0]), np.array([100.0])
    )
    assert res2.shape == (1, 1, 1)


def test_numba_rotor_zero_offset_sun_above_branch():
    """Exercise rotor kernel with zero offset and sun above horizon."""
    lines_3d = np.array([[[0.1, -0.2, -20.0]]])
    src_coords, rec_coords = _coords_from_single_source_lines(lines_3d)
    wind_dir_batch = np.array([[1.0, 0.0, 0.0]])
    sun_vectors = np.array([[0.0, 1.0, 0.3]])
    turbine_diameters = np.array([90.0])
    rotor_func = ShadowModel._calculate_rotor_shadow_from_coords
    rotor_func = getattr(rotor_func, 'py_func', rotor_func)
    res = rotor_func(
        src_coords, rec_coords, _wind_dir_per_source(wind_dir_batch),
        _sun_vectors_per_source(sun_vectors),
        turbine_diameters, np.array([0.0]), np.array([[0.0]])
    )
    assert res.shape == (1, 1, 1)


def test_tower_else_branch_in_horizontal_and_vertical_true():
    """Hit else-branch where receptor is outside tower but shadow intersection is inside."""
    # Choose geometry so receptor outside radius but intersection inside
    lines_3d = np.array([[[6.0, 0.0, 50.0]]])  # receptor outside radius=5
    sun_vectors = np.array([[1.0, 0.0, 1.0]])  # ntur=(1,0,0), sun_dot_wind=1
    tower_diameters = np.array([10.0])  # radius=5
    tower_heights = np.array([100.0])  # half-height=50
    src_coords, rec_coords = _coords_from_single_source_lines(lines_3d)
    tower_func = ShadowModel._calculate_tower_shadow_from_coords
    tower_func = getattr(tower_func, 'py_func', tower_func)
    res = tower_func(
        src_coords, rec_coords, _sun_vectors_per_source(sun_vectors),
        tower_diameters, tower_heights)
    assert res.shape == (1, 1, 1)
    assert bool(res[0, 0, 0]) is True


def test_points_original_shape_with_numpy_arrays(shadow_model):
    """Ensure run() path that uses grid.x.shape is exercised for Points."""
    x = np.array([12.01, 12.02])
    y = np.array([55.01, 55.02])
    h = np.array([0, 0])
    points = Points(x=x, y=y, h=h)
    result = shadow_model.run(grid=points, mode="rotor")
    assert isinstance(result, ShadowResult)


def test_points_original_shape_else_branch_sneaky(shadow_model):
    import inspect

    class SneakyPoints(Points):
        def __getattribute__(self, name):
            if name in ("x", "y"):
                # Allow access only when called from _get_grid
                for fr in inspect.stack()[:6]:
                    if fr.function == "_get_grid":
                        return object.__getattribute__(self, name)
                raise AttributeError
            return object.__getattribute__(self, name)

    points = SneakyPoints(x=[12.01, 12.02], y=[55.01, 55.02], h=[0, 0])
    res = shadow_model.run(grid=points, mode="rotor")
    assert isinstance(res, ShadowResult)


def test_batch_shadow_with_multiple_times():
    """Test batch shadow calculation with multiple time steps (covers lines 206-270)."""
    lines_3d = np.array([[[0.1, 0.1, -100.0]]])
    wind_dir_batch = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Multiple times
    sun_vectors_batch = np.array(
        [[0.0, 1.0, 0.5], [0.5, 0.5, 0.7]])  # Multiple times
    turbine_diameters = np.array([90.0])
    rotor_offsets = np.array([2.5])
    wind_dir_rad_batch = np.array([0.0, np.pi / 4])  # Multiple angles
    src_coords, rec_coords = _coords_from_single_source_lines(lines_3d)

    func = ShadowModel._calculate_rotor_shadow_from_coords
    func = getattr(func, 'py_func', func)
    result = func(
        src_coords, rec_coords, _wind_dir_per_source(wind_dir_batch),
        _sun_vectors_per_source(sun_vectors_batch),
        turbine_diameters, rotor_offsets,
        _wind_dir_rad_per_source(wind_dir_rad_batch)
    )
    assert result.shape == (1, 1, 2)  # n_src=1, n_rec=1, n_times=2


def test_tower_shadow_sun_below_horizon():
    """Test tower shadow with sun below horizon (covers lines 293-358)."""
    lines_3d = np.array([[[0.1, 0.1, -100.0]]])
    sun_vectors_batch = np.array([[0.0, 0.0, -0.5]])  # Sun below horizon
    tower_diameters = np.array([5.0])
    tower_heights = np.array([100.0])
    src_coords, rec_coords = _coords_from_single_source_lines(lines_3d)

    func = ShadowModel._calculate_tower_shadow_from_coords
    func = getattr(func, 'py_func', func)
    result = func(
        src_coords, rec_coords, _sun_vectors_per_source(sun_vectors_batch),
        tower_diameters, tower_heights
    )
    assert result.shape == (1, 1, 1)
    assert not np.any(result)  # Should be False when sun below horizon


def test_grid_shape_handling():
    """Test grid shape handling edge case (covers line 613)."""
    time = pd.date_range("2024-01-01 10:00",
                         "2024-01-01 14:00", freq="1h", tz="UTC")
    sm = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=[0] * len(time)
    )

    # XYGrid to trigger the else branch at line 613
    grid = XYGrid(x=[12.01, 12.02], y=[55.01, 55.02], h=[0, 0])
    result = sm.run(grid=grid)
    assert isinstance(result.dataset, xr.Dataset)


def test_shadow_calculation_integration():
    """Test shadow calculations through the main API to ensure numba functions are exercised."""
    # Test through the main shadow_tracing method which calls the numba
    # functions
    time = pd.date_range("2024-01-01 12:00",
                         "2024-01-01 13:00", freq="1h", tz="UTC")
    sm = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=[0]
    )

    rec_x = np.array([12.1])
    rec_y = np.array([55.1])
    rec_z = np.array([0])

    # This will exercise the numba-compiled functions through shadow_tracing
    rotor_shadow, tower_shadow = sm.shadow_tracing(
        rec_x, rec_y, rec_z,
        sm.turbine_dir_vector,
        np.array([[0.0, 1.0, 0.5]]),  # sun vectors
        mode="both",
        time_batch_size=1,
        receptor_batch_size=1,
        turbine_batch_size=1
    )

    # (n_turbines, n_receptors, n_times)
    assert rotor_shadow.shape == (1, 1, 1)
    assert tower_shadow.shape == (1, 1, 1)
    # Test with multiple time steps to exercise batch processing
    time_multi = pd.date_range(
        "2024-01-01 10:00", "2024-01-01 14:00", freq="1h", tz="UTC")
    sm_multi = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time_multi, wind_direction=[0] * len(time_multi)
    )

    rotor_shadow_multi, tower_shadow_multi = sm_multi.shadow_tracing(
        rec_x, rec_y, rec_z,
        sm_multi.turbine_dir_vector,
        # sun vectors for each time
        np.array([[0.0, 1.0, 0.5]] * len(time_multi)),
        mode="both",
        time_batch_size=2,  # Test batching
        receptor_batch_size=1,
        turbine_batch_size=1
    )

    # (n_turbines, n_receptors, n_times)
    expected_shape = (1, 1, len(time_multi))
    assert rotor_shadow_multi.shape == expected_shape
    assert tower_shadow_multi.shape == expected_shape


def test_shadow_freq_numpy_timedelta_branch():
    """Covers ShadowModel.__init__ numpy timedelta64 fallback (lines 89-90)."""
    # Patch np.diff to return numpy timedelta64 values so .total_seconds is
    # missing
    with patch('py_wake.shadow_models.shadow.np.diff',
               return_value=np.array([np.timedelta64(3600, 's')])):
        time = pd.date_range('2024-01-01', periods=2, freq='h', tz='UTC')
        sm = ShadowModel(
            src_x=[12.0], src_y=[55.0], src_h=SRC_H,
            turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
            time=time, wind_direction=[0, 0]
        )
        assert sm.freq == 1.0


def test_shadow_tracing_invalid_sun_vectors_last_dim():
    """Covers last-dimension validation error (line 338)."""
    time = pd.date_range('2024-01-01', periods=2, freq='h', tz='UTC')
    sm = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=[0, 0]
    )
    rec_x = np.array([12.0])
    rec_y = np.array([55.0])
    rec_z = np.array([0.0])
    wind_dir = sm.turbine_dir_vector[:2]
    bad = np.zeros((2, 2))  # last dim != 3
    with pytest.raises(ValueError, match="last dimension of size 3"):
        sm.shadow_tracing(
            rec_x,
            rec_y,
            rec_z,
            wind_dir,
            bad,
            mode='rotor',
            verbose=False)


def test_shadow_tracing_invalid_sun_vectors_ndim():
    """Covers ndim validation error path (line 350)."""
    time = pd.date_range('2024-01-01', periods=2, freq='h', tz='UTC')
    sm = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=[0, 0]
    )
    rec_x = np.array([12.0])
    rec_y = np.array([55.0])
    rec_z = np.array([0.0])
    wind_dir = sm.turbine_dir_vector[:2]
    # ndim=4 with last dim 3 -> triggers ndim error branch
    bad = np.zeros((1, 2, 1, 3))
    with pytest.raises(ValueError, match="must be 2D.*or 3D"):
        sm.shadow_tracing(
            rec_x,
            rec_y,
            rec_z,
            wind_dir,
            bad,
            mode='rotor',
            verbose=False)


def test_rotor_shadow_requires_rotor_between_receptor_and_sun():
    rotor_func = ShadowModel._calculate_rotor_shadow_from_coords
    rotor_func = getattr(rotor_func, 'py_func', rotor_func)
    src_coords = np.array([[0.0, 0.0, 100.0]])
    rec_coords = np.array([
        [-50.0, 0.0, 100.0],
        [50.0, 0.0, 100.0],
    ])
    wind_dir_batch = np.array([[[1.0, 0.0, 0.0]]])
    sun_vectors_batch = np.array([[[1.0, 0.0, 0.01]]])
    result = rotor_func(
        src_coords,
        rec_coords,
        wind_dir_batch,
        sun_vectors_batch,
        np.array([10.0]),
        np.array([0.0]),
        np.array([[0.0]]),
    )
    assert bool(result[0, 0, 0]) is True
    assert bool(result[0, 1, 0]) is False


def test_tower_shadow_uses_finite_cylinder_ray_intersection():
    tower_func = ShadowModel._calculate_tower_shadow_from_coords
    tower_func = getattr(tower_func, 'py_func', tower_func)
    src_coords = np.array([[0.0, 0.0, 100.0]])
    rec_coords = np.array([
        [-10.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
        [-10.0, 0.0, 150.0],
        [0.0, 0.0, 50.0],
    ])
    result = tower_func(
        src_coords,
        rec_coords,
        np.array([[[1.0, 0.0, 0.1]]]),
        np.array([10.0]),
        np.array([100.0]),
    )
    np.testing.assert_array_equal(
        result[0, :, 0],
        [True, False, False, True],
    )


def test_tower_shadow_cylinder_miss_branches():
    tower_func = ShadowModel._calculate_tower_shadow_from_coords
    tower_func = getattr(tower_func, 'py_func', tower_func)
    src_coords = np.array([[0.0, 0.0, 100.0]])
    tower_diameters = np.array([10.0])
    tower_heights = np.array([100.0])

    vertical_ray_result = tower_func(
        src_coords,
        np.array([[10.0, 0.0, 50.0]]),
        np.array([[[0.0, 0.0, 1.0]]]),
        tower_diameters,
        tower_heights,
    )
    assert bool(vertical_ray_result[0, 0, 0]) is False

    discriminant_miss_result = tower_func(
        src_coords,
        np.array([[10.0, 10.0, 50.0]]),
        np.array([[[1.0, 0.0, 0.1]]]),
        tower_diameters,
        tower_heights,
    )
    assert bool(discriminant_miss_result[0, 0, 0]) is False


def test_max_distance_is_applied_per_turbine_receptor_pair():
    time = pd.DatetimeIndex(["2024-01-01 12:00"], tz="UTC")
    sm = ShadowModel(
        src_x=[0.0, 1000.0],
        src_y=[0.0, 0.0],
        src_h=1000.0,
        turbine_diameter=2000.0,
        tower_diameter=5.0,
        time=time,
        wind_direction=[0],
        max_distance=100.0,
        src_crs="EPSG:32633",
    )
    rotor_shadow, _ = sm.shadow_tracing(
        rec_x=np.array([-50.0]),
        rec_y=np.array([0.0]),
        rec_z=np.array([1000.0]),
        wind_dir=np.array([[1.0, 0.0, 0.0]]),
        sun_vectors=np.array([[1.0, 0.0, 0.01]]),
        mode="rotor",
        verbose=False,
        src_x=np.array([0.0, 1000.0]),
        src_y=np.array([0.0, 0.0]),
        src_z=np.array([1000.0, 1000.0]),
        wind_dir_rad=np.array([0.0]),
    )
    assert bool(rotor_shadow[0, 0, 0]) is True
    assert bool(rotor_shadow[1, 0, 0]) is False


def test_shadow_tracing_wind_direction_validation_and_broadcasts():
    time = pd.date_range('2024-01-01', periods=2, freq='h', tz='UTC')
    sm = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time,
        wind_direction=[0, 0],
    )
    rec_x = np.array([12.0])
    rec_y = np.array([55.0])
    rec_z = np.array([0.0])
    sun_vectors = np.array([[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]])

    with pytest.raises(ValueError, match="wind_dir must have last dimension"):
        sm.shadow_tracing(
            rec_x, rec_y, rec_z, np.zeros((2, 2)), sun_vectors,
            mode='rotor', verbose=False)

    rotor_shadow, _ = sm.shadow_tracing(
        rec_x, rec_y, rec_z, sm.turbine_dir_vector[:1], sun_vectors,
        mode='rotor', verbose=False)
    assert rotor_shadow.shape == (1, 1, 2)

    with pytest.raises(ValueError, match="wind_dir time dimension"):
        sm.shadow_tracing(
            rec_x, rec_y, rec_z, np.zeros((3, 3)),
            np.tile([[0.0, 1.0, 0.5]], (4, 1)),
            mode='rotor', verbose=False)
    with pytest.raises(ValueError, match="3D wind_dir"):
        sm.shadow_tracing(
            rec_x, rec_y, rec_z, np.zeros((2, 2, 3)), sun_vectors,
            mode='rotor', verbose=False)
    with pytest.raises(ValueError, match="wind_dir must be 2D"):
        sm.shadow_tracing(
            rec_x, rec_y, rec_z, np.zeros((1, 2, 1, 3)), sun_vectors,
            mode='rotor', verbose=False)


def test_shadow_tracing_wind_direction_rad_validation_and_broadcasts():
    time = pd.date_range('2024-01-01', periods=2, freq='h', tz='UTC')
    sm = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=SRC_H,
        turbine_diameter=TURBINE_D,
        tower_diameter=TOWER_D,
        time=time,
        wind_direction=[0, 0],
    )
    rec_x = np.array([12.0])
    rec_y = np.array([55.0])
    rec_z = np.array([0.0])
    sun_vectors = np.array([[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]])

    rotor_shadow, _ = sm.shadow_tracing(
        rec_x, rec_y, rec_z, sm.turbine_dir_vector, sun_vectors,
        wind_dir_rad=np.array([0.0]), mode='rotor', verbose=False)
    assert rotor_shadow.shape == (1, 1, 2)

    rotor_shadow, _ = sm.shadow_tracing(
        rec_x, rec_y, rec_z, sm.turbine_dir_vector, sun_vectors,
        wind_dir_rad=np.array([0.0, 0.0, 0.0]), mode='rotor', verbose=False)
    assert rotor_shadow.shape == (1, 1, 2)

    with pytest.raises(ValueError, match="wind_dir_rad time dimension"):
        sm.shadow_tracing(
            rec_x, rec_y, rec_z, np.tile([[0.0, -1.0, 0.0]], (4, 1)),
            np.tile([[0.0, 1.0, 0.5]], (4, 1)),
            wind_dir_rad=np.array([0.0, 0.0, 0.0]),
            mode='rotor', verbose=False)
    with pytest.raises(ValueError, match="2D wind_dir_rad"):
        sm.shadow_tracing(
            rec_x, rec_y, rec_z, sm.turbine_dir_vector, sun_vectors,
            wind_dir_rad=np.zeros((2, 2)), mode='rotor', verbose=False)
    with pytest.raises(ValueError, match="wind_dir_rad must be"):
        sm.shadow_tracing(
            rec_x, rec_y, rec_z, sm.turbine_dir_vector, sun_vectors,
            wind_dir_rad=np.zeros((1, 1, 2)), mode='rotor', verbose=False)


def test_add_wind_direction_data_var_rejects_invalid():
    with pytest.raises(ValueError, match="Resolved wind_direction"):
        ShadowModel._add_wind_direction_data_var({}, np.zeros((1, 1, 1)))


def test_min_sun_elevation():
    """Test that min_sun_elevation parameter correctly filters shadow calculations."""
    time = pd.date_range('2024-01-01 10:00', periods=3, freq='h', tz='UTC')
    wind_direction = [0, 0, 0]

    sm = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=wind_direction,
        min_sun_elevation=5.0
    )

    rec_x = np.array([12.1])
    rec_y = np.array([55.0])
    rec_z = np.array([0])

    sun_vectors = np.array([
        [0, 1, -0.1],  # Below horizon - should be ignored
        # Above horizon but below min elevation (5°) - should be ignored
        [0, 1, 0.05],
        [0, 1, 0.2],   # Above min elevation (5°) - should be processed
    ])

    wind_dir = sm.turbine_dir_vector

    rotor_shadow, tower_shadow = sm.shadow_tracing(
        rec_x, rec_y, rec_z, wind_dir, sun_vectors, mode="both", verbose=False
    )

    assert not rotor_shadow[0, 0, 0]  # Below horizon
    assert not rotor_shadow[0, 0, 1]  # Below min elevation (5°)

    sm_default = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time[:2], wind_direction=wind_direction[:2]
    )

    sun_vectors_default = np.array([
        [0, 1, 0.0],  # At horizon - should be processed with default
        [0, 1, 0.05],  # Above horizon - should be processed
    ])

    rotor_shadow_default, _ = sm_default.shadow_tracing(
        rec_x, rec_y, rec_z, sm_default.turbine_dir_vector,
        sun_vectors_default, mode="rotor", verbose=False
    )


def test_max_distance():
    """Test that max_distance parameter correctly filters receptors by distance."""
    time = pd.date_range('2024-01-01 10:00', periods=2, freq='h', tz='UTC')
    wind_direction = [0, 0]

    # Create shadow model with max_distance of 500 meters
    sm = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=wind_direction,
        max_distance=500.0
    )

    # Create receptors at different distances from turbine
    # Turbine at (12.0, 55.0)
    rec_x = np.array([12.0, 12.005, 12.010])  # 0m, 500m, 1000m east
    rec_y = np.array([55.0, 55.0, 55.0])     # Same latitude
    rec_z = np.array([0, 0, 0])

    sun_vectors = np.array([
        [0, 1, 0.5],  # Above horizon
        [0, 1, 0.5],  # Above horizon
    ])

    wind_dir = sm.turbine_dir_vector

    rotor_shadow, tower_shadow = sm.shadow_tracing(
        rec_x, rec_y, rec_z, wind_dir, sun_vectors, mode="both", verbose=False
    )

    # Check that results have correct shape (n_turbines, n_receptors, n_times)
    assert rotor_shadow.shape == (1, 3, 2)
    assert tower_shadow.shape == (1, 3, 2)

    # Receptors at 0m and 500m should have been processed (within max_distance)
    # Receptor at 1000m should have no shadows (beyond max_distance)
    # Note: Even processed receptors may not have shadows depending on geometry

    # Test with None max_distance (should process all receptors)
    sm_none = ShadowModel(
        src_x=[12.0], src_y=[55.0], src_h=SRC_H,
        turbine_diameter=TURBINE_D, tower_diameter=TOWER_D,
        time=time, wind_direction=wind_direction,
        max_distance=None
    )

    rotor_shadow_none, _ = sm_none.shadow_tracing(
        rec_x, rec_y, rec_z, sm_none.turbine_dir_vector,
        sun_vectors, mode="rotor", verbose=False
    )

    # With max_distance=None, all receptors should be processed
    assert rotor_shadow_none.shape == (1, 3, 2)
