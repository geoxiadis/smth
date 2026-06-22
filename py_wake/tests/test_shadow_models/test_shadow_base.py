import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import os
import xarray as xr
from py_wake.flow_map import Points, XYGrid
from py_wake.shadow_models.shadow import ShadowModel
from py_wake.shadow_models.shadow_base import ShadowMap, ShadowResult
import matplotlib.dates as mdates
from unittest.mock import patch
import inspect
from zoneinfo import ZoneInfoNotFoundError


@pytest.fixture
def sample_shadow_map_dataset():
    time = pd.date_range("2024-01-01", "2024-01-02", freq="1h", tz="UTC")
    sm = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=100,
        turbine_diameter=90,
        tower_diameter=5,
        time=time,
        wind_direction=[0] * 25,
    )
    grid = XYGrid(
        x=np.linspace(11.99, 12.01, 3), y=np.linspace(54.99, 55.01, 3), h=0
    )
    return sm.run(grid=grid)


@pytest.fixture
def sample_shadow_result_dataset():
    time = pd.date_range("2024-01-01", "2024-01-02", freq="1h", tz="UTC")
    sm = ShadowModel(
        src_x=[12.0],
        src_y=[55.0],
        src_h=100,
        turbine_diameter=90,
        tower_diameter=5,
        time=time,
        wind_direction=[0] * 25,
    )
    grid = Points(x=[12.0, 12.01], y=[55.1, 55.0], h=[0, 0])
    return sm.run(grid=grid)


@pytest.fixture
def complex_shadow_dataset():
    times = pd.date_range("2024-01-01", "2024-01-02", freq="1h", tz="UTC")
    x = np.linspace(10, 12, 10)
    y = np.linspace(50, 51, 10)
    wt = np.array([0, 1])
    rec = np.array([0, 1, 2])
    shadow_shape = (len(wt), len(rec), len(times))
    rotor_shadow = np.random.randint(0, 2, size=shadow_shape).astype(bool)
    tower_shadow = np.random.randint(0, 2, size=shadow_shape).astype(bool)
    combined_shadow = np.logical_or(rotor_shadow, tower_shadow)
    rotor_shadow[0, 0, 0] = np.nan
    tower_shadow[1, 1, 1] = np.nan
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], rotor_shadow),
            "Tower shadow": (["wt", "rec", "time"], tower_shadow),
            "Combined shadow": (["wt", "rec", "time"], combined_shadow),
            "rec_x": (["rec"], [10, 11.1, 11.2]),
            "rec_y": (["rec"], [50, 50.1, 51.2]),
            "src_x": (["wt"], [10.2, 11.8]),
            "src_y": (["wt"], [50.2, 50.8]),
            "x": (["x"], x),
            "y": (["y"], y)
        },
        coords={
            "wt": wt,
            "rec": rec,
            "time": times,
        },
        attrs={
            "mode": "both",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    return ds


def test_shadow_map_plot(sample_shadow_map_dataset):
    shadow_map = sample_shadow_map_dataset
    fig, ax = shadow_map.plot(show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)
    fig, ax = shadow_map.plot(
        cmap="RdBu",
        figsize=(8, 6),
        title="Custom Title",
        colorbar_label="Hours",
        source_style={"color": "red", "marker": "*"},
        levels=[0, 2, 4, 8, 16],
        show=False,
    )
    plt.close(fig)


def test_shadow_map_animation(sample_shadow_map_dataset):
    shadow_map = sample_shadow_map_dataset
    anim = shadow_map.animate(
        step=2,
        interval=200,
        title="Test Animation",
        legend_position="bottom",
        show=False,
    )
    assert anim is not None
    plt.close("all")
    custom_colors = [
        (0, 0, 0, 0),
        (1, 0, 0, 0.5),
        (0, 0, 1, 0.5),
        (0.5, 0, 0.5, 0.5),
    ]
    anim = shadow_map.animate(
        custom_colors=custom_colors, legend_position="right", show=False
    )
    plt.close("all")


def test_shadow_map_combined_animation_and_show(sample_shadow_map_dataset):
    ds = sample_shadow_map_dataset.copy(deep=True)
    ds.attrs["mode"] = "combined"
    ds["Combined shadow"] = (
        ds["Rotor shadow"].fillna(False) | ds["Tower shadow"].fillna(False)
    )
    ds["Combined shadow"][:, :, :, 0] = True
    shadow_map = ShadowMap(ds)

    with patch("matplotlib.pyplot.show") as show:
        anim = shadow_map.animate(show=True)
    show.assert_called_once()
    anim._func(0)
    plt.close("all")


def test_shadow_result_plot(sample_shadow_result_dataset):
    shadow_result = sample_shadow_result_dataset
    fig, ax = shadow_result.plot(show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)
    fig, ax = shadow_result.plot(
        figsize=(12, 8),
        title="Custom Result Plot",
        cmap="plasma",
        marker_size=150,
        annotate=True,
        aspect="auto",
        show=False,
    )
    plt.close(fig)


def test_shadow_result_calendar(sample_shadow_result_dataset):
    shadow_result = sample_shadow_result_dataset
    fig, axes = shadow_result.calendar(
        hour_range=(6, 18),
        title="Shadow Calendar",
        receptor_labels=["R1", "R2"],
        turbine_labels=["T1"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_shadow_result_combined_calendar(sample_shadow_result_dataset):
    ds = sample_shadow_result_dataset.copy(deep=True)
    ds.attrs["mode"] = "combined"
    ds["Combined shadow"] = (
        ds["Rotor shadow"].fillna(False) | ds["Tower shadow"].fillna(False)
    )
    ds["Combined shadow"][:, :, 0] = True
    shadow_result = ShadowResult(ds)
    fig, axes = shadow_result.calendar(show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_plot_save_functionality(
    sample_shadow_map_dataset, sample_shadow_result_dataset, tmp_path
):
    shadow_map = sample_shadow_map_dataset
    shadow_result = sample_shadow_result_dataset
    map_path = tmp_path / "shadow_map.png"
    shadow_map.plot(save_path=str(map_path), show=False)
    assert map_path.exists()
    result_path = tmp_path / "shadow_result.png"
    shadow_result.plot(save_path=str(result_path), show=False)
    assert result_path.exists()
    calendar_path = tmp_path / "shadow_calendar.png"
    shadow_result.calendar(save_path=str(calendar_path), show=False)
    assert calendar_path.exists()


def test_animation_save(sample_shadow_map_dataset, tmp_path):
    shadow_map = sample_shadow_map_dataset
    gif_path = tmp_path / "animation.gif"
    shadow_map.animate(save_path=str(gif_path), show=False)
    assert gif_path.exists()
    try:
        mp4_path = tmp_path / "animation.mp4"
        shadow_map.animate(save_path=str(mp4_path), show=False)
        assert mp4_path.exists()
    except Exception as e:
        pytest.skip(f"MP4 saving failed (possibly no ffmpeg): {e}")


def test_shadow_base_invalid_mode():
    ds = xr.Dataset(attrs={"mode": "invalid", "freq": 1.0})
    shadow = ShadowMap(ds)
    with pytest.raises(ValueError):
        shadow.get_combined_shadow()


def test_shadow_base_attribute_error():
    ds = xr.Dataset(attrs={"mode": "both", "freq": 1.0})
    shadow = ShadowMap(ds)
    with pytest.raises(AttributeError):
        shadow.nonexistent_attribute


def test_shadow_base_item_access():
    ds = xr.Dataset(
        data_vars={"test": (["x"], [1, 2, 3])},
        attrs={"mode": "both", "freq": 1.0},
    )
    shadow = ShadowMap(ds)
    assert np.array_equal(shadow["test"], [1, 2, 3])


def test_shadow_map_save_load(sample_shadow_map_dataset, tmp_path):
    shadow_map = ShadowMap(sample_shadow_map_dataset)
    save_path = tmp_path / "test_shadow"
    shadow_map.save(save_path)
    assert os.path.exists(save_path.with_suffix(".nc"))
    with pytest.raises(FileNotFoundError):
        ShadowMap.load(tmp_path / "nonexistent.nc")
    loaded_map = ShadowMap.load(save_path.with_suffix(".nc"))
    assert isinstance(loaded_map, ShadowMap)
    assert loaded_map.mode == shadow_map.mode


def test_shadow_map_plot_edge_cases(sample_shadow_map_dataset):
    shadow_map = ShadowMap(sample_shadow_map_dataset)
    fig, ax = shadow_map.plot(levels=[0, 1, 2, 4, 8], show=False)
    plt.close(fig)


def test_shadow_map_repr_collapse_and_plot_show(sample_shadow_map_dataset):
    shadow_map = ShadowMap(sample_shadow_map_dataset.copy(deep=True))
    assert "xarray.ShadowMap" in repr(shadow_map)

    collapsed = ShadowMap(sample_shadow_map_dataset.copy(deep=True)).collapse_shadow_type()
    assert collapsed.mode == "combined"
    assert "Combined shadow" in collapsed.dataset
    assert "Rotor shadow" not in collapsed.dataset

    ds = sample_shadow_map_dataset.copy(deep=True)
    ds["Rotor shadow"][:] = True
    ds["Tower shadow"][:] = True
    shadow_with_many_hours = ShadowMap(ds)
    with patch("matplotlib.pyplot.show") as show:
        fig, ax = shadow_with_many_hours.plot(show=True)
    show.assert_called_once()
    plt.close(fig)


def test_shadow_result_plot_edge_cases(sample_shadow_result_dataset):
    shadow_result = ShadowResult(sample_shadow_result_dataset)
    ds_nan = sample_shadow_result_dataset.copy()
    ds_nan["src_x"] = ds_nan["src_x"].where(ds_nan["src_x"] < 0)
    shadow_result_nan = ShadowResult(ds_nan)
    fig, ax = shadow_result_nan.plot(show=False)
    plt.close(fig)
    fig, ax = shadow_result.plot(
        title="Custom Title",
        marker_size=200,
        annotate=True,
        aspect="auto",
        show=False,
    )
    plt.close(fig)


def test_shadow_result_calendar_edge_cases(sample_shadow_result_dataset):
    shadow_result = ShadowResult(sample_shadow_result_dataset)
    fig, axes = shadow_result.calendar(hour_range=(6, 18), show=False)
    plt.close(fig)
    fig, axes = shadow_result.calendar(
        receptor_labels=["R1", "R2"], turbine_labels=["T1"], show=False
    )
    plt.close(fig)
    with pytest.raises(
        ValueError,
        match="Length of receptor_labels must match number of receptors",
    ):
        shadow_result.calendar(receptor_labels=["R1"], show=False)


def test_shadow_result_plot_show(sample_shadow_result_dataset):
    shadow_result = ShadowResult(sample_shadow_result_dataset)
    with patch("matplotlib.pyplot.show") as show:
        fig, ax = shadow_result.plot(show=True)
    show.assert_called_once()
    plt.close(fig)


def test_shadow_result_calendar_long_date_range_and_show():
    times = pd.DatetimeIndex(
        ["2024-01-01 12:00", "2024-04-01 12:00", "2024-07-01 12:00"],
        tz="UTC",
    )
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.ones((1, 1, 3), dtype=bool)),
            "rec_x": (["rec"], [12.0]),
            "rec_y": (["rec"], [55.0]),
            "src_x": (["wt"], [12.0]),
            "src_y": (["wt"], [55.0]),
        },
        coords={"wt": [0], "rec": [0], "time": times},
        attrs={"mode": "rotor", "freq": 1.0, "tz": "UTC"},
    )
    shadow_result = ShadowResult(ds)
    with patch("matplotlib.pyplot.show") as show:
        fig, axes = shadow_result.calendar(show=True)
    show.assert_called_once()
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_collapse_methods_edge_cases(sample_shadow_map_dataset):
    ds_no_time = sample_shadow_map_dataset.isel(time=slice(0, 0))
    shadow_no_time = ShadowMap(ds_no_time)
    shadow_no_time.collapse_time()
    assert shadow_no_time.dataset.sizes["time"] == 1
    ds_single_wt = sample_shadow_map_dataset.isel(wt=[0])
    shadow_single_wt = ShadowMap(ds_single_wt)
    result = shadow_single_wt.collapse_turbine()
    assert result.dataset.sizes["wt"] == 1
    ds_with_int_var = sample_shadow_map_dataset.copy(deep=True)
    int_data = np.random.randint(0, 100, size=ds_with_int_var.wt.shape)
    ds_with_int_var['int_var_wt_dependent'] = xr.DataArray(
        int_data, dims=['wt'], coords={'wt': ds_with_int_var.wt})
    shadow_with_int_var = ShadowMap(ds_with_int_var)
    collapsed_with_int_var = shadow_with_int_var.collapse_turbine()
    assert collapsed_with_int_var.dataset.sizes["wt"] == 1
    assert np.isnan(
        collapsed_with_int_var.dataset['int_var_wt_dependent'].values).all()
    assert collapsed_with_int_var.dataset['int_var_wt_dependent'].dtype == float
    ds_invalid = sample_shadow_map_dataset.copy()
    ds_invalid.attrs["mode"] = "invalid"
    shadow_invalid = ShadowMap(ds_invalid)
    with pytest.raises(ValueError):
        shadow_invalid.collapse_shadow_type()


def test_sel_method_error_handling(sample_shadow_map_dataset):
    shadow_map = ShadowMap(sample_shadow_map_dataset)
    with pytest.raises(KeyError):
        shadow_map.sel(invalid_coord=1)
    with pytest.raises(Exception):
        shadow_map.sel(time='2025-01-01')


def test_get_shadow_data_errors(sample_shadow_map_dataset):
    ds = sample_shadow_map_dataset.copy()
    ds = ds.drop_vars(['Rotor shadow'])
    shadow = ShadowMap(ds)
    with pytest.raises(KeyError):
        shadow.get_rotor_shadow()


def test_shadow_type_conversion(complex_shadow_dataset):
    for mode in ['combined', 'both', 'rotor', 'tower']:
        ds = complex_shadow_dataset.copy()
        ds.attrs['mode'] = mode
        shadow_mode = ShadowMap(ds)
        combined = shadow_mode.get_combined_shadow()
        assert isinstance(combined, xr.DataArray)


def test_shadow_data_handling(sample_shadow_map_dataset):
    ds = sample_shadow_map_dataset.copy()
    ds['Rotor shadow'][:, 0, 0] = np.nan
    shadow_nan = ShadowMap(ds)
    collapsed = shadow_nan.collapse_turbine()
    assert isinstance(collapsed, ShadowMap)


def test_file_operations(sample_shadow_map_dataset, tmp_path):
    shadow = ShadowMap(sample_shadow_map_dataset)
    invalid_path = tmp_path / "nonexistent" / "test.nc"
    with pytest.raises(Exception):
        shadow.save(invalid_path)
    with pytest.raises(FileNotFoundError):
        ShadowMap.load(tmp_path / "nonexistent.nc")
    ds_problem_time = sample_shadow_map_dataset.copy(deep=True)
    time_strs = [dt.strftime('%Y-%m-%dT%H:%M:%S')
                 for dt in pd.to_datetime(ds_problem_time.time.values)]
    ds_problem_time = ds_problem_time.assign_coords(time=time_strs)
    shadow_problem_time = ShadowMap(ds_problem_time)
    save_path_problem_time = tmp_path / "test_shadow_problem_time.nc"
    shadow_problem_time.save(save_path_problem_time)
    assert save_path_problem_time.exists()
    loaded_problem_time_map = ShadowMap.load(save_path_problem_time)
    assert isinstance(loaded_problem_time_map, ShadowMap)
    ds_unparseable_time = sample_shadow_map_dataset.copy(deep=True)
    ds_unparseable_time.coords['time'] = (
        ('time'), ['not a time string'] * len(ds_unparseable_time.time))
    shadow_unparseable_time = ShadowMap(ds_unparseable_time)
    save_path_unparseable_time = tmp_path / "test_shadow_unparseable_time.nc"
    with pytest.raises(ValueError, match="Cannot convert time data to a serializable format"):
        shadow_unparseable_time.save(save_path_unparseable_time)


def test_animation_error_handling(sample_shadow_map_dataset):
    shadow = ShadowMap(sample_shadow_map_dataset)
    with pytest.raises(Exception):
        shadow.animate(save_path="/invalid/path/anim.gif")


def test_animate_time_processing_errors(sample_shadow_map_dataset):
    ds_renamed_time = sample_shadow_map_dataset.copy(deep=True)
    if 'time' in ds_renamed_time.coords:
        ds_renamed_time = ds_renamed_time.rename({'time': 'time_actual'})
    elif 'time' in ds_renamed_time.dims and 'time' not in ds_renamed_time.coords:
        ds_renamed_time = ds_renamed_time.rename_dims({'time': 'time_actual'})
    shadow_renamed_time = ShadowMap(ds_renamed_time)
    with pytest.raises(ValueError, match=r"Error processing time data: ('time'|\"No variable named 'time'. Variables on the dataset include .*\")"):
        shadow_renamed_time.animate(show=False)
    plt.close('all')

    ds_bad_time_values = sample_shadow_map_dataset.copy(deep=True)
    bad_times = ["not a datetime string"] * len(ds_bad_time_values.time)
    ds_bad_time_values['time'] = (('time'), bad_times)
    shadow_bad_time_values = ShadowMap(ds_bad_time_values)
    with pytest.raises(ValueError, match=r"Error processing time data: (Unknown datetime string format, unable to parse: not a datetime string|Unknown string format: not a datetime string(?: present at position 0)?)"):
        shadow_bad_time_values.animate(show=False)
    plt.close('all')

    ds_invalid_tz_attr = sample_shadow_map_dataset.copy(deep=True)
    ds_invalid_tz_attr.attrs['tz'] = "Invalid/Timezone"
    shadow_invalid_tz_attr = ShadowMap(ds_invalid_tz_attr)
    with pytest.raises(ValueError, match=r"Error processing time data: 'Invalid/Timezone'"):
        shadow_invalid_tz_attr.animate(show=False)
    plt.close('all')

    with patch(
        "pandas.DatetimeIndex.tz_convert",
        side_effect=ZoneInfoNotFoundError("Missing/Timezone"),
    ):
        with pytest.raises(ValueError, match=r"Error processing time data: 'UTC'"):
            ShadowMap(sample_shadow_map_dataset).animate(show=False)
    plt.close('all')


def test_plot_error_handling(sample_shadow_map_dataset):
    shadow = ShadowMap(sample_shadow_map_dataset)
    with pytest.raises(Exception):
        shadow.plot(save_path="/invalid/path/plot.png", show=False)


def test_calendar_validation(sample_shadow_result_dataset):
    shadow = ShadowResult(sample_shadow_result_dataset)
    with pytest.raises(ValueError):
        shadow.calendar(hour_range=(25, 30), show=False)
    with pytest.raises(ValueError):
        shadow.calendar(hour_range=(18, 6), show=False)


def test_calendar_sunrise_sunset_edge_cases(complex_shadow_dataset):
    ds_template = complex_shadow_dataset.copy(deep=True)
    if len(ds_template.time) < 24:
        new_times = pd.date_range(
            ds_template.time.values[0],
            periods=24,
            freq='H',
            tz='UTC')
        ds_template = ds_template.reindex(
            {'time': new_times}, method='nearest', tolerance='1D')
    ds_extreme_lat = ds_template.copy(deep=True)
    ds_extreme_lat['rec_y'] = (
        ds_extreme_lat.rec.dims, np.full_like(
            ds_extreme_lat.rec_y, 90.0))
    shadow_extreme_lat = ShadowResult(ds_extreme_lat)
    fig, ax = shadow_extreme_lat.calendar(show=False, hour_range=(0, 23))
    plt.close(fig)
    sparse_times = pd.to_datetime([
        '2024-01-01T06:00:00Z', '2024-01-01T06:30:00Z',
        '2024-01-03T18:00:00Z', '2024-01-03T18:30:00Z',
        '2024-01-05T07:00:00Z', '2024-01-05T07:30:00Z'
    ])
    if len(ds_template.time) < len(sparse_times):
        original_times = ds_template.time.values
        padding_needed = len(sparse_times) - len(original_times)
        padded_times = np.concatenate([original_times, pd.to_datetime(
            original_times[-1]) + pd.to_timedelta(np.arange(1, padding_needed + 1), unit='h')])
        ds_template = ds_template.reindex({'time': padded_times}, method=None)
    ds_sparse_events = ds_template.reindex(
        {'time': sparse_times}, method='nearest', tolerance='1D')
    ds_sparse_events = ds_sparse_events.dropna(dim='time', how='all')
    for shadow_var_name in ["Rotor shadow", "Tower shadow", "Combined shadow"]:
        if shadow_var_name in ds_sparse_events:
            ds_sparse_events[shadow_var_name] = ds_sparse_events[shadow_var_name].fillna(
                False).astype(bool)
    if len(ds_sparse_events.time) > 1:
        shadow_sparse = ShadowResult(ds_sparse_events)
        fig, ax = shadow_sparse.calendar(show=False)
        plt.close(fig)
    single_day_times = pd.date_range(
        "2024-07-01T00:00:00Z",
        "2024-07-01T23:59:59Z",
        freq="30min")
    ds_single_day = ds_template.reindex(
        {'time': single_day_times}, method='nearest', tolerance='1D')
    ds_single_day = ds_single_day.dropna(dim='time', how='all')
    for shadow_var_name in ["Rotor shadow", "Tower shadow", "Combined shadow"]:
        if shadow_var_name in ds_single_day:
            ds_single_day[shadow_var_name] = ds_single_day[shadow_var_name].fillna(
                False).astype(bool)
    if len(ds_single_day.time) > 1:
        shadow_single_day = ShadowResult(ds_single_day)
        fig, ax = shadow_single_day.calendar(show=False)
        plt.close(fig)


def test_calendar_sunrise_sunset_continue_coverage(
        sample_shadow_result_dataset):
    times = pd.date_range(
        "2024-03-20 00:00:00", "2024-03-20 23:59:00", freq="1min", tz="UTC"
    )
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.zeros((1, 1, len(times)), dtype=bool)),
            "Tower shadow": (["wt", "rec", "time"], np.zeros((1, 1, len(times)), dtype=bool)),
            "rec_x": (["rec"], [12.0]),
            "rec_y": (["rec"], [55.0]),
            "src_x": (["wt"], [12.0]),
            "src_y": (["wt"], [55.0]),
        },
        coords={
            "wt": [0],
            "rec": [0],
            "time": times,
        },
        attrs={"mode": "both", "freq": 1.0, "tz": "UTC"},
    )
    shadow_result = ShadowResult(ds)
    fig, axes = shadow_result.calendar(hour_range=(10, 14), show=False)
    plt.close(fig)


def test_calendar_xlim_and_xlabel_logic(
        complex_shadow_dataset,
        sample_shadow_result_dataset):
    ds_single_point_time = sample_shadow_result_dataset.isel(time=0)
    shadow_single_date = ShadowResult(ds_single_point_time)
    fig_single, axes_single_list = shadow_single_date.calendar(show=False)
    ax_single = axes_single_list.flatten()[0]
    xlim_single = ax_single.get_xlim()
    single_time_val_dt = pd.to_datetime(
        ds_single_point_time.time.values.item())
    expected_center_date_num = mdates.date2num(single_time_val_dt)
    assert np.isclose(xlim_single[0], expected_center_date_num - 1)
    assert np.isclose(xlim_single[1], expected_center_date_num + 1)
    plt.close(fig_single)
    n_rec_multi_row = 4
    times_multi_row = pd.date_range(
        "2024-01-01", "2024-01-03", freq="1h", tz="UTC")
    ds_multi_row = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(2, n_rec_multi_row, len(times_multi_row))).astype(bool)),
            "Tower shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(2, n_rec_multi_row, len(times_multi_row))).astype(bool)),
            "Combined shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(2, n_rec_multi_row, len(times_multi_row))).astype(bool)),
            "rec_x": (["rec"], np.linspace(10, 11, n_rec_multi_row)),
            "rec_y": (["rec"], np.linspace(50, 51, n_rec_multi_row)),
            "src_x": (["wt"], [10.2, 11.8]),
            "src_y": (["wt"], [50.2, 50.8]),
        },
        coords={
            "wt": [0, 1],
            "rec": np.arange(n_rec_multi_row),
            "time": times_multi_row,
        },
        attrs=complex_shadow_dataset.attrs.copy()
    )
    shadow_multi_row = ShadowResult(ds_multi_row)
    fig_multi_row, axes_multi_row = shadow_multi_row.calendar(show=False)
    assert axes_multi_row[1, 0].get_xlabel() == "Date"
    assert axes_multi_row[0, 2].get_xlabel() == ""
    plt.close(fig_multi_row)


def test_complex_calendar_scenarios(sample_shadow_result_dataset):
    ds_single = sample_shadow_result_dataset.isel(time=slice(0, 1))
    shadow_single = ShadowResult(ds_single)
    fig_s, axes_s = shadow_single.calendar(show=False)
    plt.close(fig_s)
    times_week = pd.date_range(
        '2024-01-01',
        periods=7 * 24,
        freq='h',
        tz='UTC')
    ds_week = sample_shadow_result_dataset.reindex(
        {'time': times_week}, method='nearest')
    ds_week.attrs.update(sample_shadow_result_dataset.attrs)
    shadow_week = ShadowResult(ds_week)
    fig_w, axes_w = shadow_week.calendar(show=False)
    plt.close(fig_w)
    times_month = pd.date_range(
        '2024-01-01',
        periods=60 * 24,
        freq='h',
        tz='UTC')
    ds_month = sample_shadow_result_dataset.reindex(
        {'time': times_month}, method='nearest')
    ds_month.attrs.update(sample_shadow_result_dataset.attrs)
    shadow_month = ShadowResult(ds_month)
    fig_m, axes_m = shadow_month.calendar(show=False)
    plt.close(fig_m)


def test_complex_data_operations(complex_shadow_dataset):
    ds = complex_shadow_dataset.copy()
    ds = ds.drop_vars(['Combined shadow', "Rotor shadow"])
    shadow_missing = ShadowMap(ds)
    with pytest.raises(KeyError):
        shadow_missing.get_combined_shadow()
    ds.attrs['mode'] = 'invalid'
    shadow_invalid = ShadowMap(ds)
    with pytest.raises(ValueError):
        shadow_invalid.get_combined_shadow()


def test_visualization_edge_cases(sample_shadow_result_dataset):
    shadow = ShadowResult(sample_shadow_result_dataset)
    ds_nan_rec = sample_shadow_result_dataset.copy(deep=True)
    ds_nan_rec['rec_x'] = ds_nan_rec['rec_x'].where(ds_nan_rec['rec_x'] < 0)
    shadow_nan_rec = ShadowResult(ds_nan_rec)
    fig, ax = shadow_nan_rec.plot(show=False, annotate=True)
    plt.close(fig)
    ds_nan_src = sample_shadow_result_dataset.copy(deep=True)
    ds_nan_src['src_x'] = ds_nan_src['src_x'].where(ds_nan_src['src_x'] < 0)
    shadow_nan_src = ShadowResult(ds_nan_src)
    fig, ax = shadow_nan_src.plot(show=False, annotate=True)
    plt.close(fig)
    with pytest.raises(Exception):
        shadow.plot(save_path='/invalid/path/test.png', show=False)


def test_legend_handling(sample_shadow_result_dataset):
    shadow = ShadowResult(sample_shadow_result_dataset)
    fig, axes = shadow.calendar(
        turbine_labels=['T1'],
        receptor_labels=['R1', 'R2'],
        show=False
    )
    plt.close(fig)
    ds_no_shadow = sample_shadow_result_dataset.copy(deep=True)
    if "Rotor shadow" in ds_no_shadow:
        ds_no_shadow["Rotor shadow"] = xr.full_like(
            ds_no_shadow["Rotor shadow"], False)
    if "Tower shadow" in ds_no_shadow:
        ds_no_shadow["Tower shadow"] = xr.full_like(
            ds_no_shadow["Tower shadow"], False)
    if "Combined shadow" not in ds_no_shadow and "Rotor shadow" in ds_no_shadow:
        rotor_shadow = ds_no_shadow["Rotor shadow"]
        ds_no_shadow["Combined shadow"] = xr.zeros_like(rotor_shadow)
    elif "Combined shadow" not in ds_no_shadow and "Tower shadow" in ds_no_shadow:
        tower_shadow = ds_no_shadow["Tower shadow"]
        ds_no_shadow["Combined shadow"] = xr.zeros_like(tower_shadow)
    elif "Combined shadow" in ds_no_shadow:
        ds_no_shadow["Combined shadow"] = xr.full_like(
            ds_no_shadow["Combined shadow"], False)
    shadow_no_legend = ShadowResult(ds_no_shadow)
    fig_no_legend, ax_no_legend = shadow_no_legend.calendar(show=False)
    assert len(fig_no_legend.legends) == 0
    plt.close(fig_no_legend)
    ds = sample_shadow_result_dataset.copy()
    ds = ds.isel(wt=slice(0, 1))
    shadow_single = ShadowResult(ds)
    fig, axes = shadow_single.calendar(show=False)
    plt.close(fig)
    fig, ax = shadow.plot(
        marker_size=200,
        annotate=True,
        show=False
    )
    plt.close(fig)


def test_error_conditions(sample_shadow_result_dataset):
    ds_missing_rec_x = sample_shadow_result_dataset.copy(deep=True)
    ds_missing_rec_x = ds_missing_rec_x.drop_vars(['rec_x'])
    shadow_invalid_coords = ShadowResult(ds_missing_rec_x)
    with pytest.raises(KeyError):
        shadow_invalid_coords.plot(show=False)


def test_sel_function(sample_shadow_map_dataset):
    shadow = ShadowMap(sample_shadow_map_dataset)
    selected = shadow.sel(time="2024-01-01 12:00")
    assert len(selected.dataset.time) == 1
    assert set(selected.dataset.dims) == set(shadow.dataset.dims)
    multi_selected = shadow.sel(
        time=slice("2024-01-01 12:00", "2024-01-01 14:00"),
        wt=0
    )
    assert len(multi_selected.dataset.time) == 3
    assert len(multi_selected.dataset.wt) == 1
    assert set(multi_selected.dataset.dims) == set(shadow.dataset.dims)


def test_animate_with_explicit_and_default_axes(sample_shadow_map_dataset):
    shadow_map = ShadowMap(sample_shadow_map_dataset)

    # Explicit axes are used as-is
    fig, ax = plt.subplots()
    anim = shadow_map.animate(ax=ax, show=False)
    assert anim is not None
    plt.close(fig)

    # When no axes are given, the current axes (plt.gca()) are used
    fig = plt.figure()
    anim = shadow_map.animate(show=False)
    assert anim is not None
    plt.close(fig)


def test_legend_with_mixed_types():
    def extract_number(x):
        return int(x.split()[-1]) if x.split()[-1].isdigit() else x
    assert extract_number("Turbine 2") == 2
    assert extract_number("Turbine 10") == 10
    assert extract_number("Turbine A") == "Turbine A"

    def safe_sort_key(x):
        val = extract_number(x)
        return (0, val) if isinstance(val, int) else (1, val)
    test_keys = ["Turbine 10", "Turbine 2", "Turbine A"]
    sorted_keys = sorted(test_keys, key=safe_sort_key)
    assert sorted_keys[0] == "Turbine 2"
    assert sorted_keys[1] == "Turbine 10"
    assert sorted_keys[2] == "Turbine A"
    test_empty = []
    assert sorted(test_empty, key=safe_sort_key) == []
    test_single = ["Turbine X"]
    assert sorted(test_single, key=safe_sort_key) == ["Turbine X"]
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    ds_simple = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(2, 2, len(times))).astype(bool)),
            "Tower shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(2, 2, len(times))).astype(bool)),
            "rec_x": (["rec"], [12.0, 12.01]),
            "rec_y": (["rec"], [55.1, 55.0]),
            "src_x": (["wt"], [12.0, 12.1]),
            "src_y": (["wt"], [55.0, 55.1]),
        },
        coords={
            "wt": [0, 1],
            "rec": [0, 1],
            "time": times,
        },
        attrs={
            "mode": "both",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow_simple = ShadowResult(ds_simple)
    fig, axes = shadow_simple.calendar(
        turbine_labels=["Turbine 2", "Turbine 3"],
        show=False
    )
    plt.close(fig)


def test_calendar_empty_legend(sample_shadow_result_dataset):
    ds = sample_shadow_result_dataset.copy(deep=True)
    ds_zeros = ds.copy(deep=True)
    ds_zeros["Rotor shadow"] = xr.zeros_like(ds_zeros["Rotor shadow"])
    ds_zeros["Tower shadow"] = xr.zeros_like(ds_zeros["Tower shadow"])
    if "Combined shadow" in ds_zeros:
        ds_zeros["Combined shadow"] = xr.zeros_like(
            ds_zeros["Combined shadow"])
    shadow_zeros = ShadowResult(ds_zeros)
    fig, axes = shadow_zeros.calendar(show=False)
    plt.close(fig)


def test_solar_position_extreme_cases(complex_shadow_dataset):
    ds = complex_shadow_dataset.copy(deep=True)
    ds_south_pole = ds.copy(deep=True)
    ds_south_pole["rec_y"] = ds_south_pole["rec_y"] * 0 - 85.0
    shadow_south_pole = ShadowResult(ds_south_pole)
    fig, axes = shadow_south_pole.calendar(show=False)
    plt.close(fig)
    if len(ds.time) > 0:
        ds_single_time = ds.isel(time=slice(0, 1))
        shadow_single_time = ShadowResult(ds_single_time)
        fig, axes = shadow_single_time.calendar(show=False)
        plt.close(fig)


def test_calendar_segmentation(complex_shadow_dataset):
    ds = complex_shadow_dataset.copy(deep=True)
    if len(ds.time) >= 3:
        times = ds.time.values
        sparse_indices = [0, len(times) // 2, len(times) - 1]
        ds_sparse = ds.isel(time=sparse_indices)
        if "Rotor shadow" in ds_sparse:
            ds_sparse["Rotor shadow"] = ds_sparse["Rotor shadow"].astype(
                bool) | True
        if "Tower shadow" in ds_sparse:
            ds_sparse["Tower shadow"] = ds_sparse["Tower shadow"].astype(
                bool) | True
        shadow_sparse = ShadowResult(ds_sparse)
        fig, axes = shadow_sparse.calendar(show=False)
        plt.close(fig)


def test_integer_var_handling(sample_shadow_map_dataset):
    ds = sample_shadow_map_dataset.copy(deep=True)
    wt_size = ds.sizes["wt"]
    ds["int_var1"] = xr.DataArray(
        np.ones(
            wt_size,
            dtype=np.int32),
        dims=["wt"])
    ds["int_var2"] = xr.DataArray(
        np.ones(
            (wt_size, 2), dtype=np.int64), dims=[
            "wt", "dummy"])
    shadow = ShadowMap(ds)
    collapsed = shadow.collapse_turbine()
    assert np.isnan(collapsed.dataset["int_var1"].values).all()
    assert collapsed.dataset["int_var1"].dtype == float
    assert np.isnan(collapsed.dataset["int_var2"].values).all()
    assert collapsed.dataset["int_var2"].dtype == float


def test_collapse_turbine_integer_multidim(sample_shadow_map_dataset):
    ds = sample_shadow_map_dataset.copy(deep=True)
    wt_size = ds.sizes["wt"]
    rec_size = 3
    time_size = 2
    ds["int_var_multidim"] = xr.DataArray(
        np.ones((wt_size, rec_size, time_size), dtype=np.int32),
        dims=["wt", "rec_test", "time_test"],
        coords={
            "wt": ds.wt,
            "rec_test": np.arange(rec_size),
            "time_test": np.arange(time_size)
        }
    )
    shadow = ShadowMap(ds)
    collapsed = shadow.collapse_turbine()
    assert collapsed.dataset["int_var_multidim"].dtype == float
    assert np.isnan(collapsed.dataset["int_var_multidim"].values).all()


def test_plot_with_nan_turbine_coords(sample_shadow_result_dataset):
    ds = sample_shadow_result_dataset.copy(deep=True)
    ds["src_x"] = ds["src_x"].astype(float)
    ds["src_y"] = ds["src_y"].astype(float)
    ds["src_x"][0] = np.nan
    ds["src_y"][0] = 55.0
    shadow_result = ShadowResult(ds)
    fig, ax = shadow_result.plot(annotate=True, show=False)
    plt.close(fig)


def test_calendar_with_nan_altitude(complex_shadow_dataset):
    ds = complex_shadow_dataset.copy(deep=True)
    ds["rec_y"] = ds["rec_y"] * 0 + 89.9
    shadow_result = ShadowResult(ds)
    fig, axes = shadow_result.calendar(show=False)
    plt.close(fig)


def test_calendar_sunrise_sunset_with_nans(complex_shadow_dataset):
    ds = complex_shadow_dataset.copy(deep=True)
    if len(ds.time) < 48:
        new_times = pd.date_range(
            ds.time.values[0],
            periods=48,
            freq='30min',
            tz='UTC')
        ds = ds.reindex({'time': new_times}, method='nearest', tolerance='1D')
    times = pd.date_range('2024-12-21', '2024-12-22', freq='30min', tz='UTC')
    ds = ds.reindex({'time': times}, method='nearest', tolerance='1D')
    ds["rec_y"] = ds["rec_y"] * 0 + 88.0
    for var_name in ["Rotor shadow", "Tower shadow", "Combined shadow"]:
        if var_name in ds:
            ds[var_name] = ds[var_name].fillna(0).astype(bool)
    shadow_result = ShadowResult(ds)
    fig, axes = shadow_result.calendar(show=False)
    plt.close(fig)


def test_calendar_single_date_xlim(sample_shadow_result_dataset):
    ds = sample_shadow_result_dataset.copy(deep=True)
    # single_time = ds.time.values[0]
    ds_single_time = ds.isel(time=slice(0, 1))
    for var_name in ["Rotor shadow", "Tower shadow", "Combined shadow"]:
        if var_name in ds_single_time:
            ds_single_time[var_name] = ds_single_time[var_name].fillna(
                False).astype(bool)
    shadow_result = ShadowResult(ds_single_time)
    fig, axes = shadow_result.calendar(show=False)
    ax = axes.flatten()[0]
    xlim = ax.get_xlim()
    assert xlim[1] > xlim[0], "XLim should have a positive range even with single date"
    plt.close(fig)


def test_calendar_bottom_row_xlabel(complex_shadow_dataset):
    ds = complex_shadow_dataset.copy(deep=True)
    n_rec = 7
    new_rec_x = np.linspace(10, 12, n_rec)
    new_rec_y = np.linspace(50, 51, n_rec)
    ds_multi_row = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"],
                             np.random.randint(0, 2, size=(len(ds.wt), n_rec, len(ds.time))).astype(bool)),
            "Tower shadow": (["wt", "rec", "time"],
                             np.random.randint(0, 2, size=(len(ds.wt), n_rec, len(ds.time))).astype(bool)),
            "Combined shadow": (["wt", "rec", "time"],
                                np.random.randint(0, 2, size=(len(ds.wt), n_rec, len(ds.time))).astype(bool)),
            "rec_x": (["rec"], new_rec_x),
            "rec_y": (["rec"], new_rec_y),
            "src_x": (["wt"], ds["src_x"].values),
            "src_y": (["wt"], ds["src_y"].values),
        },
        coords={
            "wt": ds.wt,
            "rec": np.arange(n_rec),
            "time": ds.time,
        },
        attrs=ds.attrs.copy()
    )
    shadow_result = ShadowResult(ds_multi_row)
    fig, axes = shadow_result.calendar(show=False)
    n_rows = axes.shape[0]
    for row_idx in range(n_rows):
        for col_idx in range(axes.shape[1]):
            if row_idx == n_rows - 1:
                assert axes[row_idx, col_idx].get_xlabel(
                ) == "Date" or not axes[row_idx, col_idx].get_visible()
            else:
                if axes[row_idx, col_idx].get_visible():
                    assert axes[row_idx, col_idx].get_xlabel() == ""
    plt.close(fig)


def test_legend_sorting_and_empty(complex_shadow_dataset):
    ds = complex_shadow_dataset.copy(deep=True)
    # shadow_result = ShadowResult(ds)
    mixed_labels = ["Turbine 10", "Turbine 2", "Turbine A"]
    ds_visible_shadows = ds.copy(deep=True)
    ds_visible_shadows["Rotor shadow"][:, :, 0] = True
    ds_visible_shadows["Tower shadow"][:, :, 0] = True
    shadow_with_shadows = ShadowResult(ds_visible_shadows)
    fig, axes = shadow_with_shadows.calendar(
        turbine_labels=mixed_labels[:len(ds.wt)],
        show=False
    )
    plt.close(fig)
    ds_no_shadows = ds.copy(deep=True)
    ds_no_shadows["Rotor shadow"][:] = False
    ds_no_shadows["Tower shadow"][:] = False
    ds_no_shadows["Combined shadow"][:] = False
    shadow_no_shadows = ShadowResult(ds_no_shadows)
    fig_empty, axes_empty = shadow_no_shadows.calendar(show=False)
    assert len(fig_empty.legends) == 0
    plt.close(fig_empty)


def test_collapse_turbine_var_without_wt_dimension():
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(2, 2, len(times))).astype(bool)),
            "Tower shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(2, 2, len(times))).astype(bool)),
            "rec_x": (["rec"], [12.0, 12.01]),
            "rec_y": (["rec"], [55.1, 55.0]),
            "src_x": (["wt"], [12.0, 12.1]),
            "src_y": (["wt"], [55.0, 55.1]),
            "non_wt_var": (["rec", "time"], np.random.rand(2, len(times))),
        },
        coords={
            "wt": [0, 1],
            "rec": [0, 1],
            "time": times,
        },
        attrs={
            "mode": "both",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow = ShadowMap(ds)
    collapsed = shadow.collapse_turbine()
    assert "non_wt_var" in collapsed.dataset.data_vars
    assert np.array_equal(
        collapsed.dataset["non_wt_var"].values,
        ds["non_wt_var"].values)


def test_animate_scatter_exception():
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    x = np.linspace(10, 11, 5)
    y = np.linspace(50, 51, 5)
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(1, 1, len(times))).astype(bool)),
            "Tower shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(1, 1, len(times))).astype(bool)),
            "rec_x": (["rec"], [12.0]),
            "rec_y": (["rec"], [55.0]),
            "src_x": (["wt"], [12.0]),
            "src_y": (["wt"], [55.0]),
        },
        coords={
            "wt": [0],
            "rec": [0],
            "time": times,
            "x": x,
            "y": y
        },
        attrs={
            "mode": "both",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow_map = ShadowMap(ds)
    original_scatter = plt.Axes.scatter

    def patched_scatter(self, *args, **kwargs):
        stack = inspect.stack()
        if any('animate' in frame.function for frame in stack):
            if hasattr(shadow_map, 'src_x') and hasattr(shadow_map, 'src_y'):
                raise RuntimeError("Forced exception in scatter")
        return original_scatter(self, *args, **kwargs)
    plt.Axes.scatter = patched_scatter
    try:
        anim = shadow_map.animate(show=False)
        assert anim is not None
    finally:
        plt.Axes.scatter = original_scatter
        plt.close('all')


def test_animate_invalid_source_coordinates_type():
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    x = np.linspace(10, 11, 5)
    y = np.linspace(50, 51, 5)
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(1, 1, len(times))).astype(bool)),
            "Tower shadow": (["wt", "rec", "time"], np.random.randint(0, 2, size=(1, 1, len(times))).astype(bool)),
            "rec_x": (["rec"], [12.0]),
            "rec_y": (["rec"], [55.0]),
        },
        coords={
            "wt": [0],
            "rec": [0],
            "time": times,
            "x": x,
            "y": y
        },
        attrs={
            "mode": "both",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow_map = ShadowMap(ds)
    shadow_map.src_x = "invalid_x"
    shadow_map.src_y = "invalid_y"
    anim = shadow_map.animate(show=False)
    assert anim is not None
    plt.close('all')


def test_calendar_sunrise_sunset_nan_altitude():
    times = pd.date_range("2024-12-21", "2024-12-23", freq="1h", tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (
                [
                    "wt", "rec", "time"], np.random.randint(
                    0, 2, size=(
                        1, 1, len(times))).astype(bool)), "Tower shadow": (
                            [
                                "wt", "rec", "time"], np.random.randint(
                                    0, 2, size=(
                                        1, 1, len(times))).astype(bool)), "rec_x": (
                                            ["rec"], [12.0]), "rec_y": (
                                                ["rec"], [89.9]), "src_x": (
                                                    ["wt"], [12.0]), "src_y": (
                                                        ["wt"], [89.9]), }, coords={
                                                            "wt": [0], "rec": [0], "time": times, }, attrs={
            "mode": "both", "freq": 1.0, "tz": "UTC"})

    def mock_solar_position(times, latitude, longitude):
        shape = (len(latitude), len(times), 2)
        solar_pos = np.zeros(shape)
        celestial_coords = np.zeros(shape)
        celestial_coords[:, 0, 1] = 5.0
        celestial_coords[:, 1, 1] = -5.0
        celestial_coords[:, 2, 1] = 5.0
        celestial_coords[:, 3, 1] = np.nan
        return solar_pos, celestial_coords
    with patch('py_wake.shadow_models.shadow_base.solar_position', mock_solar_position):
        shadow_result = ShadowResult(ds)
        fig, axes = shadow_result.calendar(show=False)
        plt.close(fig)


def test_calendar_rotor_mode():
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.ones((1, 2, len(times)), dtype=bool)),
            "rec_x": (["rec"], [12.0, 12.01]),
            "rec_y": (["rec"], [55.1, 55.0]),
            "src_x": (["wt"], [12.0]),
            "src_y": (["wt"], [55.0]),
        },
        coords={
            "wt": [0],
            "rec": [0, 1],
            "time": times,
        },
        attrs={
            "mode": "rotor",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow_result = ShadowResult(ds)
    fig, axes = shadow_result.calendar(show=False)
    plt.close(fig)


def test_calendar_tower_mode():
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "Tower shadow": (["wt", "rec", "time"], np.ones((1, 2, len(times)), dtype=bool)),
            "rec_x": (["rec"], [12.0, 12.01]),
            "rec_y": (["rec"], [55.1, 55.0]),
            "src_x": (["wt"], [12.0]),
            "src_y": (["wt"], [55.0]),
        },
        coords={
            "wt": [0],
            "rec": [0, 1],
            "time": times,
        },
        attrs={
            "mode": "tower",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow_result = ShadowResult(ds)
    fig, axes = shadow_result.calendar(show=False)
    plt.close(fig)


def test_calendar_invalid_mode():
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "rec_x": (["rec"], [12.0, 12.01]),
            "rec_y": (["rec"], [55.1, 55.0]),
            "src_x": (["wt"], [12.0]),
            "src_y": (["wt"], [55.0]),
        },
        coords={
            "wt": [0],
            "rec": [0, 1],
            "time": times,
        },
        attrs={
            "mode": "invalid_mode",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow_result = ShadowResult(ds)
    with pytest.raises(ValueError, match="Invalid shadow mode"):
        shadow_result.calendar(show=False)


def test_calendar_mismatched_turbine_labels():
    times = pd.date_range(
        "2024-01-01",
        "2024-01-01 02:00",
        freq="1h",
        tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (["wt", "rec", "time"], np.ones((2, 2, len(times)), dtype=bool)),
            "Tower shadow": (["wt", "rec", "time"], np.ones((2, 2, len(times)), dtype=bool)),
            "rec_x": (["rec"], [12.0, 12.01]),
            "rec_y": (["rec"], [55.1, 55.0]),
            "src_x": (["wt"], [12.0, 12.1]),
            "src_y": (["wt"], [55.0, 55.1]),
        },
        coords={
            "wt": [0, 1],
            "rec": [0, 1],
            "time": times,
        },
        attrs={
            "mode": "both",
            "freq": 1.0,
            "tz": "UTC"
        }
    )
    shadow_result = ShadowResult(ds)
    with pytest.raises(ValueError, match="Length of turbine_labels must match number of turbines"):
        shadow_result.calendar(turbine_labels=["T1"], show=False)


def test_calendar_sunrise_sunset_invalid_time_mask():
    times = pd.date_range("2024-01-01", "2024-01-03", freq="1h", tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (
                [
                    "wt", "rec", "time"], np.random.randint(
                    0, 2, size=(
                        1, 1, len(times))).astype(bool)), "Tower shadow": (
                            [
                                "wt", "rec", "time"], np.random.randint(
                                    0, 2, size=(
                                        1, 1, len(times))).astype(bool)), "rec_x": (
                                            ["rec"], [12.0]), "rec_y": (
                                                ["rec"], [55.0]), "src_x": (
                                                    ["wt"], [12.0]), "src_y": (
                                                        ["wt"], [55.0]), }, coords={
                                                            "wt": [0], "rec": [0], "time": times, }, attrs={
            "mode": "both", "freq": 1.0, "tz": "UTC"})
    shadow_result = ShadowResult(ds)
    fig, axes = shadow_result.calendar(hour_range=(10, 14), show=False)
    plt.close(fig)


def test_calendar_sunrise_sunset_continue():
    times = pd.date_range("2024-01-01", "2024-01-02", freq="1h", tz="UTC")
    ds = xr.Dataset(
        data_vars={
            "Rotor shadow": (
                [
                    "wt", "rec", "time"], np.random.randint(
                    0, 2, size=(
                        1, 1, len(times))).astype(bool)), "Tower shadow": (
                            [
                                "wt", "rec", "time"], np.random.randint(
                                    0, 2, size=(
                                        1, 1, len(times))).astype(bool)), "rec_x": (
                                            ["rec"], [12.0]), "rec_y": (
                                                ["rec"], [55.0]), "src_x": (
                                                    ["wt"], [12.0]), "src_y": (
                                                        ["wt"], [55.0]), }, coords={
                                                            "wt": [0], "rec": [0], "time": times, }, attrs={
            "mode": "both", "freq": 1.0, "tz": "UTC"})

    def mock_solar_position(times, latitude, longitude):
        shape = (len(latitude), len(times), 2)
        solar_pos = np.zeros(shape)
        celestial_coords = np.zeros(shape)
        for i in range(len(times) - 1):
            if i % 2 == 0:
                celestial_coords[:, i, 1] = 5.0
                celestial_coords[:, i + 1, 1] = -5.0
        return solar_pos, celestial_coords
    with patch('py_wake.shadow_models.shadow_base.solar_position', mock_solar_position):
        shadow_result = ShadowResult(ds)
        fig, axes = shadow_result.calendar(hour_range=(8, 12), show=False)
        plt.close(fig)


if __name__ == "__main__":

    pytest.main([__file__, "-v", "--color=yes"])
