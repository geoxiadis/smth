import numpy as np
import pandas as pd
import pytest
from pyproj import CRS, Transformer

from py_wake.flow_map import Points
from py_wake.shadow_models.shadow import ShadowModel


def _shadow_model(**kwargs):
    time = pd.date_range("2024-01-01 10:00", periods=2, freq="h", tz="UTC")
    defaults = dict(
        src_x=[12.0],
        src_y=[55.0],
        src_h=100,
        turbine_diameter=90,
        tower_diameter=5,
        time=time,
        wind_direction=[0, 0],
        solar_position_method="nrel_numpy",
    )
    defaults.update(kwargs)
    return ShadowModel(**defaults)


def test_auto_calculation_crs_is_projected_meter_crs():
    sm = _shadow_model()
    result = sm.run(
        grid=Points(x=[12.001], y=[55.001], h=[0]),
        mode="rotor",
        verbose=False,
    )

    calculation_crs = CRS.from_user_input(result.dataset.attrs["calculation_crs"])
    assert calculation_crs.is_projected
    assert result.dataset.attrs["rec_crs"] == "EPSG:4326"
    assert "rec_lon" in result.dataset
    assert "rec_lat" in result.dataset


def test_accepts_projected_source_and_receptor_crs():
    wgs84_to_utm = Transformer.from_crs(
        "EPSG:4326", "EPSG:32633", always_xy=True)
    src_x, src_y = wgs84_to_utm.transform([12.0], [55.0])
    rec_x, rec_y = wgs84_to_utm.transform([12.001], [55.001])

    sm = _shadow_model(src_x=src_x, src_y=src_y, src_crs="EPSG:32633")
    result = sm.run(
        grid=Points(x=rec_x, y=rec_y, h=[0]),
        rec_crs="EPSG:32633",
        mode="rotor",
        verbose=False,
    )

    assert result.dataset.attrs["src_crs"] == "EPSG:32633"
    assert result.dataset.attrs["rec_crs"] == "EPSG:32633"
    np.testing.assert_allclose(result.dataset["rec_lon"], [12.001], atol=1e-6)
    np.testing.assert_allclose(result.dataset["rec_lat"], [55.001], atol=1e-6)


def test_user_calculation_crs_override():
    sm = _shadow_model(calculation_crs="EPSG:32633")
    result = sm.run(
        grid=Points(x=[12.001], y=[55.001], h=[0]),
        mode="rotor",
        verbose=False,
    )

    assert CRS.from_user_input(result.dataset.attrs["calculation_crs"]).to_epsg() == 32633


def test_rejects_non_projected_calculation_crs():
    sm = _shadow_model(calculation_crs="EPSG:4326")
    with pytest.raises(ValueError, match="calculation_crs must be a projected CRS"):
        sm.run(
            grid=Points(x=[12.001], y=[55.001], h=[0]),
            mode="rotor",
            verbose=False,
        )
