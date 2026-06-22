
import numpy as np
import pytest
from numpy import newaxis as na

from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.deficit_models.large_wind_farm_correction_model import (
    LargeWindFarmCorrectionDeficitModel,
    LargeWindFarmCorrectionModel,
)
from py_wake.site import UniformSite


def test_large_wind_farm_correction_deficit_model_returns_correct_deficit() -> None:
    lwfc_deficit_model = LargeWindFarmCorrectionDeficitModel()

    d_rotor = 120.0
    dw_points = np.arange(0.0, 50.0 * d_rotor, 5.0 * d_rotor)[:, na, na]
    dw_ijlk = np.array([dw_points, dw_points, dw_points])
    hcw_ijlk = np.zeros_like(dw_ijlk)
    hcw_ijlk[1, :, :, :] = 150.0
    hcw_ijlk[2, :, :, :] = -150.0

    deficit = lwfc_deficit_model.calc_deficit(
        WS_ilk=np.array([[[10.0]]]),
        dw_ijlk=dw_ijlk,
        hcw_ijlk=hcw_ijlk,
        h_ilk=np.array([[[125.0]]]),
        dh_ijlk=np.array([[[[10.0]]]]),
        D_src_il=np.array([[d_rotor]]),
        D_dst_ijl=np.array([[[d_rotor]]]),
        ct_ilk=np.array([[[0.9]]]),
    )

    expected_deficit = np.array(
        [
            [
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.12635673]],
                [[0.28104624]],
                [[0.40360127]],
                [[0.50468285]],
                [[0.59043625]],
                [[0.66472218]],
                [[0.73012120]],
            ],
            [
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
            ],
            [
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
                [[0.00000000]],
            ],
        ]
    )

    assert np.allclose(deficit, expected_deficit)


def test_invalid_roughness_raises_error() -> None:
    with pytest.raises(
        expected_exception=ValueError,
        match="The increased roughness must be larger than the base roughness",
    ):
        _ = LargeWindFarmCorrectionDeficitModel(
            base_roughness=1.0,
            increased_roughness=0.5,
        )


def test_invalid_negative_wind_speed_raises_error() -> None:
    lwfc_deficit_model = LargeWindFarmCorrectionDeficitModel()
    with pytest.raises(
        expected_exception=ValueError,
        match="Negative wind speed values are not valid",
    ):
        _ = lwfc_deficit_model.calc_deficit(
            WS_ilk=np.array([[[-1.0]]]),
            dw_ijlk=np.array([[[[100.0]]]]),
            hcw_ijlk=np.array([[[[100.0]]]]),
            h_ilk=np.array([[[100.0]]]),
            dh_ijlk=np.array([[[[10.0]]]]),
            D_src_il=np.array([[100.0]]),
            D_dst_ijl=np.array([[[100.0]]]),
            ct_ilk=np.array([[[0.8]]]),
        )


def test_invalid_negative_thrust_raises_error() -> None:
    lwfc_deficit_model = LargeWindFarmCorrectionDeficitModel()
    with pytest.raises(
        expected_exception=ValueError,
        match="Negative thrust coefficient.*not valid",
    ):
        _ = lwfc_deficit_model.calc_deficit(
            WS_ilk=np.array([[[10.0]]]),
            dw_ijlk=np.array([[[[100.0]]]]),
            hcw_ijlk=np.array([[[[100.0]]]]),
            h_ilk=np.array([[[100.0]]]),
            dh_ijlk=np.array([[[[10.0]]]]),
            D_src_il=np.array([[100.0]]),
            D_dst_ijl=np.array([[[100.0]]]),
            ct_ilk=np.array([[[-0.05]]]),
        )


def test_invalid_nan_inputs_raises_error() -> None:
    model = LargeWindFarmCorrectionDeficitModel()

    with pytest.raises(ValueError, match="Found NaN value in the input array 'WS_ilk'"):
        model.calc_deficit(
            WS_ilk=np.array([[[12.0, np.nan]]]),
            dw_ijlk=np.array([[[[250.0, 250.0]]]]),
            hcw_ijlk=np.array([[[[100.0, 100.0]]]]),
            h_ilk=np.array([[[100.0, 100.0]]]),
            dh_ijlk=np.array([[[[10.0, 10.0]]]]),
            D_src_il=np.array([[[100.0, 100.0]]]),
            D_dst_ijl=np.array([[[100.0, 100.0]]]),
            ct_ilk=np.array([[[0.1, 0.1]]]),
        )


def test_large_wind_farm_correction_wind_farm_model_returns_correct_aep() -> None:
    model = LargeWindFarmCorrectionModel(
        site=UniformSite(p_wd=[0.1, 0.3, 0.2, 0.4], ws=10),
        windTurbines=HornsrevV80(),
        base_roughness=0.0002,
        increased_roughness=0.03,
    )
    results = model(
        x=np.array(
            [0.0, 200.0, 400.0, 1200.0, 5000.0, 0.0, 200.0, 400.0, 1200.0, 5000.0, 0.0, 200.0, 400.0, 1200.0, 5000.0]
        ),
        y=np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 300.0, 300.0, 300.0, 300.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        ),
        h=np.array([60.0, 65.0, 60.0, 65.0, 60.0, 60.0, 65.0, 60.0, 65.0, 60.0, 60.0, 65.0, 60.0, 65.0, 60.0]),
    )
    aep = results.aep().sum().values.item()

    assert np.isclose(aep, 173.69966794)
