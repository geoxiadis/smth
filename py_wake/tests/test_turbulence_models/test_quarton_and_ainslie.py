"""Tests for the Quarton and Ainslie turbulence model.

"""

import pytest
from py_wake import np
from py_wake.turbulence_models.quarton_and_ainslie import (
    QuartonAndAinslieTurbulenceModel,
)


@pytest.fixture
def calc_added_turbulence_kwargs() -> dict[str, np.ndarray]:
    """Test case arguments to the wake added turbulence calculation."""
    return {
        "WS_ilk": np.array(
            [
                [
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                ]
            ]
        ),
        "WS_eff_ilk": np.array(
            [
                [
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                ]
            ]
        ),
        "TI_ilk": np.array(
            [
                [
                    [0.05, 0.31, 0.11, 0.33, 0.09],
                    [0.02, 0.23, 0.11, 0.33, 0.09],
                    [0.08, 0.22, 0.11, 0.31, 0.09],
                ]
            ]
        ),
        "TI_eff_ilk": np.array(
            [
                [
                    [0.05, 0.31, 0.11, 0.33, 0.09],
                    [0.02, 0.23, 0.11, 0.33, 0.09],
                    [0.08, 0.22, 0.11, 0.31, 0.09],
                ]
            ]
        ),
        "dw_ijlk": np.array(
            [
                [
                    [
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                    ],
                    [
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                    ],
                ]
            ]
        ),
        "cw_ijlk": np.array(
            [
                [
                    [
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                    ],
                    [
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                    ],
                ]
            ]
        ),
        "ct_ilk": np.array(
            [
                [
                    [0.85, 0.46, 0.99, 0.74, 0.66],
                    [0.80, 0.40, 0.90, 0.70, 0.60],
                    [0.80, 0.46, 0.99, 0.70, 0.66],
                ]
            ]
        ),
        "D_src_il": np.array([[100.0]]),
        "D_dst_ijl": np.array([[[100.0]]]),
        "wake_radius_ijlk": np.array(
            [
                [
                    [
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                    ],
                    [
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                    ],
                ]
            ]
        ),
    }


@pytest.fixture
def expected_added_turbulence() -> np.ndarray:
    """Expected array of added turbulence results."""
    return np.array(
        [
            [
                [0.07305408, 0.0, 0.19462119, 0.0, 0.0153155],
                [0.03586518, 0.0, 0.09688273, 0.0, 0.01408464],
                [0.09177531, 0.0, 0.19462119, 0.0, 0.0153155],
            ],
            [
                [0.07305408, 0.0, 0.19462119, 0.0, 0.0153155],
                [0.03586518, 0.0, 0.09688273, 0.0, 0.01408464],
                [0.09177531, 0.0, 0.19462119, 0.0, 0.0153155],
            ],
        ]
    )


def test_calc_added_turbulence(
    calc_added_turbulence_kwargs: dict[str, np.ndarray],
    expected_added_turbulence: np.ndarray,
) -> None:
    """Assert the turbulence model returns the correct values."""
    model = QuartonAndAinslieTurbulenceModel()
    added_turbulence = model.calc_added_turbulence(**calc_added_turbulence_kwargs)
    assert np.allclose(added_turbulence, expected_added_turbulence)
