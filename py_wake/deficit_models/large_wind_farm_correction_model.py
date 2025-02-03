"""Large Wind Farm Correction (LWFC) model implementations.

This module includes a deficit model implementation and a pre-defined
wind farm model implementation.

The LWFC model was developed as kind of bias correction model for the
Eddy Viscosity (EV) model when applied to large wind farms. It is
intended to be used together with the EV model, and will not produce
meaningful results on its own. The process is to run the LWFC model
first, and then feed the EV model with the corrected wind speed
predictions as input.

References:

    Schlez, W. and Neubert, A. "New Developments in Large Wind Farm
    Modelling," European Wind Energy Conference and Exhibition, 2009.

"""

import functools
from typing import Any, Final, Optional

import numpy as np
from numpy import newaxis as na
from scipy import optimize as scipy_opt

from py_wake.deficit_models.deficit_model import DeficitModel, WakeDeficitModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter
from py_wake.site import XRSite
from py_wake.superposition_models import MaxSum
from py_wake.wind_farm_models.engineering_models import All2All
from py_wake.wind_turbines import WindTurbines

DEFAULT_BASE_ROUGHNESS: Final[float] = 0.0002

DEFAULT_INCREASED_ROUGHNESS: Final[float] = 0.03

DEFAULT_RECOVERY_START_DISTANCE: Final[float] = 60.0

DEFAULT_HALF_RECOVERY_DISTANCE: Final[float] = 40.0

DEFAULT_ROW_SPACING_LIMIT: Final[float] = 5.0

DEFAULT_MAXIMUM_WAKE_DISTANCE: Final[float] = 50000.0


class LargeWindFarmCorrectionDeficitModel(WakeDeficitModel):
    """Large Wind Farm Correction (LWFC) wake deficit model.

    This model is intended to correct for deep array impact that is not
    captured for large wind farms when using a model where only the
    maximum wake deficit from the surrounding turbines is applied.
    """

    def __init__(
        self,
        base_roughness: float = DEFAULT_BASE_ROUGHNESS,
        increased_roughness: float = DEFAULT_INCREASED_ROUGHNESS,
        recovery_start_distance: float = DEFAULT_RECOVERY_START_DISTANCE,
        half_recovery_distance: float = DEFAULT_HALF_RECOVERY_DISTANCE,
        row_spacing_limit: float = DEFAULT_ROW_SPACING_LIMIT,
        maximum_wake_distance: float = DEFAULT_MAXIMUM_WAKE_DISTANCE,
    ) -> None:
        """Initiate a ``LargeWindFarmCorrectionDeficitModel``.

        :param base_roughness: the dimensional base surface roughness
            value in meters (m)
        :param increased_roughness: the dimensional increased surface
            roughness value associated with the presence of the wind
            farm in meters (m)
        :param recovery_start_distance: the dimensionless wind speed
            recovery start distance in rotor diameters
        :param half_recovery_distance: the dimensionless half wake
            recovery distance in rotor diameters
        :param row_spacing_limit: the dimensionless row spacing in rotor
            diameters
        :param maximum_wake_distance: the dimensional maximum downstream
            distance over which the large wind farm deficit is applied
            in meters (m)
        """
        if increased_roughness <= base_roughness:
            raise ValueError(
                "The increased roughness must be larger than the base roughness."
            )

        DeficitModel.__init__(
            self,
            rotorAvgModel=RotorCenter(),
            groundModel=None,
            use_effective_ws=False,
            use_effective_ti=False,
        )
        self.base_roughness = base_roughness
        self.increased_roughness = increased_roughness
        self.recovery_start_distance = recovery_start_distance
        self.half_recovery_distance = half_recovery_distance
        self.row_spacing_limit = row_spacing_limit
        self.maximum_wake_distance = maximum_wake_distance
        fetch_distance_values = np.arange(
            0.0,
            self.maximum_wake_distance + 100.1,
            100.0
        )
        ibl_height_values = self.calculate_ibl_height(
            fetch_distance=fetch_distance_values,
            roughness_length=self.increased_roughness,
        )
        self.ibl_height_interpolator = functools.partial(
            np.interp,
            xp=fetch_distance_values,
            fp=ibl_height_values,
        )

    def calc_deficit(
        self,
        WS_ilk: np.ndarray,
        dw_ijlk: np.ndarray,
        hcw_ijlk: np.ndarray,
        h_ilk: np.ndarray,
        dh_ijlk: np.ndarray,
        D_src_il: np.ndarray,
        D_dst_ijl: np.ndarray,
        ct_ilk: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        """Calculate the estimate of wake velocity deficit.

        :param WS_ilk: the free wind speed at the source turbine
        :param dw_ijlk: the down-wind distance from the source to the
            destination turbine
        :param hcw_ijlk: the horizontal cross-wind distance from the
            source to the destination turbine
        :param h_ilk: the hub height of the source turbine
        :param dh_ijlk: the difference in height from the source turbine
            to the destination turbine
        :param D_src_il: the diameter of the source turbine
        :param D_dst_ijl: the diameter of the destination turbine
        :param ct_ilk: the source turbine thrust coefficient
        :return: the estimated dimensional wind speed deficit due to
            large wind farm deep array effects (as an array by the ith
            source turbine, the jth destination turbine, in the lth wind
            direction sector and for the kth wind speed bin)
        """
        for input_array, input_array_name in [
            (WS_ilk, "'WS_ilk' (free wind speed)"),
            (dw_ijlk, "'dw_ijlk' (downwind distance)"),
            (hcw_ijlk, "'hcw_ijlk' (horizontal crosswind distance)"),
            (h_ilk, "'h_ilk' (source turbine hub height)"),
            (dh_ijlk, "'dh_ijlk' (turbine hub height difference)"),
            (D_src_il, "'D_src_il' (source turbine diameter)"),
            (D_dst_ijl, "'D_dst_ijl' (destination turbine diameter)"),
            (ct_ilk, "'ct_ilk' (thrust coefficient)"),
        ]:
            if np.any(np.isnan(input_array)):
                raise ValueError(
                    f"Found NaN value in the input array {input_array_name}."
                )

        if np.min(WS_ilk) < 0.0:
            raise ValueError("Negative wind speed values are not valid.")

        if np.min(ct_ilk) < 0.0:
            raise ValueError("Negative thrust coefficient (ct) values are not valid.")

        u0_dst_ijlk = WS_ilk[na, :]

        d_src_ijlk = D_src_il[:, na, :, na]
        d_dst_ijlk = D_dst_ijl[:, :, :, na]

        # Height of internal boundary layer
        dw_ijlk_f = np.maximum(dw_ijlk, 0.0)
        dw_ijlk_f = np.minimum(dw_ijlk_f, self.maximum_wake_distance)
        h_ibl = self.ibl_height_interpolator(dw_ijlk_f.flatten(order="C"))
        h_ibl_ijlk = h_ibl.reshape(dw_ijlk.shape, order="C")

        # Adjusted height of internal boundary layer, which is set to
        # twice the increased roughness if the downstream distance is
        # negative, zero or very small
        h_ibl_adj_ijlk = h_ibl_ijlk + ((2.0 / 3.0) * h_ilk[:, na, :, :])
        h_ibl_adj_ijlk = np.where(
            dw_ijlk > self.increased_roughness,
            h_ibl_adj_ijlk,
            2.0 * self.increased_roughness,
        )

        # Adjusted height (lower rotor edge) at destination (target)
        # turbine, which is limited to twice the increased roughness to
        # avoid invalid values in the following calculations
        z_adj_ijlk = h_ilk[:, na, :, :] + dh_ijlk - (0.5 * d_dst_ijlk)
        z_adj_ijlk = np.maximum(z_adj_ijlk, 2.0 * self.increased_roughness)

        # Wake wind speed is taken as freestream wind speed in upper IBL region
        u_ijlk = u0_dst_ijlk.copy()

        # Expression for lower IBL region
        u_ijlk = np.where(
            z_adj_ijlk <= 0.09 * h_ibl_adj_ijlk,
            u0_dst_ijlk * (
                np.log(h_ibl_adj_ijlk / self.base_roughness) *
                np.log(z_adj_ijlk / self.increased_roughness)
            ) / (
                np.log(h_ibl_adj_ijlk / self.increased_roughness) *
                np.log(z_adj_ijlk / self.base_roughness)
            ),
            u_ijlk,
        )

        # Expression for interpolated region between lower and upper regions
        u_ijlk = np.where(
            (z_adj_ijlk > (0.09 * h_ibl_adj_ijlk)) &
            (z_adj_ijlk < (0.3 * h_ibl_adj_ijlk)),
            u0_dst_ijlk / np.log(z_adj_ijlk / self.base_roughness) * (
                (
                    np.log(h_ibl_adj_ijlk / self.base_roughness) /
                    np.log(h_ibl_adj_ijlk / self.increased_roughness)
                ) * np.log(0.09 * h_ibl_adj_ijlk / self.increased_roughness) * (
                    1.0 - (
                        np.log(z_adj_ijlk / (0.09 * h_ibl_adj_ijlk))
                        / np.log(0.3 / 0.09)
                    )
                ) + (
                    np.log(0.3 * h_ibl_adj_ijlk / self.base_roughness) *
                    np.log(z_adj_ijlk / (0.09 * h_ibl_adj_ijlk)) /
                    np.log(0.3 / 0.09)
                )
            ),
            u_ijlk,
        )

        # Apply wind speed recovery term beyond recovery start distance
        u_ijlk = np.where(
            (dw_ijlk / d_src_ijlk) >= self.recovery_start_distance,
            u0_dst_ijlk * (
                1.0 - (1.0 - u_ijlk / u0_dst_ijlk) * np.power(
                    0.5,
                    (
                        (dw_ijlk - d_src_ijlk * self.recovery_start_distance) /
                        (d_src_ijlk * self.half_recovery_distance)
                    ),
                )
            ),
            u_ijlk,
        )

        deficit_ijlk = u0_dst_ijlk - u_ijlk

        # Limit to minimum and maximum wake distance
        deficit_ijlk = np.where(
            (dw_ijlk > 0.0) & (dw_ijlk <= self.maximum_wake_distance),
            deficit_ijlk,
            0.0,
        )

        # Limit the wake deficit to be applied only within region of impact
        # in the horizontal cross-wind direction
        deficit_ijlk = np.where(
            np.absolute(hcw_ijlk) <= d_src_ijlk / 2.0,
            deficit_ijlk,
            0.0,
        )

        # Limit the wake deficit to be applied only with non-zero thrust coefficient
        deficit_ijlk = np.where(
            np.isclose(ct_ilk[:, na, :, :], 0.0),
            0.0,
            deficit_ijlk,
        )

        # Remove wake impact when the wind blows along a row of turbines
        # and perpendicular spacing to other rows is above the limit
        return np.where(
            np.any(
                np.min(
                    np.where(
                        hcw_ijlk > d_src_ijlk / 2.0,
                        hcw_ijlk,
                        np.inf,
                    ),
                    axis=0,
                    keepdims=True,
                )
                - np.max(
                    np.where(
                        hcw_ijlk < -d_src_ijlk / 2.0,
                        hcw_ijlk,
                        -np.inf,
                    ),
                    axis=0,
                    keepdims=True,
                )
                <= self.row_spacing_limit * d_src_ijlk,
                axis=0,
                keepdims=True,
            ),
            deficit_ijlk,
            0.0,
        )

    @staticmethod
    def ibl_height_equation(
        h_ibl: np.ndarray,
        fetch_distance: np.ndarray,
        roughness_length: float,
    ) -> np.ndarray:
        """The roots of the equation for the disturbance IBL height.

        :param h_ibl: the dimensional height of the internal boundary
            layer (IBL) in meters (m)
        :param fetch_distance: the dimensional fetch distance behind the
            disturbance in meters (m)
        :param roughness_length: the dimensional surface roughness
            length associated with the disturbance in meters (m), which
            is assumed to be higher than the background surface
            roughness length
        :return: the values that should be zero at the roots of the
            equation, to be used for solving for the IBL height.
        """
        return (
            (h_ibl / roughness_length) * (np.log(h_ibl / roughness_length) - 1.0) -
            (0.9 * fetch_distance / roughness_length)
        )

    @staticmethod
    def calculate_ibl_height(
        fetch_distance: np.ndarray,
        roughness_length: float,
    ) -> np.ndarray:
        """Calculate the internal boundary layer (IBL) height.

        :param fetch_distance: the down-wind distance from the source to
            the destination turbine
        :param roughness_length: the dimensional surface roughness
            length associated with the disturbance in meters (m), which
            is assumed to be higher than the background surface
            roughness length
        :return: the dimensional height of the internal boundary layer
            (IBL) in meters (m) at the destination turbine due to the
            disturbance introduced by the source turbine
        """
        fetch_distance_f = np.maximum(fetch_distance, 2.0 * roughness_length)
        fetch_distance_flat = fetch_distance_f.flatten(order="C")

        sol = scipy_opt.root(
            fun=LargeWindFarmCorrectionDeficitModel.ibl_height_equation,
            x0=np.sqrt(fetch_distance_flat),
            args=(fetch_distance_flat, roughness_length),
            method="hybr",
        )
        if not sol.success:
            raise ValueError(f"Failed to calculate IBL height: {sol.message}.")

        ibl_height = sol.x.reshape(fetch_distance_f.shape, order="C")

        return np.where(
            fetch_distance > roughness_length,
            ibl_height,
            roughness_length,
        )


class LargeWindFarmCorrectionModel(All2All):
    """Pre-defined wind farm model based on the LWFC deficit model."""

    def __init__(
        self,
        site: XRSite,
        windTurbines: WindTurbines,
        base_roughness: Optional[float] = None,
        increased_roughness: Optional[float] = None,
    ) -> None:
        """Initiate a ``LargeWindFarmCorrectionModel``.

        This model is intended to capture deep array effects and need to
        be combined with a model that captures the smaller scale wake
        impacts from the nearest turbines.

        The superposition model is fixed at ``MaxSum`` since the
        implementation does not make sense in conjunction with a
        different model. Similarly, the rotor averaging model is fixed
        at the rotor centre model.

        It is not intended to use this model together with a blockage
        model in a combined computation.

        The combination with deflection models is not supported, as
        wake deflection cannot be properly captured in this type of
        large-scale model. The phenomena targeted with this model should
        not be expected to be much affected by deflection.

        The large wind farm correction model does not depend on
        turbulence and therefore combination with a turbulence model is
        not supported.

        :param site: the site object to create the model for
        :param windTurbines: the wind turbines object
        :param base_roughness: the dimensional base surface roughness
            value in meters (m)
        :param increased_roughness: the dimensional increased surface
            roughness value associated with the presence of the wind
            farm in meters (m)
        """
        deficit_model_kwargs: dict[str, float] = {}
        if base_roughness is not None:
            deficit_model_kwargs["base_roughness"] = base_roughness
        if increased_roughness is not None:
            deficit_model_kwargs["increased_roughness"] = increased_roughness

        super().__init__(
            site=site,
            windTurbines=windTurbines,
            wake_deficitModel=LargeWindFarmCorrectionDeficitModel(**deficit_model_kwargs),
            superpositionModel=MaxSum(),
        )
