"""Formulations for the simplified Eddy Viscosity (EV) model.

This module comprises formulations of the system of equations to solve
for the simplified EV model. The simplified solution was proposed by
Anderson (2011) as a reduced formulation of the original model equations
developed by Ainslie (1988). The simplified version utilises the
observation that the wake profile is self-similar to reduce the original
two-dimensional problem (in terms of downstream and radial distance) to
a one-dimensional problem (in terms of downstream distance only). The
simplified formulation facilitates much faster solutions without
sacrificing accuracy or precision.


Two alternative formulations are included: the first defined in terms of
the dimensionless wake wind speed, as in the Anderson (2011) paper, and
the second defined in terms of the dimensionless wake wind speed deficit
(each of these two values being unity minus the other). The two versions
produce very similar results. Both are included for testing purposes.

References:

    Ainslie, J F. "Calculating the flowfield in the wake of wind
    turbines," Journal of Wind Engineering and Industrial Aerodynamics,
    volume 27, pp. 213-224, 1988.

    Anderson, M. "Simplified solution to the eddy-viscosity wake model,"
    Renewable Energy Systems Ltd (RES) report. Document reference 01327-
    000202, issue 03, 2011.

"""

from typing import Protocol

import numpy as np


def mixing_function(x: np.ndarray) -> np.ndarray:
    """The near wake mixing filter function defined by Ainslie (1988).

    :param x: dimensionless downstream distance from the source turbine,
        normalised by the source turbine rotor diameter
    :return: an array of the value(s) of the mixing function evaluated
        at ``x``, with the same shape as ``x``
    """
    fx = 0.65 + np.cbrt((x - 4.5) / 23.32)
    return np.where(2.0 < x < 5.5, np.minimum(fx, 1.0), 1.0)


class SimplifiedEddyViscosityFormulationProvider(Protocol):
    """Protocol for simplified Eddy Viscosity formulation providers."""

    @staticmethod
    def is_deficit_formulation() -> bool:
        """Whether defined in terms of wind speed deficit.

        If ``True``, the initial value and derivatives are expected to
        be calculated in terms of the dimensionless wind speed deficit.
        If ``False``, those functions are expected to be implemented in
        terms of the dimensionless wake wind speed.
        """

    @staticmethod
    def initial_u_value(ct: np.ndarray, ti0: np.ndarray) -> np.ndarray:
        """The initial value at two rotor diameters downstream.

        This can be either the initial dimensionless wake wind speed or
        the initial dimensionless wake wind speed deficit, depending on
        the formulation (return value of ``is_deficit_formulation``).

        The formulation of the EV model starts at two rotor diameters
        downstream and does not include any solution for the very near
        wake region the first two rotor diameters behind the source
        turbine.

        The function is vectorised to operate on ``numpy`` arrays.

        :param ct: source turbine thrust coefficient
        :param ti0: turbulence intensity incident to the source turbine
        :return: an array of the initial value(s), having a shape
            corresponding to the product of ``ct`` and ``ti0``
        """

    @staticmethod
    def streamwise_derivative(
        x: np.ndarray,
        u: np.ndarray,
        ct: np.ndarray,
        ti0: np.ndarray,
        use_mixing_function: bool = True,
    ) -> np.ndarray:
        """Derivative of speed variable with respect to distance.

        In the case of a wake wind speed formulation, this is the
        derivative of the dimensionless wake wind speed with respect to
        dimensionless downstream distance. In the case of a deficit
        formulation this is the derivative of the dimensionless wake
        wind speed deficit with respect to dimensionless downstream
        distance.

        The function is vectorised to operate on ``numpy`` arrays.

        :param x: dimensionless downstream distance from the source turbine,
            normalised by the source turbine rotor diameter
        :param u: dimensionless wind speed variable at ``x``
        :param ct: source turbine thrust coefficient
        :param ti0: turbulence intensity incident to the source turbine
        :param use_mixing_function: whether to apply the mixing filter
            function in the near wake region, as per the original model
            formulation by Ainslie (1988); note this must be a simple
            ``bool`` value and not an array for vectorisation
        """


class SimplifiedEddyViscositySpeedFormulation:
    """Simplified EV model formulation in terms of wake wind speed.

    See documentation of ``SimplifiedEddyViscosityFormulationProvider``
    for further details.
    """

    @staticmethod
    def is_deficit_formulation() -> bool:
        """Whether a deficit formulation, which is always ``False``."""
        return False

    @staticmethod
    def initial_u_value(ct: np.ndarray, ti0: np.ndarray) -> np.ndarray:
        """The initial dimensionless wake wind speed value."""
        u = 1 - ct + 0.05 + (16.0 * ct - 0.5) * (ti0 / 10.0)
        u = np.maximum(u, 0.0001)
        u = np.minimum(u, 1.0)
        return u

    @staticmethod
    def streamwise_derivative(
        x: np.ndarray,
        u: np.ndarray,
        ct: np.ndarray,
        ti0: np.ndarray,
        use_mixing_function: bool = True,
    ) -> np.ndarray:
        """The derivative of wake wind speed with respect to distance."""
        # First filter u to avoid numerical instability when out of expected range
        u_f = np.maximum(u, 0.001)
        u_f = np.minimum(u_f, 0.999)

        fx = np.array(1.0)
        if use_mixing_function:
            fx = mixing_function(x=x)

        du_dx_values = (
            16.0
            * fx
            * (
                0.015
                * (np.sqrt((3.56 * ct) / (4.0 * (1.0 - np.power(u_f, 2.0)))))
                * (1.0 - u_f)
                + 0.16 * ti0
            )
            * (np.power(u_f, 3.0) - np.power(u_f, 2.0) - u_f + 1.0)
            / (ct * u_f)
        )

        # Return only positive values (representing recovery of the wake wind speed)
        du_dx_values = np.maximum(du_dx_values, 0.0)

        # Return zero gradient if wind speed is close to recovered
        du_dx_values = np.where(u <= 0.999, du_dx_values, 0.0)

        return du_dx_values


class SimplifiedEddyViscosityDeficitFormulation:
    """Simplified EV model formulation in terms of wind speed deficit.

    See documentation of ``SimplifiedEddyViscosityFormulationProvider``
    for further details.
    """

    @staticmethod
    def is_deficit_formulation() -> bool:
        """Whether a deficit formulation, which is always ``True``."""
        return True

    @staticmethod
    def initial_u_value(ct: np.ndarray, ti0: np.ndarray) -> np.ndarray:
        """The initial dimensionless wake wind speed deficit value."""
        u = ct - 0.05 - (16.0 * ct - 0.5) * (ti0 / 10.0)
        u = np.maximum(u, 0.0)
        u = np.minimum(u, 0.9999)
        return u

    @staticmethod
    def streamwise_derivative(
        x: np.ndarray,
        u: np.ndarray,
        ct: np.ndarray,
        ti0: np.ndarray,
        use_mixing_function: bool = True,
    ) -> np.ndarray:
        """The derivative of wake deficit with respect to distance."""
        # First filter u to avoid numerical instability when out of expected range
        u_f = np.maximum(u, 0.001)
        u_f = np.minimum(u_f, 0.999)

        fx = np.array(1.0)
        if use_mixing_function:
            fx = mixing_function(x=x)

        du_dx_values = (
            -16.0
            * fx
            * (
                0.015 * u_f * np.sqrt((3.56 * ct) / (4.0 * u_f * (2.0 - u_f)))
                + 0.16 * ti0
            )
            * np.power(u_f, 2.0)
            * (2.0 - u_f)
            / (ct * (1.0 - u_f))
        )

        # Return only negative values (representing decay of the wind speed deficit)
        du_dx_values = np.minimum(du_dx_values, 0.0)

        # Return zero gradient if deficit is zero or very small
        du_dx_values = np.where(u >= 0.001, du_dx_values, 0.0)

        return du_dx_values
