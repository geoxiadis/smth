import pytest

from py_wake import np
from py_wake.tests import npt
from py_wake.utils.most import phi, psi, phi_eps


def test_phi():
    zeta = np.linspace(-.1, .1, 11)
    # print(list(np.round(phi(zeta), 3)))
    if 0:
        plt.plot(zeta, phi(zeta))
        plt.show()

    npt.assert_array_almost_equal(phi(zeta), [0.764, 0.792, 0.825, 0.867, 0.922, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 3)


def test_psi():
    zeta = np.linspace(-.1, .1, 51)
    # print(list(np.round(psi(zeta[::5]), 3)))
    # print(list(np.round(psi(zeta[::5], 'Wilson'), 3)))
    if 0:
        plt.plot(zeta, psi(zeta))
        plt.plot(zeta, psi(zeta, 'Wilson'), label='Wilson')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(psi(zeta[::5]),
                                  [0.326, 0.276, 0.221, 0.159, 0.087, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3)
    npt.assert_array_almost_equal(psi(zeta[::5], 'Wilson'),
                                  [2.541, 2.488, 2.427, 2.355, 2.261, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3)


def test_phi_eps():
    zeta = np.linspace(-.1, .1, 11)
    # print(list(np.round(phi_eps(zeta), 3)))
    if 0:
        plt.plot(zeta, phi_eps(zeta))
        plt.show()

    npt.assert_array_almost_equal(phi_eps(zeta), [1.1, 1.08, 1.06, 1.04, 1.02, 1.0, 1.08, 1.16, 1.24, 1.32, 1.4], 3)
