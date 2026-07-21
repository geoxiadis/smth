from py_wake import np


def phi(zeta, Cm1=5, Cm2=-19.3):
    """
    Monin-Obukhov Similarity Theory function of normalized wind shear
    """
    zeta = np.atleast_1d(zeta)
    e = np.where((zeta <= 0), -.25, 1)  # avoid warning when Cm2*zeta<-1
    return np.where(zeta <= 0, (1 + Cm2 * zeta)**e, 1 + Cm1 * zeta)


def psi(zeta, unstable='', Cm1=5, Cm2=-19.3):
    """
    Monin-Obukhov Similarity Theory function of integrated normalized wind shear
    """
    zeta = np.atleast_1d(zeta)
    aux = phi(zeta, Cm1=Cm1, Cm2=Cm2)**-1
    if unstable == 'Wilson':
        psi_n = np.where(
            zeta < 0,
            3 * np.log(1 + np.sqrt(1 + 3.6 * np.abs(zeta)**(2.0 / 3.0))),
            1.0 - 1.0 / aux
        )
    else:
        aux2 = (1.0 + aux)**2 * (1 + aux**2)
        rhs = -np.log(8.0 / aux2) - 2.0 * np.arctan(Cm2 * zeta / aux2)
        psi_n = np.where(zeta < 0, rhs, 1.0 - 1.0 / aux)
    psi_n = np.where(zeta == 0, 0.0, psi_n)
    return psi_n


def phi_eps(zeta, Cm1=5):
    """
    Monin-Obukhov Similarity Theory function of normalized dissipation of turbulent kinetic energy
    """
    zeta = np.atleast_1d(zeta)
    return np.where(zeta <= 0, 1.0 - zeta, 1.0 + (Cm1 - 1) * zeta)
