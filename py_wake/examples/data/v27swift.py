# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:55:42 2023

@author: mavar
"""

from py_wake import np
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

# Data from OpenFAST model v3
# Power is based on a density of 1.0640320278 kg/m^3
# Loss factor based on OpenFAST	0.887

power_curve = np.array([[3.0, 4.63584253599958],
                        [4.0, 10.9841749557889],
                        [5.0, 21.4446994580893],
                        [6.0, 37.0564406635783],
                        [7.0, 58.8201979723131],
                        [8.0, 87.729756982402],
                        [9.0, 122.304433964459],
                        [10.0, 160.230305517596],
                        [11.0, 200.010310684876]])

ct_curve = np.array([[3.0, 0.815113690200702],
                     [4.0, 0.814815186652044],
                     [5.0, 0.814815186652044],
                     [6.0, 0.814715685469159],
                     [7.0, 0.814317680737615],
                     [8.0, 0.805760579009434],
                     [9.0, 0.74814939411854],
                     [10.0, 0.687453672558185],
                     [11.0, 0.626061442717629]])

# Wind Turbine used in the SWiFT case
class V27SWIFT(WindTurbine):
    def __init__(self, method='linear'):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(self, name='V27SWIFT', diameter=27.0, hub_height=32.1,
                             powerCtFunction=PowerCtTabular(power_curve[:, 0], power_curve[:, 1], 'w',
                                                            ct_curve[:, 1], method=method))


def main():
    wt = V27SWIFT()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())

    import matplotlib.pyplot as plt
    ws = np.linspace(3, 20, 100)
    plt.plot(ws, wt.power(ws) * 1e-3, label='Power')
    c = plt.plot([], [], label='Ct')[0].get_color()
    plt.ylabel('Power [kW]')
    ax = plt.gca().twinx()
    ax.plot(ws, wt.ct(ws), color=c)
    ax.set_ylabel('Ct')
    plt.xlabel('Wind speed [m/s]')
    plt.gcf().axes[0].legend(loc=1)
    plt.show()


if __name__ == '__main__':
    main()