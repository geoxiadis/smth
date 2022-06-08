from py_wake import np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine

rotor_diameter = 198.0
wind_speeds = np.array([0.0, 2.9, 3.0, 4.0, 4.5147, 5.0008, 5.4574, 5.8833,
                        6.2777, 6.6397, 6.9684, 7.2632, 7.5234, 7.7484,
                        7.9377, 8.0909, 8.2077, 8.2877, 8.3308, 8.337,
                        8.3678, 8.4356, 8.5401, 8.6812, 8.8585, 9.0717,
                        9.3202, 9.6035, 9.921, 10.272, 10.6557, 10.7577,
                        11.5177, 11.9941, 12.4994, 13.0324, 13.592, 14.1769,
                        14.7859, 15.4175, 16.0704, 16.7432, 17.4342, 18.1421,
                        18.8652, 19.6019, 20.3506, 21.1096, 21.8773, 22.6519,
                        23.4317, 24.215, 25.01, 25.02, 50.0])
cp_vals = np.array([0.0, 0.0, 0.074, 0.3251, 0.3762, 0.4027, 0.4156, 0.423,
                    0.4274, 0.4293, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298,
                    0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298,
                    0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298,
                    0.4305, 0.438256, 0.425908, 0.347037, 0.307306, 0.271523,
                    0.239552, 0.211166, 0.186093, 0.164033, 0.144688, 0.12776,
                    0.112969, 0.100062, 0.0888, 0.078975, 0.070401, 0.062913,
                    0.056368, 0.05064, 0.04562, 0.041216, 0.037344, 0.033935,
                    0.0, 0.0])

ct_vals = np.array([0.0, 0.0, 0.7701, 0.7701, 0.7763, 0.7824, 0.782, 0.7802,
                    0.7772, 0.7719, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768,
                    0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768,
                    0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768,
                    0.7675, 0.7651, 0.7587, 0.5056, 0.431, 0.3708, 0.3209,
                    0.2788, 0.2432, 0.2128, 0.1868, 0.1645, 0.1454, 0.1289,
                    0.1147, 0.1024, 0.0918, 0.0825, 0.0745, 0.0675, 0.0613,
                    0.0559, 0.0512, 0.047, 0.0, 0.0])

power_curve = np.column_stack([wind_speeds,
                               cp_vals * 0.5 * 1.225 *
                               np.pi * (rotor_diameter // 2) ** 2 *
                               wind_speeds ** 3])

ct_curve = np.column_stack([wind_speeds, ct_vals])


class IEA10MW(WindTurbine):
    '''
    Data from:
    https://www.nrel.gov/docs/fy19osti/73492.pdf
    Pietro Bortolotti, Helena Canet Tarr ́es, Katherine Dykes, Karl Merz,
    Latha Sethuraman, David Verelst, and Frederik Zahle
    "IEA Wind Task 37 on Systems Engineering in Wind Energy WP2.1 Reference
    Wind Turbines", May 23 2019
    '''

    def __init__(self, method='linear'):
        u, p = power_curve.T
        WindTurbine.__init__(
            self,
            'IEA10MW',
            diameter=rotor_diameter,
            hub_height=119,
            powerCtFunction=PowerCtTabular(u, p * 1000, 'w',
                                           ct_curve[:, 1],
                                           ws_cutin=4,
                                           ws_cutout=25,
                                           method=method))


IEA10WM_RWT = IEA10MW


def main():
    wt = IEA10MW()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-', label='power [W]')
    c = plt.plot([], label='ct')[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), '.-', color=c)
    plt.show()


if __name__ == '__main__':
    main()
