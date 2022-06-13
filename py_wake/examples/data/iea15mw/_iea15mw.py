from py_wake import np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine

rotor_diameter = 240.

wind_speeds = np.array([0.0, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.25,
                        6.0, 6.2, 6.4, 6.5, 6.55, 6.6, 6.7, 6.8,
                        6.9, 6.92, 6.93, 6.94, 6.95, 6.96, 6.97,
                        6.98, 6.99, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
                        10.0, 10.25, 10.5, 10.6, 10.7, 10.72, 10.74,
                        10.76, 10.78, 10.784, 10.786, 10.787, 10.788,
                        10.789, 10.79, 10.8, 10.9, 11.0, 11.25, 11.5,
                        11.75, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 22.5,
                        25.01, 25.02, 50.0])
cp_vals = np.array([0.0, 0.09331, 0.25314, 0.33415, 0.38052, 0.39738,
                    0.41089, 0.42102, 0.442, 0.44592, 0.44907, 0.45054,
                    0.45122, 0.45189, 0.45302, 0.45393, 0.45465, 0.45476,
                    0.45482, 0.45485, 0.45488, 0.45489, 0.45491, 0.45492,
                    0.45491, 0.45492, 0.45498, 0.45501, 0.45504, 0.45504,
                    0.45505, 0.45507, 0.45513, 0.45512, 0.454522, 0.441897,
                    0.439429, 0.436978, 0.434546, 0.432132, 0.431651, 0.431411,
                    0.431291, 0.431171, 0.431052, 0.430992, 0.429736, 0.418016,
                    0.406719, 0.380203, 0.355942, 0.333702, 0.313277, 0.246401,
                    0.197283, 0.160398, 0.101009, 0.067668, 0.047525, 0.034604,
                    0.0, 0.0])

ct_vals = np.array([0.0, 0.819749, 0.801112, 0.808268, 0.821911, 0.822876,
                    0.823266, 0.830989, 0.834932, 0.833619, 0.831805, 0.829011,
                    0.826909, 0.824741, 0.82043, 0.816176, 0.8112, 0.809741,
                    0.808781, 0.808102, 0.807567, 0.807252, 0.806624, 0.806496,
                    0.806806, 0.806651, 0.80547, 0.804572, 0.803949, 0.803905,
                    0.803709, 0.803452, 0.801706, 0.801777, 0.768658, 0.707315,
                    0.698508, 0.690212, 0.682336, 0.674836, 0.673371, 0.672646,
                    0.672283, 0.671922, 0.671564, 0.671387, 0.66764, 0.635292,
                    0.607278, 0.548966, 0.501379, 0.460983, 0.425966, 0.321166,
                    0.251102, 0.201415, 0.125654, 0.085067, 0.061026, 0.045815,
                    0.0, 0.0])

power_curve = np.column_stack([wind_speeds,
                               cp_vals * 0.5 * 1.225 *
                               np.pi * (rotor_diameter // 2) ** 2 *
                               wind_speeds ** 3 / 1e3])

ct_curve = np.column_stack([wind_speeds, ct_vals])


class IEA15MW(WindTurbine):
    '''
    Data from:
    https://www.nrel.gov/docs/fy20osti/75698.pdf
    Evan Gaertner, Jennifer Rinker, Latha Sethuraman,
    Frederik Zahle, Benjamin Anderson, Garrett Barter,
    Nikhar Abbas, Fanzhong Meng, Pietro Bortolotti,
    Witold Skrzypinski, George Scott, Roland Feil,
    Henrik Bredmose, Katherine Dykes, Matt Shields,
    Christopher Allen, and Anthony Viselli
    "Definition of the IEA 15-Megawatt
    Offshore Reference Wind Turbine", March 2020
    '''

    def __init__(self, method='linear'):
        u, p = power_curve.T
        WindTurbine.__init__(
            self,
            'IEA15MW',
            diameter=rotor_diameter,
            hub_height=150,
            powerCtFunction=PowerCtTabular(u, p * 1000, 'w',
                                           ct_curve[:, 1],
                                           ws_cutin=3, ws_cutout=25,
                                           method=method))


IEA15WM_RWT = IEA15MW


def main():
    wt = IEA15MW()
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
