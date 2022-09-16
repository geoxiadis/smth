from py_wake import np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine

wsp = np.array([ 3.    ,  3.5495,  4.0679,  4.5539,  5.0064,  5.4244,  5.8069,
        6.153 ,  6.4619,  6.733 ,  6.9655,  7.1589,  7.3128,  7.4269,
        7.5009,  7.5345,  7.5412,  7.5883,  7.6757,  7.8031,  7.9702,
        8.1767,  8.4221,  8.7059,  9.0273,  9.3856,  9.78  , 10.21  ,
       10.659 , 10.673 , 11.17  , 11.699 , 12.259 , 12.848 , 13.465 ,
       14.109 , 14.778 , 15.471 , 16.185 , 16.921 , 17.674 , 18.445 ,
       19.231 , 20.03  , 20.841 , 21.661 , 22.489 , 23.323 , 24.16  ,
       25.    ])
power = np.array([   39456.,   291550.,   609950.,   986480.,  1410400.,  1869300.,
        2349300.,  2835500.,  3312700.,  3765700.,  4180200.,  4547500.,
        4855700.,  5091900.,  5248900.,  5321200.,  5335800.,  5438300.,
        5631700.,  5921400.,  6315600.,  6825000.,  7463400.,  8239000.,
        9168700., 10286000., 11618000., 13195000., 15000000., 15000000.,
       15000000., 15000000., 15000000., 15000000., 15000000., 15000000.,
       15000000., 15000000., 15000000., 15000000., 15000000., 15000000.,
       15000000., 15000000., 15000000., 15000000., 15000000., 15000000.,
       15000000., 15000000.])

ct = np.array([0.81672 , 0.79044 , 0.78393 , 0.78624 , 0.78824 , 0.78942 ,
       0.78902 , 0.7874  , 0.78503 , 0.78237 , 0.77955 , 0.77583 ,
       0.77583 , 0.77583 , 0.77583 , 0.77583 , 0.77583 , 0.77583 ,
       0.77583 , 0.77583 , 0.77583 , 0.77583 , 0.77583 , 0.77583 ,
       0.77583 , 0.77583 , 0.77583 , 0.77583 , 0.76922 , 0.7427  ,
       0.55949 , 0.46163 , 0.38786 , 0.32901 , 0.28093 , 0.24114 ,
       0.20795 , 0.1801  , 0.15663 , 0.13679 , 0.11995 , 0.10562 ,
       0.093384, 0.082908, 0.07391 , 0.066159, 0.059463, 0.053662,
       0.048622, 0.04423 ])


class IEA15MW(WindTurbine):
    '''
    Data from:
    https://www.nrel.gov/docs/fy20osti/75698.pdf
    https://github.com/IEAWindTask37/IEA-15-240-RWT/blob/master/performance/performance_ccblade.dat
    idling thrust coefficient is assumed

    '''

    def __init__(self, method='linear'):
        WindTurbine.__init__(
            self,
            'IEA15MW',
            diameter=240,
            hub_height=150,
            powerCtFunction=PowerCtTabular(wsp, power, 'w', ct, ws_cutin=3, ws_cutout=25,
                                           ct_idle=0.059, method=method))


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
