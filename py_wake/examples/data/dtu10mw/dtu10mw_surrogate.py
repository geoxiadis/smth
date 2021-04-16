import numpy as np
from py_wake.examples.data.dtu10mw import ct_curve
from py_wake.wind_turbines._wind_turbines import WindTurbine

from pathlib import Path
from py_wake.wind_turbines.power_ct_functions import PowerCtSurrogate
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.turbulence_models.stf import STF2017TurbulenceModel


class DTU10MWCTTabularSurrogate():
    """Dummy surrogate using DTU10MW CT tabular. Needed as thrust is not available in aeroelastic simulation database"""
    input_channel_names = ['U', 'TI', 'Alpha', 'Yaw']

    def predict_output(self, input):
        return np.interp(input[:, 0], ct_curve[:, 0], ct_curve[:, 1])


def get_input(ws, TI_eff, yaw, Alpha):
    # ['U', 'TI', 'Alpha', 'Yaw']
    if yaw is None:
        yaw = 0
    return [ws,
            TI_eff * 100,  # surrogate is in %, pywake is in decimal
            Alpha,
            np.degrees(yaw)]  # surrogate is in deg, pywake-windturbines is in radians


class DTU10MWSurrogateWindTurbine(WindTurbine):

    def __init__(self):

        powerCtFunction = PowerCtSurrogate(TensorflowSurrogate(Path(__file__).parent / "surrogates/Power", 'operation'), 'kw',
                                           DTU10MWCTTabularSurrogate(), get_input)
        sensors = ['Blade_root_flapwise_M_x', 'Blade_root_edgewise_M_y', 'Tower_top_tilt_M_x', 'Tower_top_yaw_M_z']
        path = Path(__file__).parent / 'surrogates'
        surrogates = [TensorflowSurrogate(path / s, 'operation') for s in sensors]

        load_surrogate = FunctionSurrogates(surrogates, get_input)
        WindTurbine.__init__(
            self,
            'DTU10MW',
            diameter=178.3,
            hub_height=119,
            powerCtFunction=powerCtFunction,
            loadFunction=load_surrogate)

    def get_input(self, ws, TI_eff, yaw, Alpha):
        # needed to test a get_input method with 'self' as first argument
        return get_input(ws, TI_eff, yaw, Alpha)


def main():
    from py_wake import NOJ
    import matplotlib.pyplot as plt

    wt_surrogate = DTU10MWSurrogateWindTurbine()

    site = Hornsrev1Site()
    sim_res = NOJ(site, wt_surrogate, turbulenceModel=STF2017TurbulenceModel())([0, -1000], [0, 0], ws=np.arange(6, 25),
                                                                                Alpha=.12)
    load_wd_averaged = sim_res.loads(normalize_probabilities=True, method='OneWT_WDAvg')
    loads = sim_res.loads(normalize_probabilities=True, method='OneWT')
    loads.DEL.isel(sensor=0, wt=0).plot()
    plt.figure()

    for s in load_wd_averaged.sensor:
        print(s.item(), load_wd_averaged.LDEL.sel(sensor=s, wt=0).item(), loads.LDEL.sel(sensor=s, wt=0).item())

    print('Diameter', wt_surrogate.diameter())
    print('Hub height', wt_surrogate.hub_height())
    wt = DTU10MW(method='pchip')

    ws = np.arange(5.5, 25, .1)
    plt.plot(ws, wt.power(ws) / 1e3, label='DTU10MW tabular')

    plt.plot(ws, wt_surrogate.power(ws, TI_eff=.05, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=5%')
    plt.plot(ws, wt_surrogate.power(ws, TI_eff=.1, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=10%')
    plt.plot(ws, wt_surrogate.power(ws, TI_eff=.2, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=20%')
    plt.plot(ws, wt_surrogate.power(ws, TI_eff=.4, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=40%')
    plt.xlabel('Wind speed [m/s]')
    plt.xlabel('Power [kW]')
    plt.legend()
#     ax = plt.twinx()
#     ax.plot(ws, wt.ct(ws), '--', label='DTU10MW tabular')
#     ax.plot(ws, wt_surrogate.ct(ws, TI_eff=.1, Alpha=0, yaw=0), '--', label='DTU10MW Surrogate')

    plt.figure()
    yaw = np.arange(-30, 50, 1)
    theta = np.deg2rad(yaw)
    ws = np.full(yaw.shape, 11)
    plt.plot(yaw, wt.power(ws, yaw=theta) / 1e3, label='DTU10MW tabular')
    plt.plot(yaw, wt_surrogate.power(ws=ws - .18, TI_eff=.1, Alpha=0, yaw=theta) / 1e3, label='DTU10MW Surrogate')
    plt.xlabel('Yaw misalignment [deg]')
    plt.xlabel('Power [kW]')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
