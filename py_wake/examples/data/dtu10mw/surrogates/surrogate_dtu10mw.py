import numpy as np
from py_wake.wind_turbines._wind_turbines import WindTurbine
from pathlib import Path
from py_wake.wind_turbines.power_ct_functions import PowerCtSurrogate, PowerSurrogate, PowerCtNDTabular
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate, TensorflowSurrogate_DTU10MW
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates, FunctionSurrogates_DTU10MW_loads
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.dtu10mw.surrogates.all_models import Power_Q50_model, DEL_Q50_FlapM_model, DEL_Q50_EdgeM_model, DEL_Q50_PitchM_model, DEL_Q50_ShaftMx_model, DEL_Q50_ShaftMz_model, DEL_Q50_TBFA_model, DEL_Q50_TBSS_model 
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
import py_wake
import os


class DTU10MWCTTabularSurrogate():
    """DTU10MW CT tabular surrogate. Needed as thrust is not available in aeroelastic simulation database"""
    input_channel_names = ['ws', 'psp', 'ti', 'Alpha', 'Air_density']
    
    ws_lst = np.arange(0, 26)
    psp_lst = np.array([0.4, 0.6, 0.8, 1.0, 1.05, 1.1])
    
    ct_array =np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0.94, 0.94, 0.94, 0.94, 0.94, 0.94],
                        [0.93, 0.93, 0.93, 0.93, 0.93, 0.93],
                        [0.9 , 0.9 , 0.9 , 0.9 , 0.9 , 0.9 ],
                        [0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
                        [0.83, 0.83, 0.83, 0.83, 0.82, 0.82],
                        [0.56, 0.84, 0.84, 0.84, 0.83, 0.82],
                        [0.37, 0.62, 0.84, 0.84, 0.83, 0.82],
                        [0.27, 0.42, 0.62, 0.84, 0.83, 0.82],
                        [0.21, 0.31, 0.43, 0.58, 0.62, 0.67],
                        [0.16, 0.24, 0.33, 0.42, 0.45, 0.47],
                        [0.13, 0.19, 0.26, 0.33, 0.34, 0.36],
                        [0.11, 0.16, 0.21, 0.26, 0.27, 0.29],
                        [0.09, 0.13, 0.17, 0.21, 0.22, 0.23],
                        [0.08, 0.11, 0.14, 0.18, 0.19, 0.19],
                        [0.07, 0.09, 0.12, 0.15, 0.16, 0.16],
                        [0.06, 0.08, 0.11, 0.13, 0.13, 0.14],
                        [0.05, 0.07, 0.09, 0.11, 0.12, 0.12],
                        [0.05, 0.06, 0.08, 0.1 , 0.1 , 0.11],
                        [0.04, 0.06, 0.07, 0.09, 0.09, 0.09],
                        [0.04, 0.05, 0.06, 0.08, 0.08, 0.08],
                        [0.03, 0.05, 0.06, 0.07, 0.07, 0.07],
                        [0.03, 0.05, 0.06, 0.07, 0.07, 0.07]])
    
    power_array = np.array([[0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [  141.16,   140.38,   136.79,   135.57,   135.49,   135.41],
                            [  635.79,   635.76,   635.51,   635.62,   635.61,   635.6 ],
                            [ 1310.51,  1310.49,  1310.05,  1310.25,  1310.24,  1310.22],
                            [ 2216.53,  2216.64,  2216.03,  2216.34,  2216.32,  2216.29],
                            [ 3361.51,  3361.94,  3361.31,  3361.71,  3367.61,  3371.77],
                            [ 3997.41,  4836.84,  4836.43,  4836.86,  4845.23,  4851.17],
                            [ 3999.97,  5993.36,  6680.62,  6680.94,  6692.95,  6701.88],
                            [ 3999.95,  5999.99,  7993.81,  8933.94,  8950.45,  8962.82],
                            [ 3999.99,  5999.98,  7999.97,  9996.91, 10493.57, 10988.26],
                            [ 3999.95,  5999.98,  7999.97,  9999.64, 10499.98, 11000.18],
                            [ 3999.96,  5999.99,  7999.98,  9999.71, 10499.95, 10999.9 ],
                            [ 3999.97,  5999.95,  7999.98,  9999.75, 10499.98, 10999.97],
                            [ 3999.98,  5999.98,  7999.96,  9999.71, 10499.99, 11000.  ],
                            [ 3999.95,  5999.97,  7999.97,  9999.64, 10500.02, 10999.98],
                            [ 3999.96,  5999.96,  7999.97,  9999.78, 10500.28, 11000.31],
                            [ 4000.01,  5999.98,  7999.96,  9999.88, 10500.47, 11000.49],
                            [ 3999.95,  5999.93,  7999.98,  9999.84, 10500.59, 11000.51],
                            [ 3999.96,  5999.97,  7999.97,  9999.81, 10500.33, 11000.18],
                            [ 3999.98,  5999.98,  7999.97,  9999.66, 10500.1 , 11000.01],
                            [ 3999.98,  5999.98,  7999.98,  9999.63, 10500.04, 10999.98],
                            [ 4000.01,  5999.96,  7999.96,  9999.69, 10499.96, 10999.92],
                            [ 4000.01,  5999.96,  7999.96,  9999.69, 10499.96, 10999.92]])
        
    powerCtFunction = PowerCtNDTabular(input_keys=['ws','psp'],
                     value_lst=[ws_lst, psp_lst],
                     power_arr=power_array, power_unit='kW', 
                     ct_arr=ct_array)
    
    # The only output is CT (run_only = 1)
    def predict_output(self, input_parser, powerCtFunction):
        return powerCtFunction(ws=input_parser[0], psp=input_parser[1], run_only=1)

class DTU10MWSurrogateWindTurbine(WindTurbine):

    def __init__(self):
        aersur_path = os.path.dirname(py_wake.__file__)+'/examples/data/dtu10mw/surrogates/all_models/'
        powerCtFunction = PowerSurrogate(TensorflowSurrogate_DTU10MW(aersur_path + "Power_Q50_model", "Power"), 
                                         'kW',
                                          DTU10MWCTTabularSurrogate(), 
                                          input_parser= lambda  ws, psp, ti, Alpha, Air_density: [psp, ti, ws, Alpha, Air_density])
        
        # Surrogates for every load channel
        sensors = ["DEL_Q50_EdgeM_model",
                    "DEL_Q50_FlapM_model", 
                    "DEL_Q50_PitchM_model", 
                    "DEL_Q50_ShaftMx_model",
                    "DEL_Q50_ShaftMz_model",
                    "DEL_Q50_TBFA_model",
                    "DEL_Q50_TBSS_model"]
        
        # Name of every channel
        output_s = ["EdgeM",
                    "FlapM",
                    "PitchM", 
                    "ShaftMx",
                    "ShaftMz",
                    "TBFA", 
                    "TBSS"]
                  
        loadFunction = FunctionSurrogates_DTU10MW_loads([TensorflowSurrogate_DTU10MW(aersur_path + s, n) for s, n in zip(sensors, output_s)] , 
                                          input_parser= lambda ws, psp, ti, Alpha, Air_density: [psp, ti, ws, Alpha, Air_density],
                                          output_keys=output_s)
        
        loadFunction.wohler_exponents = [12, 12, 12, 4, 4, 4, 4]
        
        WindTurbine.__init__(
            self,
            'DTU10MW_psp',
            diameter=178.3,
            hub_height=119,
            powerCtFunction=powerCtFunction,
            loadFunction= loadFunction
            )
        

# %%
def main():
    from py_wake import NOJ
    import matplotlib.pyplot as plt

    wt_surrogate = DTU10MWSurrogateWindTurbine()

    site = Hornsrev1Site()
    sim_res = NOJ(site, wt_surrogate, turbulenceModel=STF2017TurbulenceModel())([0, 1000], [0, 0], #x, y coords
                                                                                ws=np.array([14]),
                                                                                wd = np.array([270]), 
                                                                                psp = np.array([1]), 
                                                                                ti = np.array([0.05]), 
                                                                                Alpha=np.array([0.12]), 
                                                                                Air_density=np.array([1.19]))
    
    load_wd_averaged = sim_res.loads(normalize_probabilities=True, method='OneWT_WDAvg')
    loads = sim_res.loads(normalize_probabilities=True, method='OneWT')
    loads.DEL.isel(sensor=0, wt=0).plot()
    plt.figure()

    for s in load_wd_averaged.sensor:
        print(s.item(), load_wd_averaged.LDEL.sel(sensor=s, wt=0).item(), loads.LDEL.sel(sensor=s, wt=0).item())

    print('Diameter', wt_surrogate.diameter())
    print('Hub height', wt_surrogate.hub_height())
    

    # ws = np.arange(5.5, 25, .1)
    # plt.plot(ws, wt.power(ws) / 1e3, label='DTU10MW tabular')

    # plt.plot(ws, wt_surrogate.power(ws, TI_eff=.05, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=5%')
    # plt.plot(ws, wt_surrogate.power(ws, TI_eff=.1, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=10%')
    # plt.plot(ws, wt_surrogate.power(ws, TI_eff=.2, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=20%')
    # plt.plot(ws, wt_surrogate.power(ws, TI_eff=.4, Alpha=0, yaw=0) / 1e3, label='DTU10MW Surrogate, TI=40%')
    # plt.xlabel('Wind speed [m/s]')
    # plt.xlabel('Power [kW]')
    # plt.legend()
#     ax = plt.twinx()
#     ax.plot(ws, wt.ct(ws), '--', label='DTU10MW tabular')
#     ax.plot(ws, wt_surrogate.ct(ws, TI_eff=.1, Alpha=0, yaw=0), '--', label='DTU10MW Surrogate')

    # plt.figure()
    # yaw = np.arange(-30, 50, 1)
    # theta = np.deg2rad(yaw)
    # ws = np.full(yaw.shape, 11)
    # plt.plot(yaw, wt.power(ws, yaw=theta) / 1e3, label='DTU10MW tabular')
    # plt.plot(yaw, wt_surrogate.power(ws=ws - .18, TI_eff=.1, Alpha=0, yaw=theta) / 1e3, label='DTU10MW Surrogate')
    # plt.xlabel('Yaw misalignment [deg]')
    # plt.xlabel('Power [kW]')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
