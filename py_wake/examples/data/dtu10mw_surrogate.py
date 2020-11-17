import os
import pickle
import tensorflow as tf

import numpy as np
from numpy import newaxis as na

from py_wake.wind_turbines import OneTypeWindTurbines
# from py_wake.utils.surrogate_utils import Frandsen_TI_ik, Frandsen_WS_ik, Frandsen_pdf_ik
from py_wake.examples.data.dtu10mw import ct_curve
from topfarm.constraint_components.load import (
        SurrogateModel, predict_output)

class DTU10MWSurrogate(OneTypeWindTurbines):
    '''
    '''
    def __init__(self, load_signals, m_list):
        OneTypeWindTurbines.__init__(
            self,
            'DTU10MW',
            diameter=178.3,
            hub_height=119,
            ct_func=self._ct,
            power_func=self._power,
            power_unit='kW',
            power_yaw_model='surrogate')
        path = os.path.join(os.path.dirname(__file__), 'surrogates')
        file_list = [x[0] for x in os.walk(path)]
        file_list.pop(0)
        load_types = dict.fromkeys([os.path.basename(x) for x in file_list])
        for f, l in zip(file_list, load_types):
            model = tf.keras.models.load_model(os.path.join(f, 'model.h5'))
            with open(os.path.join(f, 'save_dic.pkl'), 'rb') as g:
                save_dic = pickle.load(g)
            load_types[l] = SurrogateModel(model, **save_dic)
        self.loads = True
        self.load_types = load_types
        self.load_signals = load_signals
        self.m_list = m_list
        lifetime = float(60 * 60 * 24 * 365 * 20)
        f1zh = 10.0 ** 7.0
        self.lifetime_on_f1zh = lifetime / f1zh



    def get_output(self, key, u, yaw, ti, alpha):
        u = np.asarray(u)
        yaw = np.broadcast_to(yaw, u.shape)
        ti = np.broadcast_to(ti, u.shape)
        alpha = np.broadcast_to(alpha, u.shape)
        surrogate = self.load_types[key]
        output, _ = predict_output(
                model=surrogate.model,
                input={
                    'TI': ti.ravel() * 100, # surrogate is in %, pywake is in decimal
                    'Alpha': alpha.ravel(),
                    'U': u.ravel(),
                    'Yaw': np.degrees(yaw).ravel(), # surrogate is in deg, pywake-windturbines is in radians
                    },
                model_in_keys=surrogate.input_channel_names,
                input_scaler=surrogate.input_scaler,
                output_scaler=surrogate.output_scaler
                )
        return output

    def _ct(self, u):
        return np.interp(u, ct_curve[:, 0], ct_curve[:, 1])

    def _power(self, u, yaw, ti, alpha=0.2):
        return self.get_output('Power', u, yaw, ti, alpha)
    
    def get_loads(self, WS_eff_ilk, TI_eff_ilk, yaw_ilk, pdf_ilk, alpha, normalize_probabilities):
        if normalize_probabilities:
            norm = pdf_ilk.sum((1, 2))[:, na, na]
        else:
            norm = 1
        print(alpha)
        shape_ilk = WS_eff_ilk.shape
        alpha = np.broadcast_to(alpha, shape_ilk)
        pdf_ilk = np.broadcast_to((pdf_ilk / norm), shape_ilk) # check that this sums to 1
        loads = {}
        for (ls, m) in zip(self.load_signals, self.m_list):
            surrogate = self.load_types[ls]
            DEL_ilk, _ = predict_output(
                    model=surrogate.model,
                    input={
                        'TI': (TI_eff_ilk.ravel()) * 100,# surrogate is in %, pywake is in decimal
                        'Alpha': alpha.ravel(),
                        'U': WS_eff_ilk.ravel(),
                        'Yaw': yaw_ilk.ravel(),# surrogate is in deg, pywake-site is in deg
                        },
                    model_in_keys=surrogate.input_channel_names,
                    input_scaler=surrogate.input_scaler,
                    output_scaler=surrogate.output_scaler)
            LDEL = (((pdf_ilk * (DEL_ilk.reshape(shape_ilk)/DEL_ilk.mean()) ** m).sum((1, 2)) * self.lifetime_on_f1zh) ** (1/m))*DEL_ilk.mean()

            loads['DEL_' + ls] = {
                'values': DEL_ilk.reshape(shape_ilk),
                'dims': ['wt', 'wd', 'ws'],
                'descr': 'Damage Equivalent Load',}

            loads['LDEL_' + ls] = {
                'values': LDEL.ravel(),
                'dims': ['wt'],
                'descr': 'Lifetime Damage Equivalent Load',}
        return loads
        


def main():
    wt = DTU10MWSurrogate(load_signals=['Blade_root_flapwise_M_x'])
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    for yaw in np.arange(-40, 50, 10):
        plt.plot(ws, wt.power(ws, yaw_i=yaw), '.-',label='yaw'+str(yaw))
    plt.legend()
    plt.ylabel('Power')
    plt.xlabel('Wsp')
    plt.show()
    for yaw in np.arange(-40, 50, 10):
        plt.plot(ws, wt.ct(ws, yaw_i=np.radians(yaw)), '.-',label='yaw'+str(yaw))
    plt.legend()
    plt.ylabel('Ct')
    plt.xlabel('Wsp')
    plt.show()

if __name__ == '__main__':
    main()
