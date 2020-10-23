import os
import pickle
import numpy as np
import tensorflow as tf

from py_wake.wind_turbines import OneTypeWindTurbines
from topfarm.constraint_components.load import (
        SurrogateModel, predict_output)

class DTU10MWSurrogate(OneTypeWindTurbines):
    '''
    '''
    def __init__(self):
        OneTypeWindTurbines.__init__(
            self,
            'DTU10MW',
            diameter=178.3,
            hub_height=119,
            ct_func=self._ct,
            power_func=self._power,
            power_unit='kW',
            ct_power_yaw_models='surrogate')
        path = r'.\surrogates'
        file_list = [x[0] for x in os.walk(path)]
        file_list.pop(0)
        load_types = dict.fromkeys([os.path.basename(x) for x in file_list])
        for f, l in zip(file_list, load_types):
            model = tf.keras.models.load_model(os.path.join(f, 'model.h5'))
            with open(os.path.join(f, 'save_dic.pkl'), 'rb') as g:
                save_dic = pickle.load(g)
            load_types[l] = SurrogateModel(model, **save_dic)
        self.load_types = load_types


    def get_output(self, key, u, yaw, ti, alpha):
        u = np.asarray(u)
        yaw = np.broadcast_to(yaw, u.shape)
        ti = np.broadcast_to(ti, u.shape)
        alpha = np.broadcast_to(alpha, u.shape)
        surrogate = self.load_types[key]
        output, _ = predict_output(
                model=surrogate.model,
                input={
                    'TI': ti.ravel(),
                    'Alpha': alpha.ravel(),
                    'U': u.ravel(),
                    'Yaw': yaw.ravel(),
                    },
                model_in_keys=surrogate.input_channel_names,
                input_scaler=surrogate.input_scaler,
                output_scaler=surrogate.output_scaler
                )
        return output

    def _ct(self, u, yaw, ti, alpha=0.2):
        return self.get_output('Ct', u, yaw, ti, alpha)

    def _power(self, u, yaw, ti, alpha=0.2):
        return self.get_output('Power', u, yaw, ti, alpha)


def main():
    wt = DTU10MWSurrogate()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    for yaw in np.arange(-40, 50, 10):
        plt.plot(ws, wt.power(ws, yaw_i=yaw), '.-')
    plt.show()
    for yaw in np.arange(-40, 50, 10):
        plt.plot(ws, wt.ct(ws, yaw_i=yaw), '.-')
    plt.show()

if __name__ == '__main__':
    main()
