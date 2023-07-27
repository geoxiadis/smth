from py_wake.examples.data import example_data_path
from py_wake.utils.tensorflow_surrogate_utils import TensorFlowModel
from py_wake.deficit_models.deficit_model import SurrogateDeficitModel
from numpy import newaxis as na
import numpy as np


def get_input(dw_ijlk, hcw_ijlk, z_ijlk, ct_ilk, TI_ilk, h_ilk, D_src_il, **_):
    ['x', 'y', 'z', 'ct', 'ti', 'ratio']
    ratio = h_ilk / D_src_il[:, :, na]
    return dw_ijlk, hcw_ijlk, z_ijlk, ct_ilk, TI_ilk, ratio


def get_output(output_ijlk, **_):
    return output_ijlk


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        plt.figure()
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.wind_farm_models.engineering_models import All2AllIterative
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        from py_wake.flow_map import XYGrid
        from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
        wt = DTU10MW()
        D = wt.diameter()
        site = IEA37Site(16)
        x, y = site.initial_position[:1].T

        model = TensorFlowModel.load_h5(example_data_path +
                                        'generic_rans_surrogate/deficits_extra_data_l5_n64_selu.hdf5')

        deficit_model = SurrogateDeficitModel(model, get_input, get_output)
        wfm = All2AllIterative(site, wt, deficit_model, blockage_deficitModel=deficit_model,
                               turbulenceModel=STF2017TurbulenceModel())
        sim_res = wfm([0], [0], wd=270, ws=10)
        x, y = np.linspace(-390, 4700, 500), np.linspace(-480, 480, 100)
        fm = sim_res.flow_map(XYGrid(x, y))
        fm.plot_wake_map(normalize_with=D)
        plt.show()
        #
        # plt.figure()
        # for d in [1, 3, 5]:
        #     fm.WS_eff.interp(x=d * D).plot(label=f'{d}D downstream')
        # plt.legend()
        # plt.grid()
        # plt.show()


main()
