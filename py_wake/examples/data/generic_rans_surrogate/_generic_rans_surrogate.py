from py_wake.examples.data import example_data_path
from py_wake.utils.tensorflow_surrogate_utils import TensorFlowModel
from py_wake.deficit_models.deficit_model import SurrogateDeficitModel
from numpy import newaxis as na
import numpy as np
from py_wake.examples.data.iea37._iea37 import IEA37WindTurbines
from py_wake.flow_map import YZGrid
from py_wake.turbulence_models.turbulence_model import SurrogateTurbulenceModel


def get_input(dw_ijlk, hcw_ijlk, z_ijlk, ct_ilk, TI_ilk, h_ilk, D_src_il, **_):
    ['x', 'y', 'z', 'ct', 'ti', 'ratio']
    x, y = dw_ijlk / D_src_il[:, na, :, na], hcw_ijlk / D_src_il[:, na, :, na]
    z = (z_ijlk - h_ilk[:, na]) / D_src_il[:, na, :, na]
    ratio = h_ilk / D_src_il[:, :, na]

    return x, y, z, ct_ilk, TI_ilk, ratio


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.wind_farm_models.engineering_models import All2AllIterative

        from py_wake.flow_map import XYGrid
        wt = IEA37WindTurbines()
        D = wt.diameter()
        site = IEA37Site(16)
        x, y = site.initial_position[:1].T

        def_surrogate = TensorFlowModel.load_h5(example_data_path +
                                                'generic_rans_surrogate/deficits_extra_data_l5_n64_selu.hdf5')
        turb_surrogate = TensorFlowModel.load_h5(example_data_path +
                                                 'generic_rans_surrogate/addedti_extra_data_l5_n64_selu.hdf5')

        deficit_model = SurrogateDeficitModel(def_surrogate, get_input)
        turbulence_model = SurrogateTurbulenceModel(turb_surrogate, get_input)
        wfm = All2AllIterative(site, wt, deficit_model, blockage_deficitModel=deficit_model,
                               turbulenceModel=turbulence_model)
        for ti in [0.01, 0.1]:
            (ax1, ax2), (ax3, ax4) = plt.subplots(2, 2)[1]
            sim_res = wfm([0], [0], wd=270, ws=10, TI=ti)
            x, y = np.linspace(-390, 4700, 500), np.linspace(-480, 480, 100)
            fm = sim_res.flow_map(XYGrid(x, y))

            fm.plot_wake_map(normalize_with=D, ax=ax1)

            fm.plot_ti_map(normalize_with=D, ax=ax3)
            for d in [3, 8, 13, 18]:
                fm = sim_res.flow_map(YZGrid(x=d * D, z=wt.hub_height()))
                ax2.plot(fm.y / D, fm.WS_eff.squeeze(), label=f'{d}D')
                ax4.plot(fm.y / D, fm.TI_eff.squeeze(), label=f'{d}D')

            plt.legend()
            plt.axis('auto')

        plt.show()


main()
