import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import shutil

# PyWake objects
from py_wake.validation.benchmark import validation_utils
from py_wake.validation.benchmark.validation_utils import kwargs_func_f2f, GaussianFilter

from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.examples.data.nysted import wt_x as n_wt_x
from py_wake.examples.data.nysted import wt_y as n_wt_y
from py_wake.examples.data.roedsand2 import wt_x as r_wt_x
from py_wake.examples.data.roedsand2 import wt_y as r_wt_y
from py_wake.examples.data.nysted import Bonus_2p3
from py_wake.examples.data.roedsand2 import SWT2p3_93
from py_wake.wind_turbines import WindTurbines

data_path = os.path.dirname(validation_utils.__file__) + '/data/'
luts_path = os.path.dirname(validation_utils.__file__) + '/validation/fuga_files'

# Loading measurements and RANS models, set up labels and colors
datafile = data_path + 'NystedRoedsand2_WFdata_WFeff.dat'
modelfiles = []
modellabels = []
casenames = ['R2', 'R2 + N']
cases = ['Roedsand2', 'NystedRoedsand2']
turbnames = ['ASL']
turb_models = ['kefP']
for n in range(len(turb_models)):
    for m in range(len(casenames)):
        filename = data_path + cases[m] + '_' + turb_models[n] + '_RANS_WFeff.dat'
        modelfiles.append(filename)
        modellabels.append(casenames[m] + ' ' + turbnames[n])

cRANS = ['--g', 'g']
linewidth = 1.5

path_roedsand = luts_path + '/Nysted_Roedsand/LUTs_Zeta0=0.0000000000_16_32_D93_zhub68.5_zi400_z0=0.00010000_z19.9-208.4_UL_nx2048_ny512_dx23.25_dy5.8125.nc'

path_NystedRoedsand = luts_path + '/Nysted_Roedsand/*.nc'

# =============================================================================
# %% Farm to Farm interaction - Nysted and Rødsand II
# =============================================================================


def plotF2FCase(wffm_list, wm_labels, linewidth, cRANS, colors, plotGAonly=False, save_plot=False, fullrose=False):
    """
    Plot the impact of the Nysted wind farm on the efficiency of Rødsand II

    Parameters
    ----------

    wffm_list : list
        list of user-specified wind farm models used for the benchmarking
    wm_labels : labels
        labels for the wind farm models
    linewidth : int or float
        line width used for the plots
    cRANS : str
        color specified for RANS results
    colors : str
        list of colors for the engineering wind farm models
    plotGAonly : bool
        whether to use or not Gaussian averaging in the power calculation
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    fullrose : bool
        whether to use the full wind resource rose for the study

    """

    warnings.filterwarnings("ignore")

    site = Hornsrev1Site(ti=0.07/0.8)

    wt_x = r_wt_x + n_wt_x      # nysted and roedsand
    wt_y = r_wt_y + n_wt_y      # nysted and roedsand

    # Both Nysted and Rødsand wind turbines
    wt = WindTurbines.from_WindTurbine_lst([Bonus_2p3(), SWT2p3_93()])
    wt._names = ["Nysted", "RodsandII"]

    # Only Rødsand wind turbine
    wt_r = SWT2p3_93()

    # Rødsand turbine's diameter used as reference
    D = wt_r.diameter

    # =============================================================================
    # Load data and model results
    # =============================================================================

    data = np.genfromtxt(datafile, skip_header=True)
    models = []
    for modefile in modelfiles:
        models.append(np.genfromtxt(modefile, skip_header=True))
    Nmodels = len(models)

    if fullrose:
        # Make data periodic
        datap = np.append(data[:, :], [data[0, :]], axis=0)
    else:
        datap = data

    # =============================================================================
    # Plotting
    # =============================================================================

    fig, ax = plt.subplots(figsize=(15, 6), facecolor='white')

    # =============================================================================
    # References, based on measurements
    # =============================================================================

    # Number of bins is expected to be a column
    ldata = ax.fill_between(datap[:, 0],
                            datap[:, 1] - datap[:, 2] / np.sqrt(datap[:, 3]),
                            datap[:, 1] + datap[:, 2] / np.sqrt(datap[:, 3]), color='k', alpha=0.3, label='Data')

    # =============================================================================
    # RANS,  based on EllipSys3D AD k-epsilon-fP
    # =============================================================================

    for i in range(Nmodels):
        if fullrose:
            # Make models periodic
            modelp = np.append(models[i][:, :], [models[i][0, :]], axis=0)
        else:
            modelp = models[i]
        if not plotGAonly:
            label = modellabels[i]
            if cRANS[i][0:2] == '--':
                dash = [5, 2]
            else:
                dash = [1, 0]
            mls, = ax.plot(modelp[:, 0], modelp[:, 1], cRANS[i], dashes=dash, linewidth=linewidth, label=label)

        else:
            label = modellabels[i] + ' GA'
            if cRANS[i][0:2] == '--':
                dash = [5, 2]
            else:
                dash = [1, 0]
            mls, = ax.plot(modelp[:, 0], modelp[:, 1], cRANS[i], dashes=dash, linewidth=linewidth, label=label)

    # =============================================================================
    # Engineering Wind Farm Models
    # =============================================================================

    # Rødsand II case

    # Setting up the Site, WindTurbine and other parameters needed for the engineering wind farm models
    kwargs_list_r, _ = kwargs_func_f2f('Roedsand', wffm_list, site, wt_r, path_roedsand)

    # ===== Instantiating wind farm models from literature =====
    wakemodels_r = []
    for wfm, kwargs in zip(wffm_list, kwargs_list_r):
        wakemodels_r.append(wfm(**kwargs))

    # Nysted + Rødsand II case

    # ===== Instantiating wind farm models from literature, Nysted + Rødsand II =====
    # TODO: check memory error for fuga

    # Setting up the Site, WindTurbine and other parameters needed for the engineering wind farm models
    _, wakemodels_nr = kwargs_func_f2f('Nysted_Roedsand', wffm_list, site, wt, path_NystedRoedsand)

    # =============================================================================
    # Plots of engineering models
    # =============================================================================

    wdir = modelp[:, 0]  # wind directions used in the study (corresponding to the wind directions available in the data)

    # ONLY RØDSAND II
    power_models_r = []
    for k, wakemodel in enumerate(wakemodels_r):
        if not plotGAonly:
            power_ilk = wakemodel(r_wt_x, r_wt_y, ws=8, wd=wdir).power_ilk
            power_models_r.append(power_ilk)
            label = 'RII '
        else:
        # TODO: check Gaussian Averaging
            power_ilk = wakemodel(r_wt_x, r_wt_y, ws=8, wd=wdir).power_ilk
            powerGA = np.zeros(power_ilk.shape)
            for i in range(len(r_wt_x)):
                powerGA[i, :, 0] = GaussianFilter(power_ilk[i, :, 0],
                                                  wdir,
                                                  3, sigma=5)
            power_models_r.append(powerGA)
            label = 'RII GA '

    # NYSTED + RØDSAND II
    power_models_nr = []

    for k, wakemodel in enumerate(wakemodels_nr):
        # TODO: when using fuga, power_ilk returns NaN
        if not plotGAonly:
            power_ilk = wakemodel(wt_x, wt_y, ws=8, wd=wdir).power_ilk
            power_models_nr.append(power_ilk)
        # else:
        # Gaussian Averaging
            power_ilk = wakemodel(wt_x, wt_y, ws=8, wd=wdir).power_ilk
            powerGA = np.zeros(power_ilk.shape)
            for i in range(len(wt_x)):
                powerGA[i, :, 0] = GaussianFilter(power_ilk[i, :, 0],
                                                  wdir,
                                                  3, sigma=5)
            power_models_nr.append(powerGA)

    norm_r = len(r_wt_x) * wt_r.power(8)
    norm_nr = len(wt_x) * wt_r.power(8)

    for k in range(len(wakemodels_r)):
        WFeff = power_models_r[k].sum(axis=0) / norm_r
        l1_r, = ax.plot(modelp[:, 0], WFeff.squeeze(), '--', color=colors[k], linewidth=linewidth, label=label + wm_labels[k])

    for k in range(len(wakemodels_nr)):
        WFeff = power_models_nr[k].sum(axis=0) / norm_nr
        l1_nr, = ax.plot(modelp[:, 0], WFeff, color=colors[k], linewidth=linewidth, label=' RII + N ' + wm_labels[k])

    ax.set_xlabel('Wind direction [deg]')
    ax.set_ylabel('Wind farm efficiency')
    ax.grid(True)
    ax.set_ylim(0.3, 1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    ncol = len(wakemodels_r) + 3

    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, ncol=ncol, loc='upper center', bbox_to_anchor=(0, 0, 1.05, 0.93), numpoints=1, scatterpoints=1)

    if save_plot is True:
        filename = 'Nysted_Roedsand2_WFeff_rect.' + 'pdf'
        fig.savefig(filename)
        shutil.copyfile(filename, 'report/figures/' + filename)
    else:
        fig.suptitle('Nysted and Rødsand II')
        plt.show()
