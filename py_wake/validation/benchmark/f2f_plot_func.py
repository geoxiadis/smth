import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import shutil

# PyWake objects
from py_wake.validation.benchmark import validation_utils
from py_wake.validation.benchmark.validation_utils import wffm_kwargs, GaussianFilter
import py_wake.examples.data.nysted as nysted
import py_wake.examples.data.roedsand2 as roedsand2
from py_wake.wind_turbines import WindTurbines

data_path = os.path.dirname(validation_utils.__file__) + '/data/'
luts_path = os.path.dirname(validation_utils.__file__) + '/fuga_files'

# Loading measurements and RANS models, set up labels and colors
datafile = data_path + 'NystedRoedsand2_WFdata_WFeff.dat'
modelfiles = []
modellabels = []
casenames = ['R2', 'R2+N']
cases = ['Roedsand2', 'NystedRoedsand2']
turbnames = ['ASL']
turb_models = ['kefP']
for n in range(len(turb_models)):
    for m in range(len(casenames)):
        filename = data_path + cases[m] + '_' + turb_models[n] + '_RANS_WFeff.dat'
        modelfiles.append(filename)
        modellabels.append(casenames[m] + ' RANS')

cRANS = ['--g', 'g']
linewidth = 1.5

path_roedsand = luts_path + '/Nysted_Roedsand/LUTs_Zeta0=0.0000000000_16_32_D93_zhub68.5_zi400_z0=0.00010000_z19.9-208.4_UL_nx2048_ny512_dx23.25_dy5.8125.nc'
lut_path_nysted = luts_path + '/Nysted_Roedsand/LUTs_Zeta0=0.0000000000_16_32_D82.4_zhub69_zi400_z0=0.00010000_z19.9-208.4_UL_nx2048_ny512_dx20.6_dy5.15.nc'
path_NystedRoedsand = [path_roedsand, lut_path_nysted]


def ws_probability_in_bin(ws_arr, site=None, wd=None, Weibull_k=None, Weibull_A=None):
    """ For a list of wind speeds (e.g., 8+-3 m/s), return probabilities of each wind speed bin, following the
    weibull distribution at the site.

    Parameters
    ----------

    ws_arr: array_like
        array of ws values included in the bin
    site: Site
        Site object with Weibull k and A params
    wd: float
        wd to pull Weibull k and A params from
    Weibull_k: float or None
        weibull shape param, if site and wd is not provided
    Weibull_A: float or  None
        weibull scale param, if site and wd is not provided
    """
    if site:
        Weibull_k = site.__getstate__()['ds'].sel(wd=wd, method='nearest')['Weibull_k'].values
        Weibull_A = site.__getstate__()['ds'].sel(wd=wd, method='nearest')['Weibull_A'].values
    wsprobs = (Weibull_k / Weibull_A) * (ws_arr / Weibull_A) ** (Weibull_k - 1) * (
        np.exp(-(ws_arr / Weibull_A) ** Weibull_k))
    wsprobs /= wsprobs.sum()  # normalize probabilities in the array
    return wsprobs


# =============================================================================
# %% Farm to Farm interaction - Nysted and Rødsand II
# =============================================================================

def plotF2FCase(wffms, plotGAonly=False, save_plot=False, fullrose=False, weibull_weight_ws=False,
                target_site='Roedsand2'):
    """ Plot the impact of the Nysted wind farm on the efficiency of Rødsand II

    Parameters
    ----------

    wffms : dict
        dictionary containing user-specified wind farm models and corresponding plot color-labels
    cRANS : str
        color specified for RANS results
    plotGAonly : bool
        whether to use or not Gaussian averaging in the power calculation
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    fullrose : bool
        whether to use the full wind resource rose for the study
    weibull_weight_wsbin : bool
        simulation results for each wind speed in the experimental wind speed bin is weighted using weibull
        distribution. If false, wind speed bin is assumed uniformly distributed
    target_site : string
        name of the farm to study the effect of wake from neighbouring wind farm
    """

    warnings.filterwarnings("ignore")
    linewidth = 1.5

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
    ax_layout = fig.add_axes([0.7, 0.3, 0.25, 0.25], frameon=True)

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
        if fullrose:  # Make models periodic
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

    wdir = np.arange(modelp[0, 0], modelp[-1, 0], 1)  # wd range to simulate
    cases = {
        'Roedsand2': {
            'name': 'Roedsand2',
            'shortname': 'R2',
            'wt': roedsand2.SWT2p3_93(),
            'site': roedsand2.Roedsand2Site(),
            'wt_x': roedsand2.wt_x,
            'wt_y': roedsand2.wt_y,
            'type': [0] * len(roedsand2.wt_x),
            'U0': 8,
            'U0binwidth': 3,  # ws = 8 +/- 3 m/s bin
            'location': 'offshore',
            'TItot': roedsand2.ti,
            'fugaPath': path_roedsand,
            'linestyle': '--',
        },
        # 'Nysted': {
        #     'skip': None,
        #     'name': 'Nysted',
        #     'shortname': 'N',
        #     'wt': nysted.Bonus_2p3(),
        #     'site': nysted.NystedSite(),
        #     'wt_x': nysted.wt_x,
        #     'wt_y': nysted.wt_y,
        #     'type': [1] * len(nysted.wt_x),
        #     'U0': 8,
        #     'U0binwidth': 3,
        #     'location': 'offshore',
        #     'TItot': nysted.ti,
        #     'fugaPath': lut_path_nysted,
        #     'linestyle': '-.',
        # },
        'NystedRoedsand2': {
            'name': 'NystedRoedsand2',
            'shortname': 'R2+N',
            'wt': WindTurbines.from_WindTurbine_lst([roedsand2.SWT2p3_93(), nysted.Bonus_2p3()]),
            'site': roedsand2.Roedsand2Site(),
            'wt_x': roedsand2.wt_x + nysted.wt_x,
            'wt_y': roedsand2.wt_y + nysted.wt_y,
            'type': [0] * len(roedsand2.wt_x) + [1] * len(nysted.wt_x),
            'U0': 8,
            'U0binwidth': 3,
            'location': 'offshore',
            'TItot': roedsand2.ti,
            'lut_paths_lst': path_NystedRoedsand,
            'linestyle': None,
        },
    }

    # =============================================================================
    # Plots of engineering models
    # =============================================================================

    for case, case_param in cases.items():
        if 'skip' not in case_param:
            for k, (wakemodel, wffm_param) in enumerate(wffms.items()):
                wffm = wffm_kwargs(case_param, wakemodel, )
                wsbin = np.arange(case_param['U0'] - case_param['U0binwidth'],
                                  case_param['U0'] + case_param['U0binwidth'] + 1, 1)
                sim_res = wffm(x=case_param['wt_x'], y=case_param['wt_y'], type=case_param['type'], ws=wsbin, wd=wdir,
                               TI=case_param['TItot'])
                sim_res = sim_res.set_xindex('type').sel(type=0)  # only use the results from Roedsand2 (type=0)
                power_ilk = sim_res.power_ilk
                if weibull_weight_ws:
                    ws_weights = ws_probability_in_bin(wsbin, case_param['site'], np.median(wdir))
                    for i, ws in enumerate(wsbin):
                        power_ilk[:, :, i] *= ws_weights[i]
                        norm_r = len(cases[target_site]['wt_x']) * cases[target_site]['wt'].power(ws) * ws_weights[i]
                        power_ilk[:, :, i] /= norm_r  # normalize by rated power
                    power_ilk = np.mean(power_ilk, axis=2)
                else:  # assumes wind speed probability is uniform within the wind speed bin
                    power_ilk = np.mean(power_ilk, axis=2)  # mean on ws axis
                    norm_r = len(cases[target_site]['wt_x']) * cases[target_site]['wt'].power(
                        wsbin).mean()  # rated (farm) power
                    power_ilk = power_ilk / norm_r  # normalize by rated power

                power_ilk = np.reshape(power_ilk, (len(cases[target_site]['wt_x']), len(wdir), 1))
                if plotGAonly:  # pass power_ilk from wd gaussian filter
                    for i in range(len(cases[target_site]['wt_x'])):
                        power_ilk[i, :, 0] = GaussianFilter(power_ilk[i, :, 0],
                                                            wdir,
                                                            3, sigma=5)
                WFeff = power_ilk.sum(axis=0)
                linep, = ax.plot(wdir, WFeff, wffm_param['color'], ls=case_param['linestyle'],
                                 label=f'{case_param["shortname"]} {wffm_param["label"]}')
    ax.set_xlabel('Wind direction [deg]')
    ax.set_ylabel('Rødsand 2 efficiency')
    ax.grid(True)
    ax.set_ylim(0.3, 1.0)
    xlim_offset = 10
    ax.set_xlim(datap[0, 0] - xlim_offset, datap[-1, 0] + xlim_offset)

    # Plot WF layout and arrow
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    coastline = np.load(f'{data_path}nystedroedsand2_coastline.npy')
    coastline /= 1e3
    ax_layout.plot(coastline[0], coastline[1], color='gray')
    layout_x = np.array(cases['NystedRoedsand2']['wt_x']) / 1e3
    layout_y = np.array(cases['NystedRoedsand2']['wt_y']) / 1e3
    ax_layout.scatter(layout_x, layout_y, s=1.5, color='r')
    ax_layout.annotate('Rødsand II', (layout_x.mean() - 4, layout_y.max() - 1))
    ax_layout.annotate('Nysted', (layout_x.max() - 3, layout_y.max() - 1))
    ax_layout.set_aspect(1)
    ax_layout.set_xticks([])
    ax_layout.set_yticks([])
    ax_layout.arrow(layout_x.min() + 1, layout_y.min() + 1,
                    1 * np.cos((270.0 - wdir) / 180.0 * np.pi).mean(),
                    1 * np.sin((270.0 - wdir) / 180.0 * np.pi).mean(), head_width=0.5, head_length=0.3, color='r')
    circle1 = plt.Circle((layout_x.min() + 1, layout_y.min() + 1), 1.3, color='k', fill=False)
    ax_layout.add_artist(circle1)
    pos1 = ax.get_position()
    pos2 = ax_layout.get_position()
    pos3 = [pos1.x0, pos1.y0, pos2.width * 1.3, pos2.height * 1.3]
    ax_layout.set_position(pos3)

    ax_layout.set_xlim(layout_x.min() - 1, layout_x.max() + 14)
    ax_layout.set_ylim(layout_y.min() - 1, layout_y.max() + 10)

    legend_ncol = len(wffms) + 2
    lines, labels = ax.get_legend_handles_labels()
    # Sort Legend
    line_label_pairs = list(zip(lines, labels))
    line_label_pairs.sort(key=lambda pair: pair[1])
    sorted_lines, sorted_labels = zip(*line_label_pairs)

    fig.legend(sorted_lines, sorted_labels, ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0, 0, 1.05, 0.93),
               numpoints=1,
               scatterpoints=1)

    if save_plot is True:
        if os.path.isdir('report/figures'):
            shutil.rmtree('report/figures')
        os.mkdir('report/figures')
        filename = 'Nysted_Roedsand2_WFeff_rect.' + 'pdf'
        fig.savefig(filename)
        shutil.copyfile(filename, 'report/figures/' + filename)
    else:
        fig.suptitle('Effect of Nysted on Rødsand 2')
        plt.show()


def main():
    # engineering wind farm models
    from py_wake.literature.noj import Jensen_1983
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    from py_wake.literature.fuga import Ott_Nielsen_2014
    from py_wake.literature.fuga import Ott_Nielsen_2014_Blockage
    from py_wake.literature.gaussian_models import Niayifar_PorteAgel_2016
    from py_wake.literature.gaussian_models import Zong_PorteAgel_2020
    from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
    from py_wake.literature.cumulative_sum import CumulativeWake
    from py_wake.literature.turbopark import Nygaard_2022

    wffms = {Jensen_1983: {'label': 'NOJ', 'color': 'deepskyblue'},
             Niayifar_PorteAgel_2016: {'label': 'NIAY', 'color': 'brown'},
             Bastankhah_PorteAgel_2014: {'label': 'GAU', 'color': 'r'},
             Blondel_Cathelain_2020: {'label': 'SUPG', 'color': 'b'},
             Nygaard_2022: {'label': 'TURBOG', 'color': 'purple'},
             CumulativeWake: {'label': 'CGAU', 'color': 'coral'}
             }
    plotF2FCase(wffms, plotGAonly=False, weibull_weight_ws=False)


if __name__ == '__main__':
    main()
