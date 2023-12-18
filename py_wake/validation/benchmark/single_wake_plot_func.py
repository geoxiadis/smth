import os
import shutil
import matplotlib
import matplotlib.style
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# PyWake objects
from py_wake.validation.benchmark import validation_utils
from py_wake.validation.benchmark.validation_utils import integrate_velocity_deficit_arc, kwargs_func_SW, kwargs_func_SWiFt, get_downstream_label
from py_wake.flow_map import HorizontalGrid
from py_wake.examples.data.v27swift import V27SWIFT
from py_wake import np

# path to reference data
data_path = os.path.dirname(validation_utils.__file__) + '/data/'
SWiFTlidar_path = os.path.dirname(validation_utils.__file__) + '/data/SWiFT_lidar_data/'

# =============================================================================
# %% Single Wake cases - including data, LES, RANS and different wake models
# =============================================================================


def deficitPlotSingleWakeCases(SingleWakeCases, wffm_list, wm_labels, linewidth, cLES, cRANS, colors, save_plot=False):
    """
    Plot velocity deficit for each single wake case

    Parameters
    ----------
    SingleWakeCases : dict
        dictionary containing all parameters for the single wake cases
    wffm_list : list
        list of user-specified wind farm models used for the benchmarking
    wm_labels : labels
        labels for the wind farm models
    linewidth : int or float
        line width used for the plots
    cLES : str
        color specified for LES results
    cRANS : str
        color specified for RANS results
    colors : str
        list of colors for the engineering wind farm models
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    """

    UdefCases = []

    for case in SingleWakeCases:

        # Wind directions used in the study (corresponding to the wind directions available in the data)
        wds = np.linspace(-30, 30, 61)

        # downstream distances normalized by diameter
        if case['name'] == 'Nordtank-500':
            xDown = case['xDown'] * 40.0
        else:
            xDown = case['xDown'] * case['D']

        # =============================================================================
        # Engineering Wind Farm Models
        # =============================================================================

        # Setting up the Site, WindTurbine and other parameters needed for the engineering wind farm models
        kwargs_list = kwargs_func_SW(case, wffm_list)

        # ===== Instantiating wind farm models from literature =====
        wakemodels = []
        for wfm, kwargs in zip(wffm_list, kwargs_list):
            wakemodels.append(wfm(**kwargs))

        wake_ws = np.zeros((len(wakemodels), len(wds), len(xDown)))

        for k, wakemodel in enumerate(wakemodels):
            # The velocity deficit at an arc is calculated by running the iwake_map for each point.
            # because it is only possible to provide a x and y array that define a rectangular plane.
            for i in range(len(wds)):
                for j in range(len(xDown)):
                    # positions of the turbines
                    x = xDown[j] * np.cos(wds[i] / 180.0 * np.pi)
                    y = xDown[j] * np.sin(wds[i] / 180.0 * np.pi)

                    WS_eff = wakemodel(x=[0.0],
                                       y=[0.0],
                                       wd=[270.0],
                                       ws=case['U0']).flow_map(HorizontalGrid(x=x, y=y)).WS_eff.squeeze()

                    # Effective wind speed for each wakemodel, wind direction and downstream distance
                    wake_ws[k, i, j] = WS_eff

        # =============================================================================
        # Plotting
        # =============================================================================

        fig, ax = plt.subplots(1, len(xDown), sharey=False, figsize=(8 * len(xDown), 5), facecolor='white')

        # Instatiating variable for storing integrated momentum deficit
        Udef = np.zeros((len(xDown), len(wakemodels) + 3))
        #Udef[:, :] = np.nan

        for j in range(len(xDown)):

            if case['xDown'][j] % 2 == 0 or (case['xDown'][j] + 1) % 2 == 0:
                xDownlabel = str(int(case['xDown'][j]))
            else:
                xDownlabel = str(case['xDown'][j]).replace('.', 'p')

            # =============================================================================
            # Loading and plotting data, based on measurements
            # =============================================================================

            # For both Wieringermeer cases, the middle wind direction is substracted and the data is normalized with the reference wind speed U0. Data is represented as error bars with the standard deviation provided from a 10min period.
            # The integrated momentum deficit is calculated using the "integrated_velocity_deficit_arc" function.

            if case['name'] == 'Wieringermeer-West' and j == 1:
                data = np.genfromtxt(data_path + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0] - 315, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                                       np.sqrt(data[:, 3]), color='k', elinewidth=1.0, linewidth=0,
                                       marker='o', zorder=0, markersize=4, label='Data')
                fdata = interp1d(data[:, 0] - 315, data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])

            elif case['name'] == 'Wieringermeer-East' and j == 0:
                data = np.genfromtxt(data_path + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0] - 31, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                                       np.sqrt(data[:, 3]), color='k', elinewidth=1.0, linewidth=0,
                                       marker='o', zorder=0, markersize=4, label='Data')
                fdata = interp1d(data[:, 0] - 31, data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])

            elif case['name'] == 'Nibe':
                # No standard deviation of the 10 min. available.
                data = np.genfromtxt(data_path + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].scatter(data[:, 0], data[:, 1], color='k', marker='o', zorder=0, s=10, label='Data')
                fdata = interp1d(data[:, 0], data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds) * case['U0'], case['xDown'][j], case['U0'])

            elif case['name'] == 'Nordtank-500' and j < 2:
                data = np.genfromtxt(data_path + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0], data[:, 2], yerr=data[:, 3] / np.sqrt(74.0),
                                       color='k', elinewidth=1.0, linewidth=0, marker='o',
                                       zorder=0, markersize=4, label='Data')

            # =============================================================================
            # Loading and plotting LES data, based on Ellipsys3D AD
            # =============================================================================

            LES = np.genfromtxt(data_path + case['name'] + '_LES_' + xDownlabel + 'D.dat')
            # Shaded area represents the standard error of the mean
            lLES = ax[j].fill_between(LES[:, 0], LES[:, 1] - LES[:, 2] / np.sqrt(LES[:, 3]),
                                      LES[:, 1] + LES[:, 2] / np.sqrt(LES[:, 3]), color=cLES, alpha=0.5, label='LES')

            # =============================================================================
            # Loading and plotting RANS data,  based on EllipSys3D AD k-epsilon-fP
            # =============================================================================

            RANS = np.genfromtxt(data_path + case['name'] + '_RANS_' + xDownlabel + 'D.dat')
            lRANS, = ax[j].plot(RANS[:, 0], RANS[:, 1], color=cRANS, linewidth=linewidth, label='RANS')

            # =============================================================================
            # Plotting engineering models
            # =============================================================================

            for k in range(len(wakemodels)):
                vel_deficit = wake_ws[k, :, j] / case['U0']   # velocity deficit
                l1, = ax[j].plot(wds, vel_deficit, color=colors[k], linewidth=linewidth, label=wm_labels[k])

                if case['name'] == 'Nordtank-500':
                    title = '%s %g %s' % ('$x=', case['xDown'][j], 'D^*$')
                else:
                    title = '%s %g %s' % ('$x=', case['xDown'][j], 'D$')
                ax[j].set_title(title)
                ax[j].set_xticks(np.arange(min(wds), max(wds) + 10.0, 10.0))
            if case['xDown'][j] < 7.0:
                ax[j].set_xlim(-30, 30)
            else:
                ax[j].set_xlim(-20, 20)
            ax[j].grid(True)
            ax[j].set_ylim(None, ymax=1.1)

            # ===== Momentum deficit for each wake model =====
            for k in range(len(wakemodels)):
                Udef[j, 3 + k] = integrate_velocity_deficit_arc(wds, wake_ws[k, :, j], case['xDown'][j], case['U0'])
            Udef[j, 1] = integrate_velocity_deficit_arc(LES[:, 0], LES[:, 1] * case['U0'], case['xDown'][j], case['U0'])
            Udef[j, 2] = integrate_velocity_deficit_arc(RANS[:, 0], RANS[:, 1] * case['U0'], case['xDown'][j], case['U0'])

        ax[0].set_ylabel('$U/U_0$', rotation=0)
        ax[0].yaxis.labelpad = 20
        ax[1].set_xlabel('Relative wind direction [deg]')
        fig.tight_layout(rect=[0, 0, 1, 0.85])

        ncol = len(wakemodels) + 3

        if case['name'] == 'Wieringermeer-West':
            lines, labels = ax[1].get_legend_handles_labels()
        else:
            lines, labels = ax[0].get_legend_handles_labels()

        fig.legend(lines, labels, ncol=ncol, loc='upper center', bbox_to_anchor=(0, 0, 1.05, 0.93), numpoints=1, scatterpoints=1)

        if save_plot is True:
            if os.path.isdir('report/figures'):
                shutil.rmtree('report/figures')
            os.mkdir('report/figures')
            filename = case['name'] + '.pdf'
            fig.savefig(filename)
            shutil.copyfile(filename, 'report/figures/' + filename)
        else:
            fig.suptitle(case['name'])
            plt.show()

        UdefCases.append(Udef)
    return UdefCases

# =============================================================================
# %% Single Wake cases - SWiFT case for various stability conditions with RANS and wake models
# =============================================================================


def plotSingleWake_SWiFT(StabilityCases, wffm_list, wm_labels, wm_colors, linewidth, cLES, save_plot=False):
    """
    Plot velocity deficit for three different stability cases

    Parameters
    ----------
    StabilityCases : dict
        dictionary containing all parameters for the SWiFT single wake case
    wffm_list : list
        list of user-specified wind farm models used for the benchmarking
    wm_labels : labels
        labels for the wind farm models
    wm_colors : str
        list of colors for the engineering wind farm models
    linewidth : int or float
        line width used for the plots
    cLES : str
        color specified for LES results
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    """

    sDown_D = np.array([2, 3, 4, 5])  # distances downstream
    turbmodels = ['keMOfP', 'keABLclfP']

    # =============================================================================
    # Setting up Wind Turbine parameters
    # =============================================================================

    windTurbines = V27SWIFT()
    h = windTurbines.hub_height()
    D = windTurbines.diameter()

    # =============================================================================
    #  Loading Lidar data for each case: stable, neutral and unstable and each downstream distance
    # =============================================================================

    for case in StabilityCases:
        datafiles = []
        modelfiles = []
        modelfilesdown1 = []
        modelfilesdown2 = []
        for j in range(sDown_D.size):
            label = get_downstream_label(sDown_D[j])
            if (case['case_name'] != 'unstable') or (case['case_name'] == 'unstable' and j == 1):
                lidarfile = SWiFTlidar_path + case['case_name'] + '_' + str(float(sDown_D[j])) + 'D_ffor_lateral_vd.csv'
                datafiles.append(lidarfile)
            else:
                datafiles.append('')
            filename = data_path + 'SWiFT_' + case['case_name'] + '_RANS_' + turbmodels[0] + '_hub_' + label + 'D.dat'
            modelfilesdown1.append(filename)
            filename = data_path + 'SWiFT_' + case['case_name'] + '_RANS_' + turbmodels[1] + '_hub_' + label + 'D.dat'
            modelfilesdown2.append(filename)
        modelfiles.append(modelfilesdown1)
        modelfiles.append(modelfilesdown2)
        modelcolors = ['g', 'c']
        modeldashes = [(5, 0)] * len(modelfiles)

        # =============================================================================
        # Plotting
        # =============================================================================

        fig, ax = plt.subplots(1, len(sDown_D), sharex=False, sharey=False,
                               figsize=(6 * len(sDown_D), 5), facecolor='white')
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        # =============================================================================
        # References, based on measurements
        # =============================================================================

        dataU0 = 1.0   # value of reference wind speed
        datawd0 = 0.0  # value of reference wind direction
        dataiU = 1     # index to get value of wind speed in lidar data file
        dataiUstd = 4  # index to get value of wind speed standard deviation in lidar data file
        dataNbins = 1  # number of bins
        for i in range(len(sDown_D)):
            if datafiles[i] != '':
                data = np.genfromtxt(datafiles[i], skip_header=True, delimiter=',')
                # Number of bins is a constant
                ldata = ax[i].errorbar(data[:, 0] - datawd0,
                                       data[:, dataiU] / dataU0,
                                       yerr=data[:, dataiUstd] / dataU0 / np.sqrt(dataNbins),
                                       color='k', elinewidth=1.0, linewidth=0,
                                       marker='o', zorder=0, markersize=4, label='Data')

        # =============================================================================
        # RANS,  based on EllipSys3D AD k-epsilon-fP
        # =============================================================================

            for j in range(len(modelfiles)):
                model = np.genfromtxt(modelfiles[j][i], skip_header=True)

                lm, = ax[i].plot(model[:, 0],
                                 model[:, 1],
                                 modelcolors[j], linewidth=linewidth, dashes=modeldashes[j], label='RANS')

        # =============================================================================
        # Engineering Wind Farm Models
        # =============================================================================

        # Getting k values for some models and keyword arguments
            kwargs_list = kwargs_func_SWiFt(case, wffm_list)

        # ===== Instantiating wind farm models from literature =====
            wakemodels = []
            for wfm, kwargs in zip(wffm_list, kwargs_list):
                wakemodels.append(wfm(**kwargs))

        # Getting velocity deficit at a horizontal plane at the wind turbine's hub height for each downstream distance
        # TODO: check if fuga works with flow_blox
            for k, wakemodel in enumerate(wakemodels):
                U_def = wakemodel(x=[0.0],
                                  y=[0.0],
                                  wd=[270.0],
                                  ws=case['U0'], TI=case['TI']/0.8).flow_box(x=sDown_D[i]*D,
                                                                             y=np.linspace(-1.5*D, 1.5*D,
                                                                                           len(model[:, 0])),
                                                                             h=h).WS_eff.squeeze()

                l1, = ax[i].plot(model[:, 0],
                                 (U_def - case['U0']) / case['U0'],
                                 wm_colors[k], linewidth=linewidth, label=wm_labels[k])
            ax[i].set_title(fr'$x={sDown_D[i]}D$')
            ax[i].set_xlim(-1.5, 1.5)
            ax[i].grid(True)
            ax[0].set_ylabel(r'$(U-U_0)/U_0$', rotation=90)
            fig.supxlabel(r'$y/D$')

            fig.tight_layout(rect=[0, 0, 1, 0.85])
            lines, labels = ax[1].get_legend_handles_labels()
    
            ncol = len(modelfiles) + len(wffm_list) + 1
    
            fig.legend(lines, labels, ncol=ncol, loc='upper center', bbox_to_anchor=(0, 0, 1, 0.93),
                       numpoints=1, scatterpoints=1)

        if save_plot is True:
            if os.path.isdir('report/figures'):
                shutil.rmtree('report/figures')
            os.mkdir('report/figures')
            outfile = 'SWiFT_' + case['case_name'] + '_hub' + '.' + 'pdf'
            fig.savefig(outfile)
            shutil.copyfile(outfile, 'report/figures/' + outfile)
        else:
            fig.suptitle(case['case_name'] + ' condition')
            plt.show()

# =============================================================================
# %% Single Wake cases - Integrated velocity deficit
# =============================================================================


def barPlotSingleWakeCases(SingleWakeCases, UdefCases, wm_labels, cLES, cRANS, colors, save_plot=False):
    """
    Bar plot of integrated velocity deficit for the different engineering models plus data, RANS and LES

    Parameters
    ----------
    SingleWakeCases : dict
        dictionary containing all parameters for the single wake cases
    UdefCases : list
        list of integrated velocity deficit for different distances downstream
    wm_labels : labels
        labels for the wind farm models
    cLES : str
        color specified for LES results
    cRANS : str
        color specified for RANS results
    colors : str
        list of colors for the engineering wind farm models
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    """
    fig, ax = plt.subplots(1, 1, figsize=(8*3, 4), facecolor='white')
    ibar = 1
    names = []
    subnames = []
    lines = []
    nDown = 3
    nWakeModels = len(UdefCases) - 3

    for i in range(len(UdefCases)):
        for j in range(nDown):
            ldata, = ax.bar(ibar, UdefCases[i][j, 0], width=0.5, color='k', edgecolor='k', label='Data')
            ibar = ibar + 1
            lLES, = ax.bar(ibar, UdefCases[i][j, 1], width=0.5, color=cLES, edgecolor=cLES, label='LES')
            ibar = ibar + 1
            lRANS, = ax.bar(ibar, UdefCases[i][j, 2], width=0.5, color=cRANS, edgecolor=cRANS, label='RANS')
            ibar = ibar + 1
            for k in range(nWakeModels):
                l1, = ax.bar(ibar, UdefCases[i][j, 3 + k], width=0.5, color=colors[k],
                             edgecolor=colors[k], label=wm_labels[k])
                ibar = ibar + 1
            ibar = ibar + 1
            if j < nDown - 1:
                ax.plot([ibar, ibar], [0, 0.35], ':k')
            else:
                ax.plot([ibar, ibar], [0, 0.35], '--k', dashes=[5, 2])
            ibar = ibar + 1
            subnames.append(str(SingleWakeCases[i]['xDown'][j]))
        names.append('Case ' + str(i + 1))

    ibar = ibar - 1
    ax.set_xticks(np.linspace(0.5 * ibar / len(UdefCases), ibar - 0.5 * ibar / len(UdefCases), len(UdefCases)))
    ax.set_xticklabels(names)
    ax.set_xlim(0, ibar)
    ax.tick_params(axis='x', direction='out')
    ax2 = ax.twiny()
    ax2.set_xticks(np.linspace(0.5 * ibar / (nDown * len(UdefCases)), ibar -
                               0.5 * ibar / (nDown * len(UdefCases)), nDown * len(UdefCases)))
    ax2.set_xticklabels(subnames)
    ax2.set_xlim(0, ibar)
    ax2.tick_params(axis='x', direction='in', pad=-15)
    ax.set_title('$x/D$')
    ax.set_ylabel('Integrated velocity deficit')
    lines, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    ncol = nWakeModels + 3
    fig.legend((lines[0:ncol]), (labels[0:ncol]), ncol=ncol, loc='upper center', bbox_to_anchor=(0, 0, 1.05, 1),
               numpoints=1, scatterpoints=1)

    if save_plot is True:
        if os.path.isdir('report/figures'):
            shutil.rmtree('report/figures')
        os.mkdir('report/figures')
        filename = 'IntegratedVelocityDeficit.pdf'
        fig.savefig(filename)
        shutil.copyfile(filename, 'report/figures/' + filename)
    else:
        plt.show()
