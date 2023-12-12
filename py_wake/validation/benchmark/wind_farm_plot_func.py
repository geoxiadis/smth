from copy import copy
import os
import shutil
import matplotlib
import matplotlib.style
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

# PyWake objects
from py_wake.validation.benchmark import validation_utils
from py_wake.validation.benchmark.validation_utils import sigma_hornsrev, GaussianFilter, kwargs_func_WF
from py_wake import np

# path to reference data
data_path = os.path.dirname(validation_utils.__file__) + '/data/'

# =============================================================================
# %% Wind farm cases - including data, LES, RANS, and different wake models
# =============================================================================


def deficitPlotWFCases(WFcases, wffm_list, wm_labels, linewidth, cLES, cRANS, colors, save_plot=False, latex_table=False):
    """
    Plot power deficit or wind farm efficiency for each wind farm case

    Parameters
    ----------
    WFcases : dict
        dictionary containing all parameters for the wind farm cases
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
    latex_table : bool
        user-defined option to create a LaTex table with the results of the Wind Farm efficiency
    """

    for case in WFcases:

        # =============================================================================
        # Engineering Wind Farm Models
        # =============================================================================

        # Setting up the Site, WindTurbine and other parameters needed for the engineering wind farm models
        kwargs_list = kwargs_func_WF(case, wffm_list)

        # ===== Instantiating wind farm models from literature =====
        wakemodels = []
        for wfm, kwargs in zip(wffm_list, kwargs_list):
            wakemodels.append(wfm(**kwargs))

        # =============================================================================
        # Plot power deficit in wind turbine row cases or calculate the WF efficiency
        # =============================================================================

        # Calculating standard deviation for each case
        if case['name'] == 'Wieringermeer':
            sigma = np.zeros((len(case['wt_x'])))
            sigma[:] = 2.5
        elif case['name'] == 'Lillgrund':
            sigma = np.zeros((len(case['wt_x'])))
            sigma[:] = 3.3
        elif case['name'] == 'Hornsrev1':
            sigma = sigma_hornsrev('vanderLaan', case['wt_x'], case['wt_y'])

        power_models = []
        powerGA_models = []
        for wakemodel in wakemodels:
            # Calculating power at each wind turbine, wind speed and wind direction
            power_ilk = wakemodel(case['wt_x'], case['wt_y'], ws=case['U0'], yaw=[0]).power_ilk
            power_models.append(power_ilk)

            # ===== Gaussian averaging =====
            powerGA = np.zeros(power_ilk.shape)
            for iAD in range(len(case['wt_x'])):
                powerGA[iAD, :, 0] = GaussianFilter(power_ilk[iAD, :, 0],
                                                    np.arange(0, 360.0, 1),
                                                    int(np.ceil(3 * sigma[iAD])), sigma[iAD])
            powerGA_models.append(powerGA)

        # ===== Make a figure per wind direction and WT row or wind farm efficiency =====
        for plot in case['plots']:

            #  ===== Wind farm efficiency =====

            # The wind farm efficiency is normalized with the power for the reference wind speed of each case and it is calculated with and without the gaussian-averaged power
            if plot['name'] == 'WFeff':
                data = np.genfromtxt(data_path + case['name'] + '_WFdata_' + plot['name'] + '.dat', skip_header=True)
                RANS = np.genfromtxt(data_path + case['name'] + '_RANS_' + plot['name'] + '.dat', skip_header=True)

                norm = len(case['wt_x']) * case['wt'].power(case['U0'])
                WFeffWakeModels = []

                for k in range(len(wakemodels)):
                    WFeff = power_models[k][:, :, 0].sum(axis=0) / norm
                    WFeffGA = powerGA_models[k][:, :, 0].sum(axis=0) / norm

                    # Getting wind farm efficiency for each engineering model
                    print('WFeff', wm_labels[k], WFeff.mean())
                    WFeffWakeModels.append(WFeff.mean())

                # Getting wind farm efficiency from RANS data and measurements
                print('WFeff RANS', RANS[:, 1].mean())
                print('WFeff DATA', data[:, 1].mean())

            #  ===== Creating a LaTex table =====
                if latex_table is True:
                    f = open('report/' + case['name'] + '_' + plot['name'] + '.tex', 'w')
                    f.write('\\begin{tabular}{lcccc}\n')
                    f.write('\hline\n')
                    # TODO: make sure the headers of the table correspond to the models in the wffm_list
                    f.write('		     & Measurement data & RANS & NOJ  &   GAU  \\\\ \n')
                    f.write('\hline\n')
                    # Estimate uncertainty in measured wind farm efficiency assuming that we
                    # have the same number of samples per wd
                    dataWFunc = np.sqrt(sum(data[:, 2] ** 2) / len(data[:, 1]))
                    f.write('%s %4.2f %s %4.3f %s %4.2f' % ('Wind farm efficiency & $',
                                                            data[:, 1].mean(), '\\pm', dataWFunc,
                                                            '$ & ', RANS[:, 1].mean()))
                    for k in range(len(wakemodels)):
                        f.write('%s %4.2f' % (' & ', WFeffWakeModels[k]))
                    f.write('\\\\ \n')
                    f.write('%s %4.0f' % ('Relative error [\\%] & - & ', 100.0 *
                                          (RANS[:, 1].mean() - data[:, 1].mean()) / data[:, 1].mean()))
                    for k in range(len(wakemodels)):
                        f.write('%s %4.0f' % (' & ',
                                              100.0 * (WFeffWakeModels[k] - data[:, 1].mean()) / data[:, 1].mean()))
                    f.write('\\\\ \n')
                    f.write('\hline\n')
                    f.write('\end{tabular}\n')
                    f.close()

            # ===== Power deficit per turbine row =====
            else:
                wd = plot['wd']
                data = np.genfromtxt(data_path + case['name'] + '_WFdata_wd' +
                                     str(int(wd)) + '_' + plot['name'] + '.dat', skip_header=True)
                RANS = np.genfromtxt(data_path + case['name'] + '_RANS_wd' +
                                     str(int(wd)) + '_' + plot['name'] + '.dat', skip_header=True)

            # =============================================================================
            # Plotting
            # =============================================================================

                fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(15, 12), facecolor='white')
                ax_layout = fig.add_axes([0.7, 0.3, 0.25, 0.25], frameon=True)

            # =============================================================================
            # Plotting data, based on measurements
            # =============================================================================

                ldata1 = ax[0].errorbar(data[:, 0], data[:, 1] / data[0, 1], yerr=data[:, 2] / np.sqrt(data[:, 3]),
                                        color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0,
                                        markersize=4, label='Data')
                ldata2 = ax[1].errorbar(data[:, 0], data[:, 1] / data[0, 1], yerr=data[:, 2] / np.sqrt(data[:, 3]),
                                        color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0,
                                        markersize=4, label='Data')

            # =============================================================================
            # Plotting RANS data
            # =============================================================================

                lRANS1, = ax[0].plot(RANS[:, 0], RANS[:, 1], color=cRANS, linewidth=linewidth, label='RANS')
                lRANS2, = ax[1].plot(RANS[:, 0], RANS[:, 2], color=cRANS, linewidth=linewidth, label='RANS GA')

            # =============================================================================
            # Plotting Engineering Models
            # =============================================================================

                for k in range(len(wakemodels)):
                    if case['name'] == 'Hornsrev1':
                        # Linear average for 267-273 deg and reshape in WT rows and columns
                        power_matrix = power_models[k][:, 267:274, 0].mean(axis=1).reshape(10, 8)
                        powerGA_matrix = powerGA_models[k][:, 267:274, 0].mean(axis=1).reshape(10, 8)

                        # Sum the innner rows
                        powersum_inner_rows = power_matrix[:, 1:6].sum(axis=1)
                        powerGAsum_inner_rows = powerGA_matrix[:, 1:6].sum(axis=1)

                        l1, = ax[0].plot(np.linspace(1, 10, 10),
                                         powersum_inner_rows / powersum_inner_rows[0], color=colors[k],
                                         linewidth=linewidth, label=wm_labels[k])
                        l2, = ax[1].plot(np.linspace(1, 10, 10),
                                         powerGAsum_inner_rows / powerGAsum_inner_rows[0], color=colors[k],
                                         linewidth=linewidth, label=wm_labels[k] + ' GA')
                    else:
                        # Collect data for a row and linear average within wd bin.
                        power_row = np.zeros((len(plot['wts']), 2))
                        for i in range(len(plot['wts'])):
                            if np.isnan(plot['wts'][i]):
                                power_row[i, :] = np.nan
                            else:
                                power_row[i, 0] = power_models[k][plot['wts'][i], int(wd) - 3:int(wd) + 4, 0].mean()
                                power_row[i, 1] = powerGA_models[k][plot['wts'][i], int(wd) - 3:int(wd) + 4, 0].mean()
                        l1, = ax[0].plot(np.linspace(1, len(plot['wts']), len(
                            plot['wts'])), power_row[:, 0] / power_row[0, 0], color=colors[k], linewidth=linewidth, label=wm_labels[k])
                        l2, = ax[1].plot(np.linspace(1, len(plot['wts']), len(
                            plot['wts'])), power_row[:, 1] / power_row[0, 1], color=colors[k], linewidth=linewidth, label=wm_labels[k] + ' GA')

                ax[0].grid(True)
                ax[1].grid(True)
                ax[0].set_ylabel('$P_i/P_1$', rotation=0)
                ax[1].set_ylabel('$P_i/P_1$', rotation=0)
                ax[0].yaxis.labelpad = 20
                ax[1].yaxis.labelpad = 20
                ax[1].set_xlabel('WT nr.')
                ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
                fig.tight_layout(rect=[0, 0, 0.75, 0.9], pad=7)

                # Plot the layout in a small plot
                xWT = (np.array(case['wt_x']) - case['wt_x'][0]) / case['wt'].diameter()
                yWT = (np.array(case['wt_y']) - case['wt_y'][0]) / case['wt'].diameter()
                ax_layout.scatter(xWT, yWT, s=2, color='k')
                for i in range(len(plot['wts'])):
                    if not np.isnan(plot['wts'][i]):
                        ax_layout.scatter(xWT[int(plot['wts'][i])], yWT[int(plot['wts'][i])], s=2, color='r')
                ax_layout.set_xticks([])
                ax_layout.set_yticks([])
                ax_layout.set_aspect(1)
                ax_layout.set_xlim(xWT.min() - 20, xWT.max() + 10)
                ax_layout.set_ylim(yWT.min() - 10, yWT.max() + 20)
                ax_layout.arrow(xWT.min() - 10, yWT.max() + 10, 6 * np.cos((270.0 - wd) / 180.0 * np.pi),
                                6 * np.sin((270.0 - wd) / 180.0 * np.pi), head_width=2, head_length=1, color='r')
                circle1 = plt.Circle((xWT.min() - 10, yWT.max() + 10), 8, color='k', fill=False)
                ax_layout.add_artist(circle1)

                # Move layout plot to the top of the main plot
                pos1 = ax[0].get_position()
                pos2 = ax_layout.get_position()
                pos3 = [pos2.x0, pos2.y0 + pos1.y0 + pos1.height - (pos2.y0 + pos2.height),
                        (pos2.width - pos2.width/2), pos2.height]
                ax_layout.set_position(pos3)

                ncol = len(wakemodels) + 2

                lines, labels = ax[0].get_legend_handles_labels()
                linesGA, labelsGA = ax[1].get_legend_handles_labels()

                ax[0].legend(lines, labels, ncol=ncol, loc='upper left',
                             bbox_to_anchor=(-0.1, 0.84, 0.5, 0.4), numpoints=1, scatterpoints=1)
                ax[1].legend(linesGA, labelsGA, ncol=ncol, loc='upper left',
                             bbox_to_anchor=(-0.1, 0.84, 0.5, 0.4), numpoints=1, scatterpoints=1)

                if save_plot is True:
                    if os.path.isdir('report/figures'):
                        shutil.rmtree('report/figures')
                    os.mkdir('report/figures')
                    filename = case['name'] + '_wd' + str(int(wd)) + '_' + plot['name'] + '.pdf'
                    fig.savefig(filename)
                    shutil.copyfile(filename, 'report/figures/' + filename)
                else:
                    ax[0].set_title(case['name'] + '_wd' + str(int(wd)) + '_' + plot['name'])
                    ax[1].set_title(case['name'] + '_wd' + str(int(wd)) + '_' + plot['name'])
                    plt.show()
