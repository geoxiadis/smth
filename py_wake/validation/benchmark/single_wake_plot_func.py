import os
import shutil
import warnings
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# PyWake objects
from py_wake.validation.benchmark import validation_utils
from py_wake.validation.benchmark.validation_utils import wffm_kwargs, integrate_velocity_deficit_arc, \
    get_downstream_label
from py_wake.flow_map import HorizontalGrid
from py_wake.examples.data.v27swift import V27SWIFT
from py_wake import np

# path to reference data
data_path = os.path.dirname(validation_utils.__file__) + '/data/'
SWiFTlidar_path = os.path.dirname(validation_utils.__file__) + '/data/SWiFT_lidar_data/'


# =============================================================================
# %% Single Wake cases - including data, LES, RANS and different wake models
# =============================================================================


def deficitPlotSingleWakeCases(SingleWakeCases, wffms, cLES, cRANS, save_plot=False):
    """
    Plot velocity deficit for each single wake case

    Parameters
    ----------
    SingleWakeCases : dict
        dictionary containing all parameters for the single wake cases
    wffms : dict
        dict of dict, mapping engineering model object and its plotting options (color, shortname etc.)
    cLES : str
        color specified for LES results
    cRANS : str
        color specified for RANS results
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    """

    UdefCases = []
    linewidth = 1.5

    for case in SingleWakeCases:

        # Wind directions used in the study (corresponding to the wind directions available in the data)
        wds = np.linspace(-30, 30, 61)

        # Turbine diameter, if not explicitly defined in the case param
        if 'D' not in case:
            case['D'] = case['wt'].diameter()

        # =============================================================================
        # Engineering Wind Farm Models
        # =============================================================================

        wake_ws = np.zeros((len(wffms), len(wds), len(case['xDown'])))
        for k, wakemodel in enumerate(wffms.keys()):
            # The velocity deficit at an arc is calculated by running the iwake_map for each point.
            # because it is only possible to provide a x and y array that define a rectangular plane.
            # This should be improved, where one can use a list of points and run wake_map once.
            for j in range(len(case['xDown'])):
                # Setting up the Site, WindTurbine and other parameters needed for the engineering wind farm models
                wffm = wffm_kwargs(case, wakemodel)
                sim_res = wffm(x=[0.0],
                               y=[0.0],
                               wd=[270.0],
                               ws=case['U0'])
                for i, wd in enumerate(wds):
                    xdown_meter = case['xDown'][j] * case['D']
                    # positions of the turbines
                    x = xdown_meter * np.cos(wds[i] / 180.0 * np.pi)
                    y = xdown_meter * np.sin(wds[i] / 180.0 * np.pi)
                    WS_eff = sim_res.flow_map(HorizontalGrid(x=x, y=y)).WS_eff.squeeze()
                    # Effective wind speed for each wakemodel, wind direction and downstream distance
                    wake_ws[k, i, j] = WS_eff

        # =============================================================================
        # Plotting
        # =============================================================================

        fig, ax = plt.subplots(1, len(case['xDown']), sharey=False, figsize=(8 * len(case['xDown']), 5), facecolor='white')

        # Instantiating variable for storing integrated momentum deficit
        Udef = np.zeros((len(case['xDown']), len(wffms) + 3))

        # =============================================================================
        # Loading and plotting RANS, LES, measurement data
        # =============================================================================
        # For both Wieringermeer cases, the middle wind direction is substracted and the data is normalized with
        # the reference wind speed U0. Data is represented as error bars with the standard deviation provided from
        # a 10min period.
        # The integrated momentum deficit is calculated using the "integrated_velocity_deficit_arc" function.
        for j, xdown_meter in enumerate(case['xDown']):
            if case['name'] == 'Wieringermeer-West' and j == 1:
                data = np.genfromtxt(data_path + case['name'] + '_data_' + get_downstream_label(case['xDown'][j]) + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0] - 315, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                                                                                       np.sqrt(data[:, 3]), color='k',
                                       elinewidth=1.0, linewidth=0,
                                       marker='o', zorder=0, markersize=4, label='Data')
                fdata = interp1d(data[:, 0] - 315, data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])

            elif case['name'] == 'Wieringermeer-East' and j == 0:
                data = np.genfromtxt(data_path + case['name'] + '_data_' + get_downstream_label(case['xDown'][j]) + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0] - 31, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                                                                                      np.sqrt(data[:, 3]), color='k',
                                       elinewidth=1.0, linewidth=0,
                                       marker='o', zorder=0, markersize=4, label='Data')
                fdata = interp1d(data[:, 0] - 31, data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])

            elif case['name'] == 'Nibe':
                # No standard deviation of the 10 min. available.
                data = np.genfromtxt(data_path + case['name'] + '_data_' + get_downstream_label(case['xDown'][j]) + 'D.dat')
                ldata = ax[j].scatter(data[:, 0], data[:, 1], color='k', marker='o', zorder=0, s=10, label='Data')
                fdata = interp1d(data[:, 0], data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds) * case['U0'], case['xDown'][j], case['U0'])

            elif case['name'] == 'Nordtank-500' and j < 2:
                data = np.genfromtxt(data_path + case['name'] + '_data_' + get_downstream_label(case['xDown'][j]) + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0], data[:, 2], yerr=data[:, 3] / np.sqrt(74.0),
                                       color='k', elinewidth=1.0, linewidth=0, marker='o',
                                       zorder=0, markersize=4, label='Data')

            # =============================================================================
            # Loading and plotting LES data, based on Ellipsys3D AD
            # =============================================================================
            try:
                LES = np.genfromtxt(data_path + case['name'] + '_LES_' + get_downstream_label(case['xDown'][j]) + 'D.dat')
                # Shaded area represents the standard error of the mean
                lLES = ax[j].fill_between(LES[:, 0], LES[:, 1] - LES[:, 2] / np.sqrt(LES[:, 3]),
                                          LES[:, 1] + LES[:, 2] / np.sqrt(LES[:, 3]), color=cLES, alpha=0.5,
                                          label='LES')
            except FileNotFoundError as e:
                LES = np.zeros((2, len(wds)))
                warnings.warn(f'{e}')
            # =============================================================================
            # Loading and plotting RANS data,  based on EllipSys3D AD k-epsilon-fP
            # =============================================================================
            try:
                RANS = np.genfromtxt(data_path + case['name'] + '_RANS_' + get_downstream_label(case['xDown'][j]) + 'D.dat')
                lRANS, = ax[j].plot(RANS[:, 0], RANS[:, 1], color=cRANS, linewidth=linewidth, label='RANS')
            except FileNotFoundError as e:
                RANS = np.zeros((2, len(wds)))
                warnings.warn(f'{e}')
            # =============================================================================
            # Plotting engineering models
            # =============================================================================

            for k, wffm_param in enumerate(wffms.values()):
                vel_deficit = wake_ws[k, :, j] / case['U0']
                l1, = ax[j].plot(wds, vel_deficit, color=wffm_param['color'], linewidth=linewidth, label=wffm_param['label'])
                title = '%s %g %s' % ('$x=', case['xDown'][j], 'D$')
                ax[j].set_title(title)
                ax[j].set_xticks(np.arange(min(wds), max(wds) + 10.0, 10.0))
            if case['xDown'][j] < 7.0:
                ax[j].set_xlim(-30, 30)
            else:
                ax[j].set_xlim(-20, 20)
            ax[j].grid(True)
            ax[j].set_ylim((None, 1.1))

            # ===== Momentum deficit for each wake model =====
            for k in range(len(wffms)):
                Udef[j, 3 + k] = integrate_velocity_deficit_arc(wds, wake_ws[k, :, j], case['xDown'][j], case['U0'])
            Udef[j, 1] = integrate_velocity_deficit_arc(LES[:, 0], LES[:, 1] * case['U0'], case['xDown'][j], case['U0'])
            Udef[j, 2] = integrate_velocity_deficit_arc(RANS[:, 0], RANS[:, 1] * case['U0'], case['xDown'][j],
                                                        case['U0'])

        ax[0].set_ylabel('$U/U_0$', rotation=0)
        ax[0].yaxis.labelpad = 20
        ax[1].set_xlabel('Relative wind direction [deg]')
        fig.tight_layout(rect=[0, 0, 1, 0.85])

        legend_ncol = len(wffms) + 3

        if case['name'] == 'Wieringermeer-West':
            lines, labels = ax[1].get_legend_handles_labels()
        else:
            lines, labels = ax[0].get_legend_handles_labels()

        fig.legend(lines, labels, ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0, 0, 1.05, 0.93), numpoints=1,
                   scatterpoints=1)

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


def plotSingleWake_SWiFT(StabilityCases, wffms, save_plot=False):
    """
    Plot velocity deficit for three different stability cases

    Parameters
    ----------
    StabilityCases : dict
        dictionary containing all parameters for the SWiFT single wake case
    wffms : dict
        dict of dict, mapping engineering model object and its plotting options (color, shortname etc.)
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    """

    linewidth = 1.5
    turbmodels = ['keMOfP', 'keABLclfP']
    # =============================================================================
    # Setting up Wind Turbine parameters (V27 by default for SWiFT study)
    # =============================================================================
    for case in StabilityCases:
        if 'wt' not in case:
            case['wt'] = V27SWIFT()
        h = case['wt'].hub_height()
        D = case['wt'].diameter()

        # =============================================================================
        # Plotting
        # =============================================================================

        fig, ax = plt.subplots(1, len(case['xDown']), sharex=False, sharey=False,
                               figsize=(6 * len(case['xDown']), 5), facecolor='white')
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        # =============================================================================
        # References, based on measurements
        # =============================================================================

        dataU0 = 1.0  # value of reference wind speed
        datawd0 = 0.0  # value of reference wind direction
        dataiU = 1  # index to get value of wind speed in lidar data file
        dataiUstd = 4  # index to get value of wind speed standard deviation in lidar data file
        dataNbins = 1  # number of bins
        ranscolors = ['g', 'c']
        for i, down_d in enumerate(case['xDown']):
            rans_mo_path = f'{data_path}SWiFT_{case["name"]}_RANS_{turbmodels[0]}_hub_{get_downstream_label(down_d)}D.dat'
            rans_abl_path = f'{data_path}SWiFT_{case["name"]}_RANS_{turbmodels[1]}_hub_{get_downstream_label(down_d)}D.dat'
            lidar_path = f'{SWiFTlidar_path}{case["name"]}_{str(float(down_d))}D_ffor_lateral_vd.csv'
            # =============================================================================
            #  Loading Lidar data for each case: stable, neutral and unstable and each downstream distance
            # =============================================================================
            try:
                data = np.genfromtxt(lidar_path, skip_header=True, delimiter=',')
                # Number of bins is a constant
                ldata = ax[i].errorbar(data[:, 0] - datawd0,
                                       data[:, dataiU] / dataU0,
                                       yerr=data[:, dataiUstd] / dataU0 / np.sqrt(dataNbins),
                                       color='k', elinewidth=1.0, linewidth=0,
                                       marker='o', zorder=0, markersize=4, label='Data')
            except FileNotFoundError as e:
                warnings.warn(f'{e}')
            # =============================================================================
            # RANS,  based on EllipSys3D AD k-epsilon-fP with 'keMOfP' and 'keABLclfP'
            # =============================================================================
            try:
                rans_mo = np.genfromtxt(rans_mo_path, skip_header=True)
                lm, = ax[i].plot(rans_mo[:, 0],
                                 rans_mo[:, 1],
                                 ranscolors[0], linewidth=linewidth, dashes=(5, 0), label='RANS-MOST')
                rans_abl = np.genfromtxt(rans_abl_path, skip_header=True)
                lm, = ax[i].plot(rans_abl[:, 0],
                                 rans_abl[:, 1],
                                 ranscolors[1], linewidth=linewidth, dashes=(5, 0), label='RANS-ABL')
            except FileNotFoundError as e:
                warnings.warn(f'{e}')

            # =============================================================================
            # Engineering Wind Farm Models
            # =============================================================================

            # Getting velocity deficit at a horizontal plane at the wind turbine's hub height for each downstream distance
            for k, (wakemodel, wffm_param) in enumerate(wffms.items()):
                # Setting up the Site, WindTurbine and other parameters needed for the engineering wind farm models
                wffm = wffm_kwargs(case, wakemodel)
                yaxis = np.linspace(-1.5 * D, 1.5 * D, 61)  # len(data[:, 0])
                U_def = wffm(x=[0.0],
                             y=[0.0],
                             wd=[270.0],
                             ws=case['U0'], TI=case['TItot'] / 0.8).flow_box(x=down_d * D,
                                                                             y=yaxis,
                                                                             h=h).WS_eff.squeeze()
                l1, = ax[i].plot(yaxis / D,
                                 (U_def - case['U0']) / case['U0'],
                                 wffm_param['color'], linewidth=linewidth, label=wffm_param['label'])
            ax[i].set_title(fr'$x={down_d}D$')
            ax[i].set_xlim(-1.5, 1.5)
            ax[i].set_ylim((None, 0.1))
            ax[i].grid(True)
            ax[0].set_ylabel(r'$(U-U_0)/U_0$', rotation=90)
            fig.supxlabel(r'$y/D$')
            fig.tight_layout(rect=[0, 0, 1, 0.85])
            lines, labels = ax[1].get_legend_handles_labels()
            legend_ncol = 2 + len(wffms) + 1  # rans, n_wffms, lidar_data
            fig.legend(lines, labels, ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0, 0, 1, 0.93),
                       numpoints=1, scatterpoints=1)
        if save_plot is True:
            if os.path.isdir('report/figures'):
                shutil.rmtree('report/figures')
            os.mkdir('report/figures')
            outfile = 'SWiFT_' + case['name'] + '_hub' + '.' + 'pdf'
            fig.savefig(outfile)
            shutil.copyfile(outfile, 'report/figures/' + outfile)
        else:
            fig.suptitle(case['name'] + ' condition')
            plt.show()


# =============================================================================
# %% Single Wake cases - Integrated velocity deficit
# =============================================================================


def barPlotSingleWakeCases(SingleWakeCases, UdefCases, cLES, cRANS, wffms, save_plot=False):
    """
    Bar plot of integrated velocity deficit for the different engineering models plus data, RANS and LES

    Parameters
    ----------
    SingleWakeCases : list of dict
        dictionary containing all parameters for the single wake cases
    UdefCases : list of dict
        list of integrated velocity deficit for different distances downstream
    cLES : str
        color specified for LES results
    cRANS : str
        color specified for RANS results
    wffms : dict
        dict of dict, mapping engineering model object and its plotting options (color, shortname etc.)
    save_plot : bool
        user-defined option to determine whether plots are saved as PDF or shown inline
    """
    fig, ax = plt.subplots(1, 1, figsize=(8 * 3, 4), facecolor='white')
    ibar = 1
    names = []
    subnames = []
    lines = []
    nDown = 3
    nWakeModels = len(wffms)

    for i in range(len(UdefCases)):
        for j in range(nDown):
            ldata, = ax.bar(ibar, UdefCases[i][j, 0], width=0.5, color='k', edgecolor='k', label='Data')
            ibar = ibar + 1
            lLES, = ax.bar(ibar, UdefCases[i][j, 1], width=0.5, color=cLES, edgecolor=cLES, label='LES')
            ibar = ibar + 1
            lRANS, = ax.bar(ibar, UdefCases[i][j, 2], width=0.5, color=cRANS, edgecolor=cRANS, label='RANS')
            ibar = ibar + 1
            for k, (wake_model, wffm_param) in enumerate(wffms.items()):
                l1, = ax.bar(ibar, UdefCases[i][j, 3 + k], width=0.5, color=wffm_param['color'],
                             edgecolor=wffm_param['color'], label=wffm_param['label'])
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
    legend_ncol = nWakeModels + 3
    fig.legend((lines[0:legend_ncol]), (labels[0:legend_ncol]), ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0, 0, 1.05, 1),
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
