from py_wake import np
import pandas as pd
import matplotlib.pyplot as plt
from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_1WT_Surrogate, IEA34_130_2WT_Surrogate
from py_wake.tests import npt
from py_wake.deficit_models.noj import NOJ
from py_wake.site.xrsite import UniformSite
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from pathlib import Path
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
from py_wake.examples.data import example_data_path
from py_wake.utils.gradients import plot_gradients, fd, autograd, cs
import pytest
from numpy import newaxis as na
import matplotlib.pyplot as plt
import json


@pytest.fixture(scope='module')
def iea34_130_1WT_Surrogate():
    return IEA34_130_1WT_Surrogate()


@pytest.fixture(scope='module')
def iea34_130_2WT_Surrogate():
    return IEA34_130_2WT_Surrogate()


def test_one_turbine_case0(iea34_130_1WT_Surrogate):

    if 0:
        f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
        with open(f + 'input_one_turbine.json') as fid:
            data = json.load(fid)
        print(", ".join(data.keys()) + " = " + ", ".join(str(v['0']) for v in data.values()))
        print(pd.concat([pd.read_csv(f + 'stats_one_turbine_mean.csv').iloc[0, [10, 25, 333]],
                         pd.read_csv(f + 'stats_one_turbine_std.csv').iloc[0, [10, 25, 333]]],
                        axis=1))
        # Free wind speed Vy, gl. coo, of gl. pos    0.00...  9.314707e+00       0.538521
        # Aero rotor thrust                                   5.344716e+02      17.355761
        # generator_servo inpvec   2  2: pelec [w]            2.940879e+06  231067.681060

        print(pd.read_csv(f + 'stats_one_turbine_del.csv').iloc[0, [28, 29, 1, 2, 9]])
        # MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1      2068.545057
        # MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1      5795.215631
        # MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment             4182.541410
        # MomentMy Mbdy:tower nodenr:   1 coo: tower  tower bottom moment             1649.970986
        # MomentMz Mbdy:tower nodenr:  11 coo: tower  tower top/yaw bearing moment    1332.371480

    ws, ti, shear = 9.2984459862, 0.0597870198, 0.2
    ws_ref = 9.314707e+00
    ws_std_ref = 0.538521
    power_ref = 2.940879e+06
    thrust_ref = 5.344716e+02
    ref_dels = [2068, 5795, 4182, 1649, 1332]

    wt = iea34_130_1WT_Surrogate
    assert wt.loadFunction.output_keys[1] == 'del_blade_edge'
    assert wt.loadFunction.wohler_exponents == [10, 10, 4, 4, 7]
    site = UniformSite(p_wd=[1], ti=ti, ws=ws)
    sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())([0], [0], wd=0, Alpha=shear)

    npt.assert_allclose(ws, ws_ref, rtol=.017)
    npt.assert_allclose(ti, ws_std_ref / ws_ref, atol=.002)
    npt.assert_allclose(sim_res.Power, power_ref, rtol=0.007)
    npt.assert_allclose(sim_res.CT, thrust_ref * 1e3 / (1 / 2 * 1.225 * (65**2 * np.pi) * ws_ref**2), rtol=0.011)

    loads = sim_res.loads()
    npt.assert_allclose(loads.DEL.squeeze(), ref_dels, rtol=.10)
    f = 20 * 365 * 24 * 3600 / 1e7
    m = np.array([10, 10, 4, 4, 7])
    npt.assert_array_almost_equal(loads.LDEL.squeeze(), (loads.DEL.squeeze()**m * f)**(1 / m))

    loads = sim_res.loads(average_ti=True)
    npt.assert_allclose(loads.DEL.squeeze(), ref_dels, rtol=.10)
    npt.assert_array_almost_equal(loads.LDEL.squeeze(), (loads.DEL.squeeze()**m * f)**(1 / m))


# def test_one_turbine_all_case_x(case=353):
#
#     f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
#     inp = pd.DataFrame(eval(Path(f + 'input_one_turbine.json').read_text())).iloc[case]
#
#     outp = pd.read_csv(f + 'stats_one_turbine_del.csv').iloc[case, [28, 29, 1, 2, 9]]
#     outp_all = pd.read_csv(f + 'stats_one_turbine_del_all_seeds.csv').iloc[:, [0, 28, 29, 1, 2, 9]]
#     points = np.array([int(l.split("_")[3]) for l in outp_all.iloc[:, 0]])
#     npt.assert_array_almost_equal(outp_all.iloc[np.where(points == case)[0]].mean(0), outp)
#     wt = IEA34_130_1WT_Surrogate()
#
#     ws, ti, shear = inp.values[:, na]
#     loads = np.array(wt.loadFunction(ws=ws, TI_eff=ti, Alpha=shear))
#     print("case", case)
#     print(inp)
#     err = (outp.values - np.array(loads).T) / outp.values * 100
#     print(np.concatenate([outp.values[:, na], loads, err.T], 1))
#
#
# def test_one_turbine_all_cases():
#
#     f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
#     inp = pd.DataFrame(eval(Path(f + 'input_one_turbine.json').read_text()))
#
#     outp = pd.read_csv(f + 'stats_one_turbine_del.csv').iloc[:, [28, 29, 1, 2, 9]]
#     outp_all = pd.read_csv(f + 'stats_one_turbine_del_all_seeds.csv').iloc[:, [0, 28, 29, 1, 2, 9]]
#     points = np.array([int(l.split("_")[3]) for l in outp_all.iloc[:, 0]])
#     wt = IEA34_130_1WT_Surrogate()
#
#     ws, ti, shear = inp.values.T
#     loads = np.array(wt.loadFunction(ws=ws, TI_eff=ti, Alpha=shear))
#
#     out_range = (outp.max(0) - outp.min(0)).values
#     err = np.abs((outp.values - np.array(loads).T) / out_range * 100).max(1)
#     err[(ws < 4) | (ws > 25)] = 0
#     worst = np.argsort(err)[::-1][:5]
#
#     for s, o, l in zip(outp, outp.values.T, loads):
#         plt.figure()
#         plt.title(s)
#         plt.plot(o, l, '.')
#         plt.plot([o.min(), o.max()], [o.min(), o.max()], 'k')
#         plt.plot(o[worst], l[worst], '.')
#     plt.show()
#
#     for i in worst:
#         test_one_turbine_all_case_x(i)


def test_one_turbine_10_01():
    ws, ti, shear = 10, .1, 0
    # MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1, EQ(m=10)    2733.4427073
    # MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1, EQ(m=10)    5713.19074073
    # MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment, EQ(m= 4)    7662.09840811
    # MomentMy Mbdy:tower nodenr:   1 coo: tower  tower bottom moment, EQ(m= 4)    3938.6210213
    # MomentMz Mbdy:tower nodenr:  11 coo: tower  tower top/yaw bearing moment    2197.0665
#     f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
#     inp = pd.DataFrame(eval(Path(f + 'input_one_turbine.json').read_text()))
#     closest_points = np.argsort((np.abs(inp.ws - 10)) + (np.abs(inp.ti - .1) * 10) + (np.abs(inp.shear)))[:10]
#     # print(inp.iloc[closest_points])
#     outp = pd.read_csv(f + 'stats_one_turbine_del.csv').iloc[:, [28, 29, 1, 2, 9]]
#     outp_all = pd.read_csv(f + 'stats_one_turbine_del_all_seeds.csv').iloc[:, [0, 28, 29, 1, 2, 9]]
#     points = np.array([int(l.split("_")[3]) for l in outp_all.iloc[:, 0]])
#     ws, ti, shear = inp.iloc[995:996].values.T
    wt = IEA34_130_1WT_Surrogate()
    ref_loads = [2733, 5713, 7665, 3938, 2197]
    loads = wt.loadFunction(ws=np.array([ws]), TI_eff=np.array([ti]), Alpha=np.array([shear]))
    err = (np.array(ref_loads) - np.array(loads)[:, 0]) / ref_loads * 100
    npt.assert_array_less(np.abs(err), [7, 1, 5, 17, 4])


def test_two_turbine_case0():
    if 0:
        i = 0
        f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
        print(list(pd.DataFrame(eval(Path(f + 'input_two_turbines_dist.json').read_text())).iloc[i]))
        # [10.9785338191, 0.2623204277, 8.64134e-05, 12.0126250539, 6.0429745131]

        print(pd.concat([pd.read_csv(f + 'stats_two_turbines_mean.csv').iloc[i, [15, 25, 333]],
                         pd.read_csv(f + 'stats_two_turbines_std.csv').iloc[i, [15, 25, 333]]],
                        axis=1))
        # Free wind speed Abs_vhor, gl. coo, of gl. pos  ...  1.088625e+01       2.653915
        # Aero rotor thrust                                   4.052527e+02      86.176889
        # generator_servo inpvec   2  2: pelec [w]            3.205083e+06  385531.551539

        print(pd.read_csv(f + 'stats_two_turbines_del.csv').iloc[i, [28, 29, 1, 2, 9]])
        # MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1       6981.199896
        # MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1       6190.827833
        # MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment             22754.653416
        # MomentMy Mbdy:tower nodenr:   1 coo: tower  tower bottom moment             10497.615411
        # MomentMz Mbdy:tower nodenr:  11 coo: tower  tower top/yaw bearing moment     6269.702940

    ws, ti, shear, wdir, dist = [10.9785338191, 0.2623204277, 8.64134e-05, 12.0126250539, 6.0429745131]

    # ref from simulation statistic (not updated yet)
    ws_ref = 1.088625e+01
    ws_std_ref = 2.653915
    thrust_ref = 4.052527e+02
    power_ref = 3.205083e+06
    ref_dels = [6981, 6190, 22754, 10497, 6269]

    wt = IEA34_130_2WT_Surrogate(inflow_input='uw')
    site = UniformSite(p_wd=[1], ti=ti, ws=ws)
    sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())([0, 0], [0, dist * 130], wd=wdir, Alpha=shear)

    npt.assert_allclose(ws, ws_ref, rtol=.012)
    npt.assert_allclose(ti, ws_std_ref / ws_ref, atol=.19)
    npt.assert_allclose(sim_res.Power.sel(wt=0), power_ref, rtol=0.014)
    npt.assert_allclose(sim_res.CT.sel(wt=0), thrust_ref * 1e3 / (1 / 2 * 1.225 * (65**2 * np.pi) * ws_ref**2),
                        rtol=0.03)

    loads = sim_res.loads()
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), ref_dels, rtol=.06)

    f = 20 * 365 * 24 * 3600 / 1e7
    m = loads.m.values
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))

    loads = sim_res.loads()
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), ref_dels, rtol=.05)
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))


def test_two_turbine_unwaked_10_01():
    ws, ti, shear = 10, .1, 0

    for inflow_input in ['uw', 'uw_eff', 'dw_eff']:
        wt = IEA34_130_2WT_Surrogate(inflow_input=inflow_input)

        ref_loads = [2900, 5733, 8158, 3325, 2238]
        if 'eff' in inflow_input:
            kwargs = {'TI_eff': np.array([ti])}
        else:
            kwargs = {'TI': np.array([ti])}

        loads = wt.loadFunction(ws=np.array([ws]), Alpha=np.array([shear]),
                                dw_ijlk=np.array([-200]), hcw_ijlk=np.array([0]), **kwargs)
        err = (np.array(ref_loads) - np.array(loads)[:, 0]) / ref_loads * 100
        npt.assert_array_less(np.abs(err), [7, 1, 5, 17, 4])


def test_two_turbine_wake_10_01():
    ws, ti, shear = 10, .1, 0

    for inflow_input in ['uw', 'uw_eff', 'dw_eff']:
        wt = IEA34_130_2WT_Surrogate(inflow_input=inflow_input)

        ref_loads = [4148, 5986, 12657, 4531, 3045]
        if 'eff' in inflow_input:
            kwargs = {'TI_eff': np.array([ti])}
        else:
            kwargs = {'TI': np.array([ti])}

        loads = wt.loadFunction(ws=np.array([ws]), Alpha=np.array([shear]),
                                dw_ijlk=np.array([3 * 130]), hcw_ijlk=np.array([-65]), **kwargs)
        print(loads)
        err = (np.array(ref_loads) - np.array(loads)[:, 0]) / ref_loads * 100
        npt.assert_array_less(np.abs(err), [7, 1, 5, 17, 4])


# def test_one_vs_two_turbine():
#     f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
#     inp = pd.DataFrame(eval(Path(f + 'input_one_turbine.json').read_text()))
#     ws, ti, shear = inp.values.T
#     wt1 = IEA34_130_1WT_Surrogate()
#     wt2 = IEA34_130_2WT_Surrogate('uw')
#
#     loads1 = wt1.loadFunction(ws=ws, TI_eff=ti, Alpha=shear)
#     loads2 = wt2.loadFunction(ws=ws, TI=ti, Alpha=shear, dw_ijlk=ws * 0 - 10, hcw_ijlk=ws * 0)
#     outp = pd.read_csv(f + 'stats_one_turbine_del.csv').iloc[:, [28, 29, 1, 2, 9]]
#
#     for s, o, l1, l2 in zip(outp, outp.values.T, loads1, loads2):
#         plt.figure()
#         plt.title(s)
#         plt.plot(o, l2, '.', label='Two WT')
#         plt.plot(o, l1, '.', label='One WT')
#         plt.plot([o.min(), o.max()], [o.min(), o.max()], 'k')
#         plt.plot([o.min(), o.max()], [o.min() * 1.1, o.max() * 1.1], 'k')
#         plt.plot([o.min(), o.max()], [o.min() * .9, o.max() * .9], 'k')
#         plt.xlabel('One WT HAWC2 mean')
#         plt.ylabel('Surrogate output')
#         print(s)
#         print(np.nanmean(np.abs(o - l1)))
#         print(np.nanmean(np.abs(o - l2)))
#
#         # plt.savefig(f'one_ws_two_{s.replace(":","").replace("/","")}.png')
#     if 0:
#         plt.show()


def test_functionSurrogate():
    surrogate_path = Path(example_data_path) / 'iea34_130rwt' / 'one_turbine'
    load_sensors = ['del_blade_flap', 'del_blade_edge']

    loadFunction = FunctionSurrogates(
        [TensorflowSurrogate.from_h5_json(surrogate_path / s, 'operating') for s in load_sensors],
        input_parser=lambda ws, TI_eff=.1, Alpha=0: [ws, TI_eff, Alpha])

    assert loadFunction.output_keys == [
        'MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1',
        'MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1']
    npt.assert_array_almost_equal(loadFunction(np.array([10, 11])), [[2899.589, 3185.122], [5742.616, 5706.622]], 3)

#
# def test_IEA34_130_WakeDeficitModel_1wt_wake_lines():
#     site = IEA37Site(16)
#     windTurbines = IEA34_130_1WT_Surrogate()
#     axes = plt.subplots(1, 2, figsize=(12, 6))[1].flatten()
#     plot = 1
#     for ax, rotor_avg in zip(axes, (False, True)):
#         wf_model = PropagateDownwind(
#             site, windTurbines,
#             wake_deficitModel=IEA34_130_DeficitModel(inflow_input='uw', shear_alpha=0, rotor_avg=rotor_avg),
#             superpositionModel=LinearSum(),
#             turbulenceModel=STF2017TurbulenceModel())
#
#         sim_res = wf_model([0], [0], wd=270, ws=8.44243438)
#         e = 50
#         fm_dw = sim_res.flow_map(XYGrid(x=np.linspace(-e * 130, e * 130, 1000) + 4 * 130,
#                                         y=np.array([-10, -5, 0, 5, 10]) * 130))
#         for y in fm_dw.y.values:
#             ax.plot(fm_dw.x / 130, fm_dw.sel(y=y).WS_eff.squeeze(), label=f'y={y / 130}D')
#         ax.set_title(('Rotor center wind speed', 'Rotor avg wind speed')[rotor_avg])
#         ax.set_xlabel('Downstream distance [D]')
#         ax.set_ylabel('Wind speed [m/s]')
#         ax.legend()
#         # ax.set_ylim([9.8, 10.2])
#
#     if plot:
#         plt.tight_layout()
#         plt.show()
#
#
# def test_IEA34_130_WakeDeficitModel_1wt_wake_map():
#     site = IEA37Site(16)
#     windTurbines = IEA34_130_1WT_Surrogate()
#     axes = plt.subplots(1, 2, figsize=(12, 6))[1].flatten()
#     plot = 1
#     for ax, rotor_avg in zip(axes, (False, True)):
#         wf_model = PropagateDownwind(
#             site, windTurbines,
#             wake_deficitModel=IEA34_130_DeficitModel(inflow_input='uw', shear_alpha=0, rotor_avg=rotor_avg),
#             superpositionModel=LinearSum(),
#             turbulenceModel=STF2017TurbulenceModel())
#
#         sim_res = wf_model([0], [0], wd=270, ws=10)
#         e = 5
#         flow_map = sim_res.flow_map(
#             XYGrid(x=np.linspace(-e * 130, e * 130, 500) + 4 * 130, y=np.linspace(-e * 130, e * 130, 500)))
#         # flow_map.plot_wake_map(ax=ax, levels=np.linspace(9, 11), cmap='seismic')
#         flow_map.plot_wake_map(ax=ax)
#         ax.set_title(('Rotor center wind speed', 'Rotor avg wind speed')[rotor_avg])
#
#     if plot:
#         plt.tight_layout()
#         plt.show()
#
#
# def test_IEA34_130_WakeDeficitModel_wake_map():
#     site = IEA37Site(16)
#     x, y = site.initial_position.T
#     windTurbines = IEA34_130_1WT_Surrogate()
#     axes = plt.subplots(1, 2, figsize=(12, 6))[1].flatten()
#     plot = 1
#     for ax, rotor_avg in zip(axes, (False, True)):
#         wf_model = PropagateDownwind(
#             site, windTurbines,
#             wake_deficitModel=IEA34_130_DeficitModel(inflow_input='uw', shear_alpha=0, rotor_avg=rotor_avg),
#             superpositionModel=LinearSum(),
#             turbulenceModel=STF2017TurbulenceModel())
#
#         sim_res = wf_model(x, y, wd=270, ws=10)
#         e = 5
#         flow_map = sim_res.flow_map()
#         flow_map.plot_wake_map(ax=ax, levels=100)
#         ax.set_title(('Rotor center wind speed', 'Rotor avg wind speed')[rotor_avg])
#
#     if plot:
#         plt.tight_layout()
#         plt.show()
#
#
# def test_IEA34_130_TurbulenceModel_1wt_wake_lines():
#     site = IEA37Site(16, ti=.1)
#     windTurbines = IEA34_130_1WT_Surrogate()
#
#     wf_model = PropagateDownwind(
#         site, windTurbines,
#         wake_deficitModel=IEA34_130_DeficitModel(inflow_input='uw', shear_alpha=0, rotor_avg=False),
#         superpositionModel=LinearSum(),
#         turbulenceModel=IEA34_130_TurbulenceModel(inflow_input='uw', shear_alpha=0))
#
#     sim_res = wf_model([0], [0], wd=270, ws=10)
#     e = 50
# #     fm_dw = sim_res.flow_map(XYGrid(x=np.linspace(-e * 130, e * 130, 1000) + 4 * 130,
# #                                     y=np.array([-10, -5, 0, 5, 10]) * 130))
#     fm_dw = sim_res.flow_map(XYGrid(x=[-130],
#                                     y=[0]))
#     for y in fm_dw.y.values:
#         plt.plot(fm_dw.x / 130, fm_dw.sel(y=y).TI_eff.squeeze() * 100, label=f'y={y / 130}D')
#     ax = plt.gca()
#     ax.set_xlabel('Downstream distance [D]')
#     ax.set_ylabel('Turbulence intensity [%]')
#     ax.legend()
#     ax.set_ylim([10, 11])
#     plt.show()
#
#
# def test_IEA34_130_TurbulenceModel_1wt_wake_map():
#     site = IEA37Site(16, ti=.1)
#     windTurbines = IEA34_130_1WT_Surrogate()
#     plot = 1
#
#     wf_model = PropagateDownwind(
#         site, windTurbines,
#         wake_deficitModel=IEA34_130_DeficitModel(inflow_input='uw', shear_alpha=0, rotor_avg=False),
#         superpositionModel=LinearSum(),
#         turbulenceModel=IEA34_130_TurbulenceModel(inflow_input='uw', shear_alpha=0))
#
#     sim_res = wf_model([0], [0], wd=270, ws=10)
#
#     e = 5
#     flow_map = sim_res.flow_map(
#         XYGrid(x=np.linspace(-e * 130, e * 130, 500) + 4 * 130, y=np.linspace(-e * 130, e * 130, 500)))
#     # flow_map.plot_wake_map(ax=ax, levels=np.linspace(9, 11), cmap='seismic')
#     flow_map.plot_ti_map()
#     if plot:
#         plt.tight_layout()
#         plt.show()
