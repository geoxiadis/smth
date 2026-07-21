import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy import newaxis as na

from py_wake.deficit_models import BastankhahGaussianDeficit, SelfSimilarityDeficit2020
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.flow_map import Points, XYGrid
from py_wake.site.streamline_distance import StreamlineDistance
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.utils import layouts
from py_wake.utils.plotting import setup_plot
from py_wake.utils.profiling import profileit, timeit
from py_wake.utils.streamline import VectorField3D
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind
from py_wake.wind_farm_models.external_wind_farm_models import (
    ExternalWFMWindFarm, ExternalXRAbsWindFarm, ExternalXRRelWindFarm)
from py_wake.wind_turbines import WindTurbines


def get_wfm(externalWindFarms=[], wfm_cls=PropagateDownwind, site=Hornsrev1Site(), blockage=False):
    windTurbines = WindTurbines.from_WindTurbine_lst([V80()] * 6)
    windTurbines._names = ["Current WF"] + [f"WF{i + 1}" for i in np.arange(5)]
    kwargs = dict(site=site, windTurbines=windTurbines,
                  wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True),
                  superpositionModel=LinearSum(),
                  externalWindFarms=externalWindFarms)
    if blockage:
        wfm_cls = All2AllIterative
        kwargs['blockage_deficitModel'] = SelfSimilarityDeficit2020()
    return wfm_cls(**kwargs)


def setup_ext_farms(cls, neighbour_x_y_angle, wfm=get_wfm(), include_wd_range=np.arange(-45, 45)):
    ext_farms = []
    for i, (x, y, angle) in enumerate(neighbour_x_y_angle, 1):
        wd_lst = (include_wd_range + angle) % 360  # relevant wind directions
        name = f'WF{i}'
        if cls is ExternalWFMWindFarm:
            ext_wf = ExternalWFMWindFarm(name, wfm, x, y, include_wd=wd_lst)
        elif cls is ExternalXRRelWindFarm:
            # Coarse grid in relative downwind, crosswind and vertical direction
            grid_xyh = (np.linspace(1200, 2500, 12),
                        np.linspace(-1000, 1000, 20),
                        np.array([0]))
            ext_wf = ExternalXRRelWindFarm.generate(name, grid_xyh, wfm, wt_x=x, wt_y=y, wd=wd_lst)
        elif cls is ExternalXRAbsWindFarm:
            # coarse grid in East, North and vertical direction covering the current wind farm
            e = 500
            grid_xyh = (np.linspace(- e, e, 20),
                        np.linspace(- e, e, 20),
                        wfm.windTurbines.hub_height())
            ext_wf = ExternalXRAbsWindFarm.generate(name, grid_xyh, wfm, wt_x=x, wt_y=y, wd=wd_lst)
        ext_farms.append(ext_wf)
    return ext_farms


def test_aep():
    # setup current, neighbour and all positions
    wf_x, wf_y = layouts.circular([1, 5], 400)

    wts = WindTurbines.from_WindTurbine_lst([V80()] * 6)
    wts._names = ["Current WF"] + [f"WF{i + 1}" for i in np.arange(5)]

    No_neighbours = 3
    neighbour_x_y_angle = [(wf_x + 2000 * np.cos(d), wf_y + 2000 * np.sin(d), (90 - np.rad2deg(d)) % 360)
                           for d in np.pi + np.linspace(0, np.pi / 2, No_neighbours)]
    neighbour_x, neighbour_y, _ = zip(*neighbour_x_y_angle)

    all_x, all_y = np.r_[wf_x, np.array(neighbour_x).flatten()], np.r_[wf_y, np.array(neighbour_y).flatten()]

    for wfm_cls in [PropagateDownwind, All2AllIterative]:
        def run():
            return get_wfm(wfm_cls=wfm_cls)(all_x, all_y, type=0).aep().isel(wt=np.arange(len(wf_x)))
        aep_ref, t = timeit(run)()
        # print(wfm_cls.__name__, np.mean(t), aep_ref.sum().item())

        ext_cls_lst = [ExternalWFMWindFarm, ExternalXRAbsWindFarm, ExternalXRRelWindFarm]
        for cls in ext_cls_lst:
            def run():
                ext_farm = setup_ext_farms(cls, neighbour_x_y_angle)
                wfm = get_wfm(ext_farm, wfm_cls=wfm_cls)
                if 0:
                    wfm(wf_x, wf_y, type=0, wd=254, ws=10).flow_map().plot_wake_map()
                    plt.title(f'{wfm_cls.__name__}, {cls.__name__}')
                    plt.show()
                return wfm(wf_x, wf_y, type=0).aep().isel(wt=np.arange(len(wf_x)))
            aep, t = timeit(run, )()

            err_msg = f'{wfm_cls.__name__}, {cls.__name__}'
            atol = {ExternalXRAbsWindFarm: 0.0006, ExternalXRRelWindFarm: 0.001}.get(cls, 1e-6)
            # print(err_msg, np.mean(t), aep.sum().item())
            npt.assert_allclose(aep.sum().item(), aep_ref.sum().item(), atol=atol, err_msg=err_msg)
            npt.assert_allclose(aep.values, aep_ref.values, atol=atol, err_msg=err_msg)


def test_functionality():
    wf_x, wf_y = layouts.circular([1, 5], 400)

    wts = WindTurbines.from_WindTurbine_lst([V80()] * 6)
    wts._names = ["Current WF"] + [f"WF{i + 1}" for i in np.arange(5)]

    No_neighbours = 3
    neighbour_x_y_angle = [(wf_x + 2000 * np.cos(d), wf_y + 2000 * np.sin(d), (90 - np.rad2deg(d)) % 360)
                           for d in np.pi + np.linspace(0, np.pi / 2, No_neighbours)]
    neighbour_x, neighbour_y, _ = zip(*neighbour_x_y_angle)

    all_x, all_y = np.r_[wf_x, np.array(neighbour_x).flatten()], np.r_[wf_y, np.array(neighbour_y).flatten()]
    ext_farm = setup_ext_farms(ExternalWFMWindFarm, neighbour_x_y_angle)
    wfm = get_wfm(ext_farm)
    wfm(wf_x, wf_y, wd=234, ws=10).flow_map().plot_wake_map()
    if 0:
        plt.show()
    plt.close('all')

    npt.assert_array_equal([wd for wd in np.arange(360) if ext_farm[0].include_wd_func(wd)], np.arange(225, 315))
    ext_farm[0].set_include_wd(np.arange(225, 300))
    npt.assert_array_equal([wd for wd in np.arange(360) if ext_farm[0].include_wd_func(wd)], np.arange(225, 300))
    ext_farm[0].set_include_wd(lambda wd: 235 <= wd <= 245)
    npt.assert_array_equal([wd for wd in np.arange(360) if ext_farm[0].include_wd_func(wd)], np.arange(235, 246))

    npt.assert_array_equal(ext_farm[0].get_relevant_wd((wf_x, wf_y, wts.hub_height())), np.arange(236, 305))
    npt.assert_array_equal(ext_farm[0].get_relevant_wd((wf_x, wf_y, wts.hub_height()), ws=18), np.arange(237, 304))
    npt.assert_array_equal(ext_farm[0].get_relevant_wd((wf_x, wf_y, wts.hub_height()), tol=1e-3), np.arange(240, 301))

    ext_farm = setup_ext_farms(ExternalXRAbsWindFarm, neighbour_x_y_angle[:1])
    assert ext_farm[0].to_xarray().deficit.shape == (20, 20, 1, 90, 23)


def test_cluster_interaction():
    wf_x, wf_y = layouts.circular([1, 5, 12, 18], 1800)
    No_neighbours = 2
    neighbour_x_y_angle = [(wf_x - 6000 * i, wf_y, 270) for i in range(1, No_neighbours + 1)]
    neighbour_x, neighbour_y, _ = zip(*neighbour_x_y_angle)
    all_x, all_y = np.r_[wf_x, np.array(neighbour_x).flatten()], np.r_[wf_y, np.array(neighbour_y).flatten()]

    types = [v for i in range(No_neighbours + 1) for v in [i] * len(wf_x)]
    wd = 270
    wfm_ref = get_wfm()
    sim_res_ref = wfm_ref(all_x, all_y, type=types, wd=wd, ws=10)

    ext_farms = setup_ext_farms(ExternalWFMWindFarm, neighbour_x_y_angle)
    wfm = get_wfm(ext_farms)
    sim_res_ext = wfm(wf_x, wf_y, type=0, wd=wd, ws=10)
    if 0:
        grid = XYGrid(x=np.linspace(-15000, 3000, 150), y=np.linspace(-2500, 2500, 100))
        fm_ref = sim_res_ref.flow_map(grid)
        fm_ref.plot_wake_map(levels=np.linspace(4, 10, 50))
        setup_plot(grid=False, figsize=(12, 3), axis='scaled')
        plt.show()
    npt.assert_allclose(sim_res_ext.Power[:len(wf_x)].sum().item(),
                        sim_res_ref.Power[:len(wf_x)].sum().item(), atol=26000)


def test_cluster_blockage():
    wf_x, wf_y = layouts.circular([1, 5, 12, 18], 1800)

    No_neighbours = 1
    neighbour_x_y_angle = [(wf_x - 6000 * i, wf_y, 270) for i in range(1, No_neighbours + 1)]
    neighbour_x, neighbour_y, _ = zip(*neighbour_x_y_angle)
    all_x, all_y = np.r_[wf_x, np.array(neighbour_x).flatten()], np.r_[wf_y, np.array(neighbour_y).flatten()]
    types = [v for i in range(No_neighbours + 1) for v in [i] * len(wf_x)]

    wfm_ref = get_wfm(blockage=True)
    sim_res_ref = wfm_ref(all_x, all_y, type=types, wd=270, ws=10)

    ext_farms = setup_ext_farms(ExternalWFMWindFarm, neighbour_x_y_angle, wfm=wfm_ref)
    wfm = get_wfm(ext_farms, blockage=True)
    sim_res_ext = wfm(wf_x, wf_y, type=0, wd=270, ws=10)

    # all WT without blockage
    npt.assert_allclose(sim_res_ref.Power[:len(wf_x)].sum().item(),
                        get_wfm()(all_x, all_y, wd=270, ws=10).Power[:len(wf_x)].sum().item(), atol=9000)
    # Current + external farms with blockage
    npt.assert_allclose(sim_res_ref.Power[:len(wf_x)].sum().item(),
                        sim_res_ext.Power[:len(wf_x)].sum().item(), atol=400)


def test_streamlines():
    wf_x, wf_y = layouts.circular([1, 5], 400)
    H = 70

    class MyVectorField(VectorField3D):
        def __init__(self):
            pass

        def __call__(self, wd, time, x, y, h):
            turning = (x + 1000) / 50
            theta = np.deg2rad(270 - wd + turning)
            return np.array([np.cos(theta), np.sin(theta), theta * 0]).T

    vf3d = MyVectorField()
    site = Hornsrev1Site()
    site_with_streamlines = Hornsrev1Site()

    site_with_streamlines.distance = StreamlineDistance(vf3d)

    wfm_ref = get_wfm(site=site_with_streamlines)
    No_neighbours = 1
    neighbour_x_y_angle = (wf_x - 2000, wf_y, 270)
    neighbour_x, neighbour_y, _ = neighbour_x_y_angle
    all_x, all_y = np.r_[wf_x, np.array(neighbour_x).flatten()], np.r_[wf_y, np.array(neighbour_y).flatten()]
    types = [v for i in range(No_neighbours + 1) for v in [i] * len(wf_x)]

    grid = XYGrid(x=np.linspace(-4000, 2000, 150), y=np.linspace(-1500, 1500, 200))

    sim_res_ref = wfm_ref(all_x, all_y, type=types, wd=270, ws=10)
    df = pd.DataFrame({'Power current WF [MW]': []})
    P_ref = sim_res_ref.Power[:len(wf_x)].sum().item()

    if 0:
        fm_ref = sim_res_ref.flow_map(grid)
        fm_ref.plot_wake_map()
        stream_lines = vf3d.stream_lines(wd=np.full_like(neighbour_x, 270), start_points=np.array([neighbour_x, neighbour_y, np.full_like(neighbour_x, 70)]).T,
                                         dw_stop=np.full_like(neighbour_x, 2000))
        for sl in stream_lines:
            plt.plot(sl[:, 0], sl[:, 1], 'k', lw=1, alpha=0.1)

        setup_plot(axis='scaled', xlabel='', ylabel='', xlim=[-4000, 2000], grid=0, figsize=(12, 4))
        plt.show()

    for cls in [ExternalWFMWindFarm, ExternalXRAbsWindFarm, ExternalXRRelWindFarm]:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        for i, (ax, site_setup) in enumerate(zip(axes, [site_with_streamlines, site]), 1):
            # setup external wind farm model
            # - option 1: with streamlines
            # - option 2: without streamlines
            ext_farms = setup_ext_farms(cls, [neighbour_x_y_angle], wfm=get_wfm(site=site_setup),
                                        include_wd_range=np.array([0]))

            # Setup simulation using site with stream lines
            wfm = get_wfm(ext_farms, site=site_with_streamlines)
            sim_res_ext = wfm(wf_x, wf_y, type=0, wd=270, ws=10)
            s = f'{cls.__name__}, option {i}'
            if i == 2:
                rtol = 0.05
            else:
                rtol = {ExternalWFMWindFarm: 1e-7,
                        ExternalXRAbsWindFarm: 0.0005,
                        ExternalXRRelWindFarm: 0.009}[cls]
            npt.assert_allclose(sim_res_ext.Power[:len(wf_x)].sum().item(), P_ref, rtol=rtol, err_msg=s)

            if 0:
                fm = sim_res_ext.flow_map(grid)

                fm.plot_wake_map(ax=ax)

                if site_setup == site_with_streamlines:
                    start_points, alpha = np.array(
                        [ext_farms[0].wt_x, ext_farms[0].wt_y, ext_farms[0].wt_x * 0 + H]).T, 0.2
                else:
                    start_points, alpha = np.array([[ext_farms[0].wf_x, ext_farms[0].wf_y, 70]]), .5
                stream_lines = vf3d.stream_lines(wd=[270], start_points=start_points,
                                                 dw_stop=np.full(start_points[:, 0].shape, 2000))
                for sl in stream_lines:
                    ax.plot(sl[:, 0], sl[:, 1], alpha=alpha)

                setup_plot(ax=ax, axis='scaled', xlabel='', ylabel='', title=f'{cls.__name__}, option {i}', grid=0)

    # plt.show()
    plt.close('all')
