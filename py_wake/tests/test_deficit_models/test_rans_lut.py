from py_wake import np
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.examples.data import hornsrev1
import matplotlib.pyplot as plt
from py_wake.deficit_models.rans_lut import RANSLUTDeficit
from py_wake.flow_map import HorizontalGrid
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.wind_farm_models.engineering_models import All2AllIterative
import pytest
from py_wake.utils.profiling import timeit
import xarray as xr


def test_rans_lut():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    path_LUT = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'
    # Create a simple LUT file for testing from Fuga (without wake added ti)
    # To be replaced with a real RANS based LUT file
    lut = xr.load_dataset(path_LUT)
    lut = lut.rename(UL='deficits')
    Dref = wts.diameter()
    zRef = wts.hub_height()
    cts = [0.1, 0.818]
    deficit = lut.deficits.transpose('x', 'y', 'z')
    # Mirror deficit at y=0 since Fuga only stores one side of the wake
    deficit2 = deficit.copy()
    deficit['y'] = -deficit['y']
    deficit = xr.merge([deficit, deficit2])
    # Add ct and ti dimensions
    subluts_ct = []
    for ct in cts:
        subluts_ct.append(-deficit * ct)
    da = xr.concat(subluts_ct, dim='ct').assign_coords(ct=cts)
    da['x'] = da['x'] / Dref
    da['y'] = da['y'] / Dref
    da['z'] = (da['z'] - zRef) / Dref
    da = da.deficits
    da = da.astype('float64')
    deficit = RANSLUTDeficit(da, zRef, Dref=Dref)
    wfm = All2AllIterative(site,
                           wts,
                           wake_deficitModel=deficit,
                           blockage_deficitModel=deficit,
                           )

    simres, _ = timeit(wfm.__call__, verbose=0, line_profile=0,
                       profile_funcs=[GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(simres.WS_eff_ilk.flatten(), [9.96796393, 9.92840392, 8.47357439, 9.99728889, 9.33989609,
                                                                8.50307297, 7.98284938], 8)
    npt.assert_array_almost_equal(simres.ct_ilk.flatten(), [0.7934485, 0.79400234, 0.80647357, 0.79303796, 0.80224145,
                                                            0.80650307, 0.80598285], 8)

    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)
    flow_map70 = simres.flow_map(HorizontalGrid(x_j, y_j, h=70))
    flow_map73 = simres.flow_map(HorizontalGrid(x_j, y_j, h=73))

    X, Y = flow_map70.XY
    Z70 = flow_map70.WS_eff_xylk[:, :, 0, 0]
    Z73 = flow_map73.WS_eff_xylk[:, :, 0, 0]

    if 0:
        flow_map70.plot_wake_map(levels=np.arange(6, 10.5, .1))
        plt.plot(X[0], Y[140])
        plt.figure()
        plt.plot(X[0], Z70[140, :], label="Z=70m")
        plt.plot(X[0], Z73[140, :], label="Z=73m")
        plt.plot(X[0, 100:400:10], Z70[140, 100:400:10], '.')
        print(list(np.round(Z70.data[140, 100:400:10], 4)))
        print(list(np.round(Z73.data[140, 100:400:10], 4)))
        plt.legend()
        plt.show()
