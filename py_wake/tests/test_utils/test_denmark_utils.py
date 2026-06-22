import matplotlib.pyplot as plt
import numpy.testing as npt
import pytest

from py_wake.utils.denmark_utils import DKWindTurbines


def test_get_dk_turbine_data():

    xlim, ylim = (690000, 700000), (6170000, 6180000)
    dk_wt = DKWindTurbines(update_cache=0)
    dk_wt.set_filter(xlim, ylim)
    dk_wt.plot()
    if 0:
        plt.show()
    wt, type, (x, y), wt_id = dk_wt.get_WindTurbines()
    assert wt.name(2) == 'NTK 500'
    assert wt.diameter(2) == 41
    assert wt.hub_height(2) == 37
    assert wt_id[type == 2] == '570714700000004575'

    dk_wt = DKWindTurbines(update_cache=False)

    ps = dk_wt.dataframe[dk_wt.dataframe['Møllenummer (GSRN)'].astype(str) == '570715000000062322'].iloc[0]
    assert ps['MatrikelLav'] == 'Horns Rev'
    assert ps.Typebetegnelse == 'V 80'
    assert ps['Kapacitet (kW)'] == 2000
    assert ps['Rotor-diameter (m)'] == 80
    assert ps['Navhøjde (m)'] == 70


def test_production():
    dk_wt = DKWindTurbines()
    da = dk_wt.get_production(update_cache=1)
    da = dk_wt.get_production(update_cache=0)
    expected_value = 2300 * 4000 * 10  # Rødsand2, 2300kW * 4000 hours * 10 years
    npt.assert_allclose(
        da.sel(id='570715000000090202', time=slice(f"2012-01", f"2022-01-01")).sum(), expected_value, rtol=0.05)


@pytest.mark.parametrize('wf_name', DKWindTurbines().wf_filters.keys())
def test_get_wind_farm(wf_name):
    dk_wt = DKWindTurbines()
    n_wt_dict = {'Rødsand1': 72, 'Rødsand2': 90, 'Hornsrev1': 80, 'Hornsrev2': 91, 'Hornsrev3': 49,
                 'Anholt': 111, 'Middelgrunden': 20, 'Vesterhav syd': 20, 'Vesterhav nord': 21, 'Krigers Flak': 72
                 }

    wf = dk_wt.get_wind_farm(wf_name)
    if 0:
        wf.plot()
        plt.title(wf_name)
        plt.show()
    assert len(wf) == n_wt_dict[wf_name]
