from py_wake.utils.maps import dk_coast
import matplotlib.pyplot as plt
from py_wake.tests import npt
import pytest


@pytest.mark.parametrize('crs,bounds', [(None, [7., 54., 14., 58.]),
                                        ('EPSG:25832', [381782.654552, 5983532.157506, 827373.774988, 6432168.470282])])
def test_dk_coast(crs, bounds):
    dk = dk_coast(crs)
    npt.assert_array_almost_equal(dk.total_bounds, bounds)
    dk.plot()
    if 0:
        plt.show()
    plt.close('all')
