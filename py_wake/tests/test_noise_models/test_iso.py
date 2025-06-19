import numpy as np
from py_wake.noise_models.iso import ISONoiseModel
from py_wake.tests import npt
from numpy import newaxis as na
from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.examples.data.swt_dd_142_4100_noise.swt_dd_142_4100 import SWT_DD_142_4100
from py_wake.examples.data.noise_example.turbine_noise import TurbineNoise
from py_wake.site._site import UniformSite
from py_wake.utils.layouts import rectangle
import warnings


def test_iso_noise_model():

    Temp = 20  # Temperature
    RHum = 80.0  # Relative humidity

    rec_x = [2000, 3000, 4000]  # receiver xy-position
    rec_y = [0, 0, 0]

    source_x = [0, 0]  # source xy-position
    source_y = [0, 500]

    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])

    # If the source and receiver are placed in complex
    # terrain then the height of the terrain should be added to these heights.
    z_hub = 100  # source height
    z_rec = 2  # receiver height

    ground_type = 1  # Ranging from 0 to 1 (hard to soft ground)

    sound_power_level = np.array([[89.4, 93.6, 97.2, 98.6, 101., 102.3, 96.7, 84.1],
                                  [89.2, 93.2, 96.2, 97.6, 100., 101.3, 95.7, 83.1]])
    iso = ISONoiseModel(src_x=source_x, src_y=source_y, src_h=z_hub, freqs=freqs, sound_power_level=sound_power_level)
    patm = 101325.0  # atmospheric pressure in pascal
    Delta_SPL = iso.transmission_loss(rec_x, rec_y, z_rec, ground_type, patm, Temp, RHum)
    delta_SPL_ref = [[[-74.18868997998351, -82.6237111823142, -85.106899362965, -84.78075941279131, -87.47936788035022,
                       -95.0531580482448, -119.90461076531315, -216.18076560019855],
                      [-77.78341234414849, -86.43781608834009, -89.6588031834609, -91.0544347291193, -96.14098255652445,
                       -107.56227940740277, -144.8146479691885, -289.1327626666617],
                      [-79.65387282494626, -89.23269240205617, -93.19182772534491, -96.30991899366798, -103.78536068849834,
                       -119.05570214846978, -168.71394345143312, -361.09322135290586]],
                     [[-74.4562086405804, -82.90475257479693, -85.43331381258002, -85.21311520726341, -88.05865461649006,
                       -95.86918353475227, -121.48366996751929, -220.71586737235828],
                      [-77.90553623031623, -86.56901853514375, -89.82054728871552, -91.28744741509202, -96.47283823614013,
                         -108.0533933516973, -145.81906793231144, -292.12576375254764],
                      [-79.70589475152278, -89.3092674680563, -93.29138286495427, -96.46309784962982, -104.01291072740128,
                         -119.40308086820056, -169.44754252350324, -363.32306335554176]]]
    npt.assert_array_almost_equal(delta_SPL_ref, Delta_SPL[:, :, 0, 0], 8)
    total_spl, spl = iso(rec_x, rec_y, z_rec, ground_type=ground_type, patm=patm, Temp=Temp, RHum=RHum)
    npt.assert_array_almost_equal([23.068816073636583, 18.034141554900494, 15.043308370722233],
                                  total_spl[:, 0, 0])


def test_iso_noise_map():
    """Test noise map functionality"""
    wt = SWT_DD_142_4100()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'The BastankhahGaussian model is not representative of the setup')
        wfm = BastankhahGaussian(UniformSite(), wt)
    x, y = rectangle(3, 3, 5 * wt.diameter())

    sim_res = wfm(x, y, wd=270, ws=10, mode=1, yaw=0)

    nm = sim_res.noise_model()
    total_spl_jlk, spl_jlkf = nm(rec_x=[1000, 2000], rec_y=[500, 500], rec_h=2,
                                 patm=101325.0, Temp=20, RHum=80, ground_type=0.5)

    nmap = sim_res.noise_map()

    assert not np.any(np.isnan(total_spl_jlk))
    assert not np.any(np.isnan(spl_jlkf))
    assert 'Total sound pressure level' in nmap
    assert 'Sound pressure level' in nmap


def test_iso_delta_swl_correction():
    """Test the critical delta_swl_correction branch using TurbineNoise directional correction"""
    wt = TurbineNoise()
    wfm = BastankhahGaussian(UniformSite(), wt)
    x, y = [0], [0]  # Single turbine

    sim_res = wfm(x, y, wd=270, ws=10, mode=1, yaw=10)

    nm = sim_res.noise_model()

    total_spl, spl = nm(rec_x=[1000], rec_y=[500], rec_h=2,
                        patm=101325.0, Temp=20, RHum=80, ground_type=0.5)

    assert not np.any(np.isnan(total_spl))
    assert total_spl.shape == (1, 1, 1)
    assert spl.shape == (1, 1, 1, len(nm.freqs))


def test_iso_edge_cases():
    """Test various edge cases for complete coverage"""
    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])

    iso = ISONoiseModel(src_x=[0], src_y=[0], src_h=100, freqs=freqs,
                        sound_power_level=np.array([[85.0, 90.0, 95.0, 92.0, 88.0, 85.0, 80.0, 75.0]]))

    total_spl_hard, _ = iso([1000], [0], rec_h=2, patm=101325.0, Temp=20, RHum=80, ground_type=0.0)
    total_spl_soft, _ = iso([1000], [0], rec_h=2, patm=101325.0, Temp=20, RHum=80, ground_type=1.0)

    iso_low = ISONoiseModel(src_x=[0], src_y=[0], src_h=10, freqs=freqs,
                            sound_power_level=np.array([[85.0, 90.0, 95.0, 92.0, 88.0, 85.0, 80.0, 75.0]]))

    total_spl_close, _ = iso_low([100], [0], rec_h=2, patm=101325.0, Temp=20, RHum=80, ground_type=1.0)
    total_spl_far, _ = iso_low([2000], [0], rec_h=2, patm=101325.0, Temp=20, RHum=80, ground_type=1.0)

    total_spl_cold, _ = iso([1000], [0], rec_h=2, patm=101325.0, Temp=-10, RHum=50, ground_type=0.5)
    total_spl_hot, _ = iso([1000], [0], rec_h=2, patm=101325.0, Temp=40, RHum=50, ground_type=0.5)

    assert not np.any(np.isnan(total_spl_hard))
    assert not np.any(np.isnan(total_spl_soft))
    assert not np.any(np.isnan(total_spl_close))
    assert not np.any(np.isnan(total_spl_far))
    assert not np.any(np.isnan(total_spl_cold))
    assert not np.any(np.isnan(total_spl_hot))
