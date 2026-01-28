import os
import xarray as xr
import numpy as np
from py_wake.wind_turbines.power_ct_functions import PowerCtXr
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils.plotting import setup_plot
from py_wake.noise_models.utils import (sound_power_level_interp,
                                        interpolate_directivity)

from scipy.interpolate import (
    griddata, RectSphereBivariateSpline, RegularGridInterpolator)
import gc


class TurbineNoise(WindTurbine):
    """Wind turbine model using data from turbine_noise.nc, including sound power levels and spherical delta_swl interpolation."""

    def __init__(self):
        # Load dataset
        ds = xr.load_dataset(
            os.path.join(os.path.dirname(__file__), "turbine_noise.nc")
        )

        self.ds = ds
        # Power and Ct functions
        power_ct = PowerCtXr(ds, "kW")

        WindTurbine.__init__(
            self,
            name="TurbineNoise",
            diameter=142,
            hub_height=109,
            powerCtFunction=power_ct,
        )


def main():
    if __name__ == "__main__":
        import matplotlib.pyplot as plt

        wt = TurbineNoise()

        ws = np.arange(8, 14)
        for m in range(2):
            plt.plot(ws, wt.power(ws, mode=m) / 1000, label=f"mode: {m}")
        setup_plot(xlabel="Wind speed [m/s]", ylabel="Power [kW]")

        freqs, swl = sound_power_level_interp(
            ds=wt.ds, ws=10, mode=m, freq=wt.ds.freq.values, yaw=20)

        plt.figure()
        for m in range(2):
            freq, sound_power = sound_power_level_interp(ds=wt.ds, ws=10,
                                                         mode=m, freq=wt.ds.freq.values,
                                                         yaw=20)
            # print(sound_power.shape)
            plt.semilogx(freq, sound_power.squeeze(), label=f"mode: {m}")
            setup_plot(xlabel="Frequency [Hz]", ylabel="Sound power [dB]")
        plt.show()

        # Test spherical delta_swl interpolation
        mode = [0]
        freqs = wt.ds["freq"].values
        ws_arr = np.full_like(freqs, 10)
        yaws = np.full_like(freqs, 0)
        lat_deg = [-55.0, 56, 67]
        lon_deg = [8.0, -12, -23]

        delta_swl_interp = interpolate_directivity(
            wt.ds, mode, freqs, ws_arr, yaws, lat_deg, lon_deg
        )

        print(
            f"Spherical Delta SWL interpolation result shape: {delta_swl_interp.shape}")

        cases = np.array(
            [
                [10.0, -20.0, 0],  # wind, yaw, mode
                [11.0, 10.0, 1],
                [12.0, 0.0, 0],
            ]
        )

        obs_lats = np.array([45.0, -15.0, 60.0, 0.0, -60.0])
        obs_lons = np.array([-30.0, 120.0, 45.0, 0.0, -120.0])
        # freqs = np.array([63.0, 125.0, 500.0, 2000.0, 4000.0, 8000.0])
        freqs = np.array([125.0, 500.0, 2000.0, 4000.0])

        result = interpolate_directivity(
            wt.ds,
            cases[:, 2],  # mode
            freqs,
            cases[:, 0],  # ws
            cases[:, 1],  # yaw
            obs_lats,
            obs_lons,
        )

        print(f"Multi-dimensional interpolation result shape: {result.shape}")


main()
