import os
import xarray as xr
import numpy as np
from py_wake.wind_turbines.power_ct_functions import PowerCtXr
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils.plotting import setup_plot

# --- Spherical helpers ---
def sph2cart(lat_rad, lon_rad):
    c = np.cos(lat_rad)
    return np.stack((c * np.cos(lon_rad), c * np.sin(lon_rad), np.sin(lat_rad)), -1)

class TurbineNoise(WindTurbine):
    """Wind turbine model using data from turbine_noise.nc, including sound power levels and spherical delta_swl interpolation."""
    def __init__(self):
        # Load dataset
        ds = xr.load_dataset(os.path.join(os.path.dirname(__file__), 'turbine_noise.nc'))
        self.ds = ds
        # Power and Ct functions
        power_ct = PowerCtXr(ds.rename({'P': 'power', 'Ct': 'ct'}), 'kW')
        # Interpolator for sound power level (swl)
        ip = GridInterpolator([
            ds.mode.values,
            ds.freq.values,
            ds.ws.values
        ], ds.swl.values, method=['nearest', 'nearest', 'linear'])

        def sound_power_level(ws, mode, freq=None, **_):
            ws = np.atleast_1d(ws)
            mode = np.zeros_like(ws) + mode
            freqs = ds.freq.values if freq is None else np.atleast_1d(freq)
            # Create meshgrid for all combinations
            mode_grid, freq_grid, ws_grid = np.meshgrid(mode, freqs, ws, indexing='ij')
            points = np.stack([mode_grid.ravel(), freq_grid.ravel(), ws_grid.ravel()], axis=-1)
            swl = ip(points).reshape(mode_grid.shape)
            # Normalize to (n_sources, n_freq) format for consistent noise model usage
            if swl.shape[0] == 1:  # Single turbine case: (1, n_freq, conditions) → (1, n_freq)
                swl = swl[0, :, 0:1].T  # Take first condition and transpose
            else:  # Multi-turbine case: (n_sources, n_freq, conditions) → (n_sources, n_freq)
                swl = swl[..., 0]  # Take first condition
            return freqs, swl

        WindTurbine.__init__(self,
            name='TurbineNoise',
            diameter=142,  
            hub_height=109,  
            powerCtFunction=power_ct,
            sound_power_level=sound_power_level
        )

    def spherical_interpolate_delta_swl(
        self,
        mode,
        freq,
        ws,
        yaw,
        lat_deg,
        lon_deg,
        max_dist=np.pi/2,
        power=3.0,
    ):

        ds = self.ds

        freq = np.atleast_1d(freq)
        ws = np.atleast_1d(ws)
        yaw = np.atleast_1d(yaw)
        mode = np.atleast_1d(mode)
        lat_deg = np.atleast_1d(lat_deg)
        lon_deg = np.atleast_1d(lon_deg)

        ws, yaw, mode = np.broadcast_arrays(ws, yaw, mode)
        n_cases = ws.size
        n_obs = lat_deg.size
        n_freq = freq.size
        n_data_obs = ds.sizes["obs"]

        obs_xyz_data = sph2cart(np.radians(ds["lat"].values), np.radians(ds["lon"].values))  # (D,3)
        obs_xyz_target = sph2cart(np.radians(lat_deg), np.radians(lon_deg))                  # (O,3)
        dist = np.arccos(np.clip(obs_xyz_target @ obs_xyz_data.T, -1.0, 1.0))                # (O,D)
        w = np.where(dist < max_dist, 1.0 / (dist + 1e-10) ** power, 0.0)
        w_sum = w.sum(axis=1, keepdims=True)
        w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum != 0)                      # (O,D)

        delta_swl_interpolator = GridInterpolator([
            ds["mode"].values,
            ds["freq"].values,
            np.arange(n_data_obs),
            ds["ws"].values,
            ds["yaw"].values,
        ], ds["delta_swl"].values, method='linear')

        # Filter observation points by distance first
        relevant_obs_mask = (dist < max_dist).any(axis=0)  # Shape: (n_data_obs,)
        relevant_obs_indices = np.where(relevant_obs_mask)[0]
        n_relevant = len(relevant_obs_indices)
        
        if n_relevant == 0:
            return np.zeros((n_cases, n_obs, 1, 1, n_freq))
        
        w_relevant = w[:, relevant_obs_indices]  # Shape: (n_obs, n_relevant)
        
        # Efficient vectorized approach using broadcasting
        # Create parameter grids for all combinations
        case_grid, freq_grid, obs_grid = np.meshgrid(
            np.arange(n_cases), freq, relevant_obs_indices, indexing='ij'
        )
        
        # Create interpolation points
        pts = np.column_stack([
            mode[case_grid.ravel()],
            freq_grid.ravel(),
            obs_grid.ravel(),
            ws[case_grid.ravel()],
            yaw[case_grid.ravel()]
        ])
        
        # Single interpolation call
        interp_result = delta_swl_interpolator(pts)
        interp_vals = interp_result.reshape(n_cases, n_freq, n_relevant)
        
        # Apply weights: (cases, freq, relevant) @ (obs, relevant).T -> (cases, freq, obs)
        result = np.einsum('cfr,or->cfo', interp_vals, w_relevant)
        
        # Reshape to final format: (n_cases, n_obs, 1, 1, n_freq)
        # result shape is (n_cases, n_freq, n_obs), we need (n_cases, n_obs, 1, 1, n_freq)
        return result.transpose(0, 2, 1)[:, :, None, None, :]

def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        wt = TurbineNoise()
        ws = np.arange(3, 27)
        for m in range(7):
            plt.plot(ws, wt.power(ws, mode=m) / 1000, label=f'mode: {m}')
        setup_plot(xlabel='Wind speed [m/s]', ylabel='Power [kW]')

        plt.figure()
        for m in range(7):
            freq, sound_power = wt.sound_power_level(ws=10, mode=m)
            plt.plot(freq, sound_power.squeeze(), label=f'mode: {m}')
        setup_plot(xlabel='Frequency [Hz]', ylabel='Sound power [dB]')
        plt.show()

        # Test spherical delta_swl interpolation
        mode = [0]
        freqs = wt.ds['freq'].values
        ws_arr = np.full_like(freqs, 10)
        yaws = np.full_like(freqs, 0)
        lat_deg = [-55.0, 56, 67]
        lon_deg = [8.0, -12, -23]
        delta_swl_interp = wt.spherical_interpolate_delta_swl(mode, freqs, ws_arr, yaws, lat_deg, lon_deg)
        print(f"Spherical interpolation result shape: {delta_swl_interp.shape}")

        # Multi-dimensional interpolation demo
        cases = np.array([
            [6.0,  -20.0, 0],   # wind, yaw, mode
            [10.0,   0.0, 1],
            [16.0,  20.0, 2],
            [22.0,  10.0, 3],
        ])
        obs_lats = np.array([45.0, -15.0, 60.0,  0.0, -60.0])
        obs_lons = np.array([-30.0, 120.0, 45.0, 0.0, -120.0])
        freqs    = np.array([63.0, 125.0, 500.0, 2000.0, 4000.0, 8000.0])

        result = wt.spherical_interpolate_delta_swl(
            cases[:, 2],  # mode
            freqs,
            cases[:, 0],  # ws
            cases[:, 1],  # yaw
            obs_lats,
            obs_lons
        )
        print(f"Multi-dimensional interpolation result shape: {result.shape}")

main()
