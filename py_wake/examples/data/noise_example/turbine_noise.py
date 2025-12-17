import os
import xarray as xr
import numpy as np
from py_wake.wind_turbines.power_ct_functions import PowerCtXr
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils.plotting import setup_plot

from scipy.interpolate import griddata
import gc


def sph2cart(lat_rad, lon_rad):
    c = np.cos(lat_rad)
    return np.stack((c * np.cos(lon_rad), c * np.sin(lon_rad), np.sin(lat_rad)), -1)


class TurbineNoise(WindTurbine):
    """Wind turbine model using data from turbine_noise.nc, including sound power levels and spherical delta_swl interpolation."""

    def __init__(self):
        # Load dataset
        ds = xr.load_dataset(
            os.path.join(os.path.dirname(__file__), "turbine_noise.nc")
        )
        self.ds = ds
        
        if "yaw" in ds["P"].dims:
            ds2 = ds.copy()
            ds2 = ds2.assign(
                              power=ds["P"].isel(yaw=1),
                              ct=ds["Ct"].isel(yaw=1),
                            )
        else:
            ds2 = ds.copy()    
            ds2 = ds2.assign(
                              power=ds["P"],
                              ct=ds["Ct"],
                            )

        # Power and Ct functions
        # power_ct = PowerCtXr(ds.rename({"P": "power", "Ct": "ct"}), "kW")
        power_ct = PowerCtXr(ds2, "kW")
        # Interpolator for sound power level (swl)
        # Create GridInterpolator
        ip = GridInterpolator(
            [ds.mode.values, ds.freq.values, ds.ws.values, ds.yaw.values],
            self.ds.swl.values,
            method=["linear", "linear", "linear", "linear"]
        )
        
        #print(ip.shape)
        #exit()

        # def sound_power_level(ws, mode, freq=None, **_):
        #     ws = np.atleast_1d(ws)
        #     mode = np.zeros_like(ws) + mode
        #     freqs = ds.freq.values if freq is None else np.atleast_1d(freq)
        #     mode_grid, freq_grid, ws_grid = np.meshgrid(mode, freqs, ws, indexing="ij")
        #     points = np.stack(
        #         [mode_grid.ravel(), freq_grid.ravel(), ws_grid.ravel()], axis=-1
        #     )
        #     swl = ip(points).reshape(len(mode), len(freqs), len(ws))[
        #         :, :, 0
        #     ]  # Take first ws condition
        #     return freqs, swl

        def sound_power_level(ws=None, mode=None, freq=None, yaw=None, **_):

            print("In sound_power_level function:")
            print("mode:", mode)
            print("freqs:", freq)
            print("ws:", ws)
            print("yaws:", yaw)
                        
            ws = np.atleast_1d(ds.ws.values if ws is None else ws)
            mode = np.atleast_1d(ds.mode.values if mode is None else 
                                 np.zeros_like(ws) + mode)
            freqs = np.atleast_1d(ds.freq.values if freq is None else freq)
            yaws = np.atleast_1d(ds.yaw.values if yaw is None else yaw)
        
            mode_grid, freq_grid, ws_grid, yaw_grid = np.meshgrid(
                mode, freqs, ws, yaws, indexing="ij"
            )
        
            points = np.stack(
                [mode_grid.ravel(), freq_grid.ravel(), ws_grid.ravel(), yaw_grid.ravel()],
                axis=-1,
            )
            
            
            swl = ip(points).reshape(len(mode), len(freqs), len(ws), len(yaws))
            
            print("In sound_power_level:")
            print("mode:", mode)
            print("freqs:", freqs)
            print("ws:", ws)
            print("yaws:", yaws)
            print("swl shape:", swl.shape)
            # exit()
            # which wind speeds and yaw to consider? why is frederik considering only first ws?

            return freqs, swl        

        WindTurbine.__init__(
            self,
            name="TurbineNoise",
            diameter=142,
            hub_height=109,
            powerCtFunction=power_ct,
            sound_power_level=sound_power_level,
        )

    def old_spherical_interpolate_delta_swl(
        self,
        mode,
        freq,
        ws,
        yaw,
        lat_deg,
        lon_deg,
        max_dist=np.pi / 2,
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
        
        # cartesian vectors of all the observer positions
        obs_xyz_data = sph2cart(
            np.radians(ds["lat"].values), np.radians(ds["lon"].values)
        )  # (D,3)
        
        # provides the vectors in cartesian coordiantes to the target
        obs_xyz_target = sph2cart(np.radians(lat_deg), np.radians(lon_deg))  # (O,3)
        
        # dot product of the target with all observers, we get the angle between
        # the two directions signifying how aligned the two directions are
        dist = np.arccos(np.clip(obs_xyz_target @ obs_xyz_data.T, -1.0, 1.0))  # (O,D)
        
        # TODO: what is happening here?
        # doing this to get the relevant observers to our target
        # are these a weighing function? the further a meas. observer is from  
        # the target observer, i.e. the larger the angle and smaller
        # the weight
        w = np.where(dist < max_dist, 1.0 / (dist + 1e-10) ** power, 0.0)
        w_sum = w.sum(axis=1, keepdims=True)
        w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum != 0)  # (O,D)
        
        # build the grid interpolator
        delta_swl_interpolator = GridInterpolator(
            [
                ds["mode"].values,
                ds["freq"].values,
                np.arange(n_data_obs),
                ds["ws"].values,
                ds["yaw"].values,
            ],
            ds["delta_swl"].values,
            # method="linear",
            method=["linear", "linear", "linear", "linear", "linear"],
        )
        
        # our relevant meas. observers are those that are < 90 degrees angle
        # from the target
        relevant_obs_mask = (dist < max_dist).any(axis=0)  # Shape: (n_data_obs,)
        relevant_obs_indices = np.where(relevant_obs_mask)[0]
        n_relevant = len(relevant_obs_indices)

        if n_relevant == 0:
            return np.zeros((n_cases, n_obs, 1, 1, n_freq))
        
        # get the weights for the relevant observers
        w_relevant = w[:, relevant_obs_indices]  # Shape: (n_obs, n_relevant)

        case_grid, freq_grid, obs_grid = np.meshgrid(
            np.arange(n_cases), freq, relevant_obs_indices, indexing="ij"
        )
                
        pts = np.column_stack(
            [
                mode[case_grid.ravel()],
                freq_grid.ravel(),
                obs_grid.ravel(),
                ws[case_grid.ravel()],
                yaw[case_grid.ravel()],
            ]
        )

        interp_result = delta_swl_interpolator(pts)
        interp_vals = interp_result.reshape(n_cases, n_freq, n_relevant)

        result = np.einsum(
            "cfr,or->cof", interp_vals, w_relevant
        )  # Apply weights: (cases, freq, relevant) @ (obs, relevant).T -> (cases, obs, freq)
        
        print("spherical_interpolate_delta_swl result shape:")
        print(result.shape)
        print(f"result[:, :, None, None, :]  = {result[:, :, None, None, :].shape}")
        # exit()

        return result[:, :, None, None, :]  # (n_cases, n_obs, 1, 1, n_freq)
    
    def spherical_interpolate_delta_swl(
        self,
        mode,
        freq,
        ws,
        yaw,
        lat_deg,
        lon_deg
    ):
        """_summary_

        Parameters
        ----------
        mode : _type_
            _description_
        freq : _type_
            _description_
        ws : _type_
            _description_
        yaw : _type_
            _description_
        lat_deg : _type_
            _description_
        lon_deg : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # observed data
        ds = self.ds
        # unpack the observed data
        obs_lat_deg = np.rad2deg(ds["lat"].values)
        obs_lon_deg = np.rad2deg(ds["lon"].values)
        # delta_swl values
        # Shape: (mode, freq, obs, ws, yaw)
        vals = ds["delta_swl"].values

        print(f"Original Data Shape: {vals.shape}")
        # we reshape the values to (obs, mode, freq, ws, yaw)
        # for vectorized interpolation
        vals_transposed = vals.transpose(2, 0, 1, 3 ,4)
        print(f"Transposed Data Shape: {vals_transposed.shape}")

        # Flatten all non-spatial dimensions into one "feature vector" 
        # per observer. We do this by stacking the mode, freq, ws, yaw 
        # dimensions into a single dimension.
        n_obs = vals_transposed.shape[0]
        other_dims = vals_transposed.shape[1:] # (mode, freq, ws, yaw)
        n_features = np.prod(other_dims)
        
        vals_flat = vals_transposed.reshape(n_obs, n_features)
        print(f"Flattened Shape: {vals_flat.shape} (obs, features)")        

        # Spatial re-gridding by defining a regular grid
        # we choose these points to maintain a
        # 4 degree per step resolution in each direction. this can be adjusted
        # for finer or coarser interpolation
        n_grid_lat = 45
        n_grid_lon = 90
        
        # lat/lon gridding spaces for smooth interpolation
        grid_lat_ax = np.linspace(-90, 90, n_grid_lat)
        grid_lon_ax = np.linspace(-180, 180, n_grid_lon)

        print(f"   Grid Bounds -> Lat: [{grid_lat_ax[0]:.2f}, {grid_lat_ax[-1]:.2f}], Lon: [{grid_lon_ax[0]:.2f}, {grid_lon_ax[-1]:.2f}]")

        gridded_lat, gridded_lon = np.meshgrid(grid_lat_ax, grid_lon_ax, 
                                               indexing='ij')
        
        print(f"   Interpolating {n_obs} scattered points to {n_grid_lat}x{n_grid_lon} grid...")

        # Run griddata on the flattened features and get interpolated 
        # values on the expanded grid
        grid_vals_flat = griddata(points=(obs_lat_deg, obs_lon_deg),
                                  values=vals_flat,
                                  xi=(gridded_lat, gridded_lon),
                                  method='linear',
                                  fill_value=0
                                  )
        
        print(f"   Gridded Shape: {grid_vals_flat.shape} (lat, lon, features)")

        # Build the interpolator on the gridded data
        # but first we need to unflatten the features back to their original shape
        shape = (n_grid_lat, n_grid_lon) + other_dims
        grid_vals_ND = grid_vals_flat.reshape(shape)
        print(f"   Unflattened Gridded Shape: {grid_vals_ND.shape}: (lat, lon, mode, freq, ws, yaw)")
        print("\n Reshaping for GridInterpolator...")
        print(f"Final 6D Matrix Shape: {grid_vals_ND.shape}")

        # now we build the grid interpolator
        # NOTE: The order here defines the column order for query input!
        # Dim 0: Lat
        # Dim 1: Lon
        # Dim 2: Mode
        # Dim 3: Freq
        # Dim 4: WS
        # Dim 5: Yaw
        coords = [
            grid_lat_ax,    
            grid_lon_ax,    
            ds['mode'].values,   
            ds['freq'].values,   
            ds['ws'].values,     
            ds['yaw'].values     
        ]        

        interp = GridInterpolator(coords,
                                 grid_vals_ND,
                                 method='linear'
                                 )

        def get_directivity_generator(mode, freq, ws, yaw, lat_deg, lon_deg,
                                      batch_size=1000):

            # ============Process query points==================
            # input data
            freq_inp = np.atleast_1d(freq)
            ws_inp = np.atleast_1d(ws)
            yaw_inp = np.atleast_1d(yaw)
            mode_inp = np.atleast_1d(mode)
            lat_inp = np.atleast_1d(lat_deg)
            lon_inp = np.atleast_1d(lon_deg)
    
            # Create indices for every dimension
            idx_obs = np.arange(len(lat_inp))
            idx_mode = np.arange(len(mode_inp))
            idx_freq = np.arange(len(freq_inp))
            idx_ws = np.arange(len(ws_inp))
            idx_yaw = np.arange(len(yaw_inp))
    
            n_obs_inp = len(lat_inp)
            # Pre-allocate the full result array
            # Shape: (N_obs, M, F, W, Y)

            print(f"Processing {n_obs} observers in batches of {batch_size}...")
            
        #     ----batch processing to avoid memory issues---- 
            ############### BATCH PROCESSING START ###############
            for i in range(0, n_obs_inp, batch_size):
                # define the batch range
                i_end = min(i + batch_size, n_obs_inp)
                batch_size_actual = i_end - i
                print(f" Processing observers {i} to {i_end}... ", end='', flush=True)
                # slice the spatial data for this batch
                lat_batch = lat_inp[i:i_end]
                lon_batch = lon_inp[i:i_end]
                # create local indices for this batch
                idx_obs_batch = np.arange(batch_size_actual)
    
                # Meshgrid to create all combinations
                # We use indexing='ij' to keep dimensions ordered as inputs
                # Result shape: (N_obs, M, F, W, Y)
                I_obs, I_mode, I_freq, I_ws, I_yaw = np.meshgrid(
                    idx_obs_batch, idx_mode, idx_freq, idx_ws, idx_yaw, indexing='ij'
                )
    
                # Flatten everything to create the query list
                # Shape: (Total_Combinations, )
                flat_obs  = I_obs.ravel()
                flat_mode = I_mode.ravel()
                flat_freq = I_freq.ravel()
                flat_ws   = I_ws.ravel()
                flat_yaw  = I_yaw.ravel()
    
                # Map indices back to actual values
                # Note: Lat and Lon have same index as they are paired by n_obs
                q_lat = lat_batch[flat_obs]
                q_lon = lon_batch[flat_obs]
                q_mode = mode_inp[flat_mode]
                q_freq = freq_inp[flat_freq]
                q_ws = ws_inp[flat_ws]
                q_yaw = yaw_inp[flat_yaw]
    
                # Stack for the Interpolator (Must match coords order: Lat, Lon, 
                # Mode, Freq, WS, Yaw)
                query_inp = np.column_stack([q_lat, q_lon, q_mode, q_freq, q_ws, 
                                             q_yaw])
                
                # ==============interpolate=========================
                batch_flat_results = interp(query_inp)
                # reshape reslut to (batch_size, mode, freq, ws, yaw)
                batch_shape = (batch_size_actual, len(mode_inp), len(freq_inp),
                               len(ws_inp), len(yaw_inp))
                batch_results_reshaped = batch_flat_results.reshape(batch_shape)

                yield i, i_end, batch_results_reshaped

                # Explicit Garbage Collection after yield returns control
                del q_lat, q_lon, query_inp, batch_flat_results, batch_results_reshaped
                gc.collect()
                # Print Progress
                if (i // batch_size) % 2 == 0:
                    print(f"     -> Processed batch {i}-{i_end}...")
 
       
       
       
        n_obs_inp = len(lat_deg)
        result_shape = (n_obs_inp, len(mode), len(freq),
                        len(ws), len(yaw))
        print(f" Allocating Output Matrix: {result_shape}")
        final_result = np.zeros(result_shape)   

        # Initialize Generator
        batch_gen = get_directivity_generator(
            mode,
            freq,
            ws,
            yaw,
            lat_deg,
            lon_deg,
            batch_size=1000 # Process 1000 at a time
        )

        # Iterate and Save
        for start, end, data_chunk in batch_gen:

            # Store into final result array
            final_result[start:end, ...] = data_chunk
    

        ############### BATCH PROCESSING END ###############
    
    
                

        # print(ds)

        from matplotlib import pyplot as plt
        plt.plot(np.rad2deg(ds["lon"].values), np.rad2deg(ds["lat"].values),
                  'o', label='data obs')        
        plt.plot(lon_deg, lat_deg, 'x', label='target obs')
        plt.legend()
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.title("Observer locations in data")
        plt.savefig("./observer_locations.png", dpi=150)

        # exit()
        result = final_result.transpose(3, 0, 1, 4, 2)
        result = result[:, :, 0, 0, :]

        print(f"result.shape = {result.shape}")
        print(f"result[:, :, None, None, :]  = {result[:, :, None, None, :].shape}")
        # exit()

        return result[:, :, None, None, :]  # (n_cases, n_obs, 1, 1, n_freq)

    def new_spherical_interpolate_delta_swl(
        self,
        mode,
        freq,
        ws,
        yaw,
        lat_deg,
        lon_deg
    ):
        """
        Adapted version of the user-provided interpolation logic.
        Uses Re-gridding (Scatter -> Grid -> Interpolator) + Batched Generator.
        Optimized for speed and memory efficiency.
        """
        # observed data
        ds = self.ds
        
        # unpack the observed data
        # NOTE: Assuming ds stores Radians, converting to Degrees for griddata
        obs_lat_deg = np.rad2deg(ds["lat"].values)
        obs_lon_deg = np.rad2deg(ds["lon"].values)
        
        # delta_swl values
        # Shape: (mode, freq, obs, ws, yaw)
        vals = ds["delta_swl"].values

        print(f"Original Data Shape: {vals.shape}")
        
        # we reshape the values to (obs, mode, freq, ws, yaw)
        # for vectorized interpolation
        vals_transposed = vals.transpose(2, 0, 1, 3 ,4)
        print(f"Transposed Data Shape: {vals_transposed.shape}")

        # Flatten all non-spatial dimensions into one "feature vector" 
        # per observer. We do this by stacking the mode, freq, ws, yaw 
        # dimensions into a single dimension.
        n_obs = vals_transposed.shape[0]
        other_dims = vals_transposed.shape[1:] # (mode, freq, ws, yaw)
        n_features = np.prod(other_dims)
        
        vals_flat = vals_transposed.reshape(n_obs, n_features)
        print(f"Flattened Shape: {vals_flat.shape} (obs, features)")        

        # Spatial re-gridding by defining a regular grid
        # we choose these points to maintain a
        # 4 degree per step resolution in each direction.
        n_grid_lat = 45
        n_grid_lon = 90
        
        # lat/lon gridding spaces for smooth interpolation
        grid_lat_ax = np.linspace(-90, 90, n_grid_lat)
        grid_lon_ax = np.linspace(-180, 180, n_grid_lon)

        print(f"   Grid Bounds -> Lat: [{grid_lat_ax[0]:.2f}, {grid_lat_ax[-1]:.2f}], Lon: [{grid_lon_ax[0]:.2f}, {grid_lon_ax[-1]:.2f}]")

        gridded_lat, gridded_lon = np.meshgrid(grid_lat_ax, grid_lon_ax, indexing='ij')
        
        print(f"   Interpolating {n_obs} scattered points to {n_grid_lat}x{n_grid_lon} grid...")

        # Run griddata on the flattened features
        grid_vals_flat = griddata(
            points=(obs_lat_deg, obs_lon_deg),
            values=vals_flat,
            xi=(gridded_lat, gridded_lon),
            method='linear',
            fill_value=0
        )
        
        print(f"   Gridded Shape: {grid_vals_flat.shape} (lat, lon, features)")

        # Build the interpolator on the gridded data
        # unflatten the features back to their original shape
        shape = (n_grid_lat, n_grid_lon) + other_dims
        grid_vals_ND = grid_vals_flat.reshape(shape)
        print(f"   Unflattened Gridded Shape: {grid_vals_ND.shape}")
        
        # NOTE: The order here defines the column order for query input!
        # Dim 0: Lat, Dim 1: Lon, Dim 2: Mode, Dim 3: Freq, Dim 4: WS, Dim 5: Yaw
        coords = [
            grid_lat_ax,    
            grid_lon_ax,    
            ds['mode'].values,   
            ds['freq'].values,   
            ds['ws'].values,     
            ds['yaw'].values     
        ]        

        interp = GridInterpolator(coords, grid_vals_ND, method='linear')

        # --- GENERATOR DEFINITION (OPTIMIZED) ---
        def get_directivity_generator(mode, freq, ws, yaw, lat_deg, lon_deg, batch_size=1000):
            # input data
            freq_inp = np.atleast_1d(freq)
            ws_inp = np.atleast_1d(ws)
            yaw_inp = np.atleast_1d(yaw)
            mode_inp = np.atleast_1d(mode)
            lat_inp = np.atleast_1d(lat_deg)
            lon_inp = np.atleast_1d(lon_deg)
    
            n_obs_inp = len(lat_inp)

            print(f"Processing {n_obs_inp} observers in batches of {batch_size}...")

            # --- OPTIMIZATION: PRE-COMPUTE OPERATIONAL TEMPLATE ---
            # These dimensions (Mode, Freq, WS, Yaw) are identical for every observer.
            # We calculate their combinations ONCE here, avoiding massive meshgrids inside the loop.
            
            idx_mode = np.arange(len(mode_inp))
            idx_freq = np.arange(len(freq_inp))
            idx_ws = np.arange(len(ws_inp))
            idx_yaw = np.arange(len(yaw_inp))

            # Meshgrid for operational dims only
            I_mode, I_freq, I_ws, I_yaw = np.meshgrid(
                idx_mode, idx_freq, idx_ws, idx_yaw, indexing='ij'
            )
            
            # Create a reusable template of values
            # Shape: (N_Combinations, 4)
            ops_template = np.column_stack([
                mode_inp[I_mode.ravel()],
                freq_inp[I_freq.ravel()],
                ws_inp[I_ws.ravel()],
                yaw_inp[I_yaw.ravel()]
            ])
            
            n_ops_combos = ops_template.shape[0]
            # ------------------------------------------------------
            
            # --- BATCH PROCESSING START ---
            for i in range(0, n_obs_inp, batch_size):
                # define the batch range
                i_end = min(i + batch_size, n_obs_inp)
                batch_size_actual = i_end - i
                
                # slice the spatial data for this batch
                lat_batch = lat_inp[i:i_end]
                lon_batch = lon_inp[i:i_end]
                
                # --- OPTIMIZED QUERY CONSTRUCTION ---
                # Instead of running meshgrid, we Repeat Spatial coords and Tile Operational template.
                # This is much faster and uses significantly less memory.
                
                # 1. Repeat Spatial Coords: [Obs1, Obs1... (n_ops times), Obs2, Obs2...]
                q_lat = np.repeat(lat_batch, n_ops_combos)
                q_lon = np.repeat(lon_batch, n_ops_combos)
                
                # 2. Tile Operational Template: [Template, Template...]
                q_ops = np.tile(ops_template, (batch_size_actual, 1))
                
                # 3. Stack together
                query_inp = np.column_stack([q_lat, q_lon, q_ops])
                
                # Interpolate
                batch_flat_results = interp(query_inp)
                
                # Reshape
                batch_shape = (batch_size_actual, len(mode_inp), len(freq_inp), len(ws_inp), len(yaw_inp))
                batch_results_reshaped = batch_flat_results.reshape(batch_shape)

                yield i, i_end, batch_results_reshaped

                # Explicit Garbage Collection
                del q_lat, q_lon, q_ops, query_inp, batch_flat_results, batch_results_reshaped
                gc.collect()
                
                if (i // batch_size) % 2 == 0:
                    print(f"     -> Processed batch {i}-{i_end}...")

        # --- CONSUME GENERATOR ---
        n_obs_inp = len(lat_deg)
        # Expected generator output shape: (obs, mode, freq, ws, yaw)
        result_shape = (n_obs_inp, len(mode), len(freq), len(ws), len(yaw))
        
        print(f" Allocating Output Matrix: {result_shape}")
        final_result = np.zeros(result_shape)   

        # Initialize Generator
        batch_gen = get_directivity_generator(
            mode, freq, ws, yaw, lat_deg, lon_deg, batch_size=1000
        )

        # Iterate and Save
        for start, end, data_chunk in batch_gen:
            final_result[start:end, ...] = data_chunk
    
        # --- PLOTTING (From your code) ---
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(np.rad2deg(ds["lon"].values), np.rad2deg(ds["lat"].values), 'o', label='Data Obs')        
        plt.plot(lon_deg[:500], lat_deg[:500], 'x', label='Target Obs (Subset)') # Plot subset for speed
        plt.legend()
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.title("Observer locations")
        plt.grid(True)
        # Check if we can write to disk, otherwise skip saving
        try:
            plt.savefig("./observer_locations.png", dpi=100)
            print("Saved plot to ./observer_locations.png")
        except:
            pass
        plt.close()

        # --- FINAL RESHAPE ---
        # Current: (Obs, Mode, Freq, WS, Yaw) indices 0,1,2,3,4
        # Target:  (WS, Obs, Mode, Yaw, Freq) indices 3,0,1,4,2
        result = final_result.transpose(3, 0, 1, 4, 2)
        
        # Reducing Dimensions: Taking index 0 for Mode and Yaw
        # This results in (WS, Obs, Freq)
        result = result[:, :, 0, 0, :]

        print(f"Result shape after slicing: {result.shape}")

        # Expand back to (WS, Obs, 1, 1, Freq)
        final_5d = result[:, :, None, None, :]
        print(f"Final 5D shape: {final_5d.shape}")
        
        return final_5d


def main():
    if __name__ == "__main__":
        import matplotlib.pyplot as plt

        wt = TurbineNoise()

        # print(wt.ds)
        ws = np.arange(8, 14)
        for m in range(2):
            plt.plot(ws, wt.power(ws, mode=m) / 1000, label=f"mode: {m}")
        setup_plot(xlabel="Wind speed [m/s]", ylabel="Power [kW]")

        plt.figure()
        for m in range(2):
            freq, sound_power = wt.sound_power_level(ws=10, mode=m, yaw=20)
            # print(sound_power.shape)
            plt.semilogx(freq, sound_power.squeeze(), label=f"mode: {m}")
            setup_plot(xlabel="Frequency [Hz]", ylabel="Sound power [dB]")
        plt.show()

        # exit() 

        # Test spherical delta_swl interpolation
        mode = [0]
        freqs = wt.ds["freq"].values
        ws_arr = np.full_like(freqs, 10)
        yaws = np.full_like(freqs, 0)
        lat_deg = [-55.0, 56, 67]
        lon_deg = [8.0, -12, -23]
        delta_swl_interp = wt.spherical_interpolate_delta_swl(
            mode, freqs, ws_arr, yaws, lat_deg, lon_deg
        )
        print(f"Spherical interpolation result shape: {delta_swl_interp.shape}")

        # Multi-dimensional interpolation demo
        # cases = np.array(
        #     [
        #         [6.0, -20.0, 0],  # wind, yaw, mode
        #         [10.0, 0.0, 1],
        #         [16.0, 20.0, 2],
        #         [22.0, 10.0, 3],
        #     ]
        # )
        cases = np.array(
            [
                [10.0, -20.0, 0],  # wind, yaw, mode
                [11.0, 10.0, 1],
                [12.0, 0.0, 0],
            ]
        )

        obs_lats = np.array([45.0, -15.0, 60.0, 0.0, -60.0])
        obs_lons = np.array([-30.0, 120.0, 45.0, 0.0, -120.0])
        #freqs = np.array([63.0, 125.0, 500.0, 2000.0, 4000.0, 8000.0])
        freqs = np.array([125.0, 500.0, 2000.0, 4000.0])

        result = wt.spherical_interpolate_delta_swl(
            cases[:, 2],  # mode
            freqs,
            cases[:, 0],  # ws
            cases[:, 1],  # yaw
            obs_lats,
            obs_lons,
        )
        print(f"Multi-dimensional interpolation result shape: {result.shape}")


main()
