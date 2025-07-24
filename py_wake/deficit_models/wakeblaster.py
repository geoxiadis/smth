import json
import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from osgeo import gdal
import datetime as dt                   # Used to add today's date to the WTG file
import xml.etree.ElementTree as ET      # Used to write the wtg file


from py_wake.deficit_models.deficit_model import WakeDeficitModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_farm_models.wind_farm_model import SimulationResult
from py_wake.superposition_models import SquaredSum
from py_wake.utils.gradients import cabs
from py_wake.deficit_models.utils import ct2a_madsen
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.utils import weibull

from py_wake.flow_map import FlowMap

from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import wakeblastersdk as wb
from wakeblastersdk.sdk import set_client_id, get_simulation_info
import energytoolbox as et
import logging
import time
import h5py
from typing import Any, Dict, Optional, Sequence
from py_wake.wind_turbines import WindTurbines
from py_wake.site import Site


logger = logging.getLogger("__main__")
logger.setLevel('INFO')

def custom_plot_wake_map(self, ax=None, cmap='Blues_r', colorbar=True, **kwargs):
    """
    Custom plot_wake_map method that loads the Surfer grid file stored in self.attrs['_grid_file']
    and plots it as an image.
    """
    if ax is None:
        ax = plt.gca()
    
    # Retrieve the grid file path from attributes
    grid_file = self.attrs.get('_grid_file', None)
    if grid_file is None:
        ax.text(0.5, 0.5, "No grid file available", transform=ax.transAxes,
                ha='center', va='center')
        return ax

    ds = gdal.Open(grid_file)
    if ds is None:
        ax.text(0.5, 0.5, f"Cannot open {grid_file}", transform=ax.transAxes,
                ha='center', va='center')
        return ax
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    # Replace the no-data placeholder with 1 and clip values so the maximum is 1.
    placeholder = 1.71041e38
    data = np.where(data == placeholder, 1, data)
    data = np.clip(data, None, 1)

    # Get geotransform and compute coordinates:
    gt = ds.GetGeoTransform()  # (originX, pixelWidth, 0, originY, 0, pixelHeight)
    nx = ds.RasterXSize
    ny = ds.RasterYSize
    x0, dx, _, y0, _, dy = gt
    x = np.linspace(x0, x0 + (nx - 1) * dx, nx)
    y = np.linspace(y0, y0 + (ny - 1) * dy, ny)

    img = ax.imshow(data, extent=(x[0], x[-1], y[-1], y[0]),
                    origin='upper', cmap=cmap, vmin=data.min(), vmax=1)
    if colorbar:
        plt.colorbar(img, ax=ax)
    ax.set_title("Flow Plane")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    return ax

# Override the default plot_wake_map method in FlowMap
FlowMap.plot_wake_map = custom_plot_wake_map


def load_surfer_grid(filename: str) -> dict:
    """
    Load a Surfer grid file and return a dictionary containing the grid data.

    This example uses GDAL to read an ASCII or binary Surfer grid file.
    It extracts:
      - x: a list of x coordinates (1D)
      - y: a list of y coordinates (1D)
      - wake_loss: a 2D numpy array of wake loss values

    Parameters
    ----------
    filename : str
        Path to the Surfer grid file.
    
    Returns
    -------
    dict
        Dictionary with keys "x", "y", and "wake_loss".
    """
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError("GDAL is required to load Surfer grid files. Please install it.")

    ds = gdal.Open(filename)
    if ds is None:
        raise RuntimeError(f"Cannot open grid file {filename}.")

    # Read the first band (assuming wake loss values are in band 1)
    band = ds.GetRasterBand(1)
    wake_loss = band.ReadAsArray()

    # Extract geotransform to compute x and y coordinates.
    gt = ds.GetGeoTransform()  # (originX, pixelWidth, 0, originY, 0, pixelHeight)
    nx = ds.RasterXSize
    ny = ds.RasterYSize
    # Create 1D lists for x and y coordinates:
    x = [gt[0] + i * gt[1] for i in range(nx)]
    y = [gt[3] + j * gt[5] for j in range(ny)]
    return {"x": x, "y": y, "wake_loss": wake_loss}




def export_wtg_file(
    filename: str,
    wind_turbines: Any
) -> None:
    """
    Export the turbine design in WAsP WTG file format.

    The output includes turbine parameters and
    a power and thrust coefficient curve.

    Parameters
    ----------
    filename : str
        File path for the WTG output.
    wind_turbines : Any
        Turbine object with methods:
          - name()
          - hub_height()
          - diameter()
          - power(ws: np.ndarray) -> np.ndarray
          - ct(ws: np.ndarray) -> np.ndarray
          - powerCtFunction.ws_tab attribute

    Raises
    ------
    TypeError
        If the turbine's powerCtFunction is not a PowerCtTabular instance.
    """
    turbine_name = wind_turbines.name()
    hub_ht = wind_turbines.hub_height()
    rotor_dia = wind_turbines.diameter()

    pcf = wind_turbines.powerCtFunction
    if not isinstance(pcf, PowerCtTabular):
        raise TypeError("Expected PowerCtTabular instance for powerCtFunction")

    ws_array = np.array(pcf.ws_tab)
    max_ws = float(ws_array.max())
    ws_range = np.arange(0, max_ws + 1, 1)

    power_crv = wind_turbines.power(ws_range)
    ct_crv = wind_turbines.ct(ws_range)

    tree = ET.parse("WTG_Template.wtg")
    root = tree.getroot()

    root.attrib.update({
        "Description": turbine_name,
        "RotorDiameter": str(rotor_dia),
    })
    root[0].text = (
        f"{turbine_name} created by PyWake/WakeBlaster integration."
    )
    root[1][0].text = str(hub_ht)
    root[2].attrib["ReleaseDate"] = dt.date.today().isoformat()

    df = pd.DataFrame({
        "Windspeed": ws_range,
        "PowerOutput": power_crv,
        "CtCurve": ct_crv,
    })
    df = df[df["Windspeed"] >= 3]

    data_parent = root[2][2]
    for _, row in df.iterrows():
        ET.SubElement(
            data_parent,
            "DataPoint",
            {
                "WindSpeed": f"{row['Windspeed']:.1f}",
                "PowerOutput": f"{row['PowerOutput']:.1f}",
                "ThrustCoEfficient": f"{row['CtCurve']:.3f}"
            }
        )

    last_ws = df["Windspeed"].iloc[-1]
    cut_node = root[2][0]
    cut_node.attrib["HighSpeedCutIn"] = str(last_ws)
    cut_node.attrib["HighSpeedCutOut"] = str(last_ws)

    tree.write(filename, encoding="UTF-8", xml_declaration=True)
    logging.info("WTG file exported to %s", filename)



class WakeBlasterConfig:
    """
    Configuration holder for WakeBlaster API credentials,
    simulation settings, and input file paths.
    """
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        simulation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.api: Dict[str, str] = {
            "key": api_key or "",
            "url": api_url or ""
        }
        self.simulation_config: Dict[str, Any] = simulation_config or {}
        self.input: Dict[str, str] = {}

    def from_file(self, filename: str) -> None:
        """
        Load configuration settings from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON config file.

        Raises
        ------
        RuntimeError
            If the file is missing or contains invalid JSON.
        """
        try:
            with open(filename, 'r') as fp:
                settings = json.load(fp)
        except FileNotFoundError:
            msg = f"Configuration file '{filename}' not found."
            logging.error(msg)
            raise RuntimeError(msg)
        except json.JSONDecodeError:
            msg = f"Configuration file '{filename}' is not valid JSON."
            logging.error(msg)
            raise RuntimeError(msg)

        self.simulation_config = settings.get(
            "simulation_config", self.simulation_config
        )
        self.api = settings.get("api", self.api)
        self.input.update(settings.get("input", {}))

    def validate(self) -> None:
        """
        Validate that API credentials and input files are correctly set.

        Raises
        ------
        ValueError
            If API key or URL are missing.
        FileNotFoundError
            If any input file path does not exist.
        """
        if not self.api.get("key") or not self.api.get("url"):
            raise ValueError(
                "API key and URL must be provided in the configuration."
            )
        for name, path in self.input.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Input file '{path}' for '{name}' not found."
                )

    def __str__(self) -> str:
        return (
            f"API: {self.api}\n"
            f"Sim Config: {self.simulation_config}\n"
            f"Inputs: {self.input}"
        )


class WakeBlasterDeficit(WakeDeficitModel):
    """
    Deficit model for WakeBlaster: returns zero deficit as
    farm-level wake effects are handled externally by WakeBlaster.
    """
    def calc_deficit(
        self,
        WS_ilk: np.ndarray,
        WS_eff_ilk: np.ndarray,
        D_src_il: np.ndarray,
        D_dst_ijl: np.ndarray,
        dw_ijlk: np.ndarray,
        cw_ijlk: np.ndarray,
        ct_ilk: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Return zero deficit array matching the wake diameter dimensions.
        """
        return np.zeros_like(dw_ijlk)

    def wake_radius(
        self,
        D_src_il: np.ndarray,
        dw_ijlk: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Return zero wake radius array as wake extent is external.
        """
        return np.zeros_like(dw_ijlk)


class WakeBlasterResult(SimulationResult):
    """
    SimulationResult subclass for handling WakeBlaster simulation outputs,
    including AEP calculation and flow map generation.
    """
    __slots__ = ()  # Prevent dynamic attributes and suppress FutureWarning

    def __init__(
        self,
        windFarmModel: Any,
        localWind: Any,
        type_i: Any,
        WS_eff_ilk: np.ndarray,
        TI_eff_ilk: np.ndarray,
        power_ilk: np.ndarray,
        ct_ilk: np.ndarray,
        **kwargs_ilk: Any
    ) -> None:
        super().__init__(
            windFarmModel=windFarmModel,
            localWind=localWind,
            type_i=type_i,
            WS_eff_ilk=WS_eff_ilk,
            TI_eff_ilk=TI_eff_ilk,
            power_ilk=power_ilk,
            ct_ilk=ct_ilk,
            **kwargs_ilk
        )
        self.windFarmModel = windFarmModel
        self.localWind = localWind

    def aep(self):
        """
        Compute Annual Energy Production (AEP) in GWh.

        Returns
        -------
        float
            Total AEP of the wind farm in GWh.
        """
        # power_ilk in W, P_ilk in hours -> Wh, then to GWh
        aep_ilk = self.power_ilk * self.P_ilk * 1e-9
        total_aep = aep_ilk.sum()
        logging.info("Computed AEP: %.3f GWh", total_aep)
        return total_aep

    def flow_map(
        self,
        wd: float,
        ws: float,
        grid_file: Optional[str] = None,
        height: float = 70.0,
        **kwargs: Any
    ) -> FlowMap:
        """
        Generate a flow map for given wind direction and speed.

        Parameters
        ----------
        wd : float
            Wind direction for the flow map.
        ws : float
            Wind speed for the flow map.
        grid_file : Optional[str], default None
            Path to an existing Surfer grid file. If None, triggers simulation.
        height : float, default 70.0
            Height at which to compute the flow plane.

        Returns
        -------
        FlowMap
            Object containing grid and wake-loss data for plotting.
        """
        # 1) Run simulation if no grid file provided
        if grid_file is None:
            grid_file = self.windFarmModel.calculate_flow_plane(ws, wd, height)
            logging.debug("Generated flow plane file: %s", grid_file)

        # 2) Load Surfer grid data
        grid_data = load_surfer_grid(grid_file)
        x_coords = np.array(grid_data["x"])        # (nx,)
        y_coords = np.array(grid_data["y"])        # (ny,)
        wake_loss = np.array(grid_data["wake_loss"])  # (ny, nx)

        # 3) Construct meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)      # (ny, nx)

        # 4) Compute effective wind speed and turbulence
        WS_eff_jlk = (ws * wake_loss)[..., np.newaxis]
        ambient_TI = (
            float(self.windFarmModel.site.ds["TI"].values)
            if "TI" in self.windFarmModel.site.ds
            else 0.1
        )
        TI_eff_jlk = np.full_like(WS_eff_jlk, ambient_TI)

        # 5) Compute local wind at grid points
        num_pts = X.size
        Xf = X.ravel()
        Yf = Y.ravel()
        hf = np.full(num_pts, height)
        localWind = self.windFarmModel.site.local_wind(
            x=Xf, y=Yf, h=hf,
            wd=np.array([wd]), ws=np.array([ws])
        )

        # 6) Preserve turbine coordinates in result
        self["x"] = ("wt", np.array(self.localWind.x.data))
        self["y"] = ("wt", np.array(self.localWind.y.data))

        # 7) Create FlowMap
        plane = ["XY"]
        fm = FlowMap(
            self, X, Y, localWind,
            WS_eff_jlk, TI_eff_jlk,
            plane=plane,
            **kwargs
        )

        # 8) Store grid file path for custom plotting
        fm.attrs["_grid_file"] = grid_file
        logging.info("Flow map generated for ws=%.1f, wd=%.1f", ws, wd)

        return fm


class WakeBlaster(PropagateDownwind):
    """
    Main interface for WakeBlaster integration within PyWake.
    Submits farm layout and flow cases to WakeBlaster, retrieves results,
    and parses them into SimulationResult.
    """
    def __init__(
        self,
        site: Site,
        wind_turbines: WindTurbines,
        wakeblasterconfig: WakeBlasterConfig,
        rotorAvgModel: Any = AreaOverlapAvgModel(),
        ct2a: Any = ct2a_madsen,
        superpositionModel: Any = SquaredSum(),
        deflectionModel: Optional[Any] = None,
        turbulenceModel: Optional[Any] = None
    ) -> None:
        super().__init__(
            site,
            wind_turbines,
            wake_deficitModel=WakeBlasterDeficit(),
            rotorAvgModel=rotorAvgModel,
            superpositionModel=superpositionModel,
            deflectionModel=deflectionModel,
            turbulenceModel=turbulenceModel
        )
        if not wakeblasterconfig:
            raise ValueError("WakeBlasterConfig must be provided.")
        self.wakeblasterconfig = wakeblasterconfig
        self.num_wd = len(site.ds.wd) - 1
        self.num_ws = len(site.ds.ws)

        set_client_id("PyWake")
        api_key = wakeblasterconfig.api['key']
        api_url = wakeblasterconfig.api['url']
        self.depends = et.WakeBlasterDependencies(api_key, api_url)

    def __call__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        return_simulationResult: bool = True
    ) -> Optional[SimulationResult]:
        """
        Run farm-level simulation and return SimulationResult if requested.
        """
        self.num_wt = len(x)
        self.flow_cases = self.create_flowcases()

        design_id = f"{self.windTurbines.name()}_{self.windTurbines.hub_height()}"
        wtg_filename = f"{design_id}.wtg"

        if not os.path.exists(wtg_filename):
            logger.info("WTG file %s not found; exporting...", wtg_filename)
            export_wtg_file(wtg_filename, self.windTurbines)
        else:
            logger.info("WTG file %s found.", wtg_filename)

        layout = pd.DataFrame({
            "Easting": x,
            "Northing": y,
            "ID": range(self.num_wt),
            "DesignID": design_id
        })
        turbines = [
            {   "id": str(row.ID),
                "easting": float(row.Easting),
                "northing": float(row.Northing),
                "design_id": row.DesignID,
                "elevation": float(self.site.elevation(row.Easting, row.Northing))
            } for _, row in layout.iterrows()
        ]
        farm_def = {"turbine_instances": turbines}

        associated_WS = None
        regional_roughness = None
        wind_resource_grid = None
        inputs = self.wakeblasterconfig.input
        for key in ('associated_WS', 'regional_roughness', 'resource'):
            if key in inputs:
                with open(inputs[key], 'r') as fp:
                    val = json.load(fp)
                if key == 'associated_WS': associated_WS = val
                elif key == 'regional_roughness': regional_roughness = val
                else: wind_resource_grid = val

        self.depends.load_turbine_design(design_id, wtg_filename)
        self.depends.set_farm(
            "PyWake_farm",
            et.WindFarm(
                farm_def,
                wind_resource_grid=wind_resource_grid,
                associated_WS=associated_WS,
                regional_roughness=regional_roughness
            )
        )

        sim = et.WakeBlasterNewSimulation(
            "PyWake_farm",
            "0",
            self.flow_cases,
            self.wakeblasterconfig.simulation_config
        )
        logger.info(f"Simulation started (ID: {sim.id})" )
        self.wait_for_results(sim.id)
        _, turbine_res = sim.get_results(
            [],
            ["power", "turbulence_intensity", "hub_wind_speed"]
        )

        if return_simulationResult:
            return self.parse_results(turbine_res, x, y)
        return None


    def create_flowcases(self) -> pd.DataFrame:
        """
        Generate flow cases DataFrame from site wind statistics.
        """
        records = []
        ds = self.site.ds
        for i, (wd,freq) in enumerate(zip(ds.coords["wd"].values[:-1],
                              ds["Sector_frequency"].values[:-1])):
            WS_POI = ds["ws"].values
            g = []
            for si, _ in enumerate(WS_POI):
                if si > 0:
                    g.append(
                        weibull.cdf(WS_POI[si], ds["Weibull_A"][i].values, ds["Weibull_k"][i].values) -
                        weibull.cdf(WS_POI[si-1], ds["Weibull_A"][i].values, ds["Weibull_k"][i].values)
                    )
                else:
                    g.append(
                        weibull.cdf(WS_POI[si], ds["Weibull_A"][i].values, ds["Weibull_k"][i].values) -
                        weibull.cdf(0, ds["Weibull_A"][i].values, ds["Weibull_k"][i].values)
                    )

            for cnt, ws in enumerate(WS_POI):
                records.append([ws, wd, freq * g[cnt] * 8766])
        flow_cases = pd.DataFrame(records, columns=["WindSpeed_avg", "WindDirection", "weighting"])
        flow_cases['WindSpeed_std'] = ds['TI'].values * flow_cases['WindSpeed_avg']
        flow_cases.reset_index(drop=True, inplace=True)
        logger.debug("Flow cases shape: %s", flow_cases.shape)
        return flow_cases


    def wait_for_results(self, sim_id: str) -> None:
        """
        Poll simulation until completion or timeout (12h).
        """
        elapsed = 0
        interval = 30
        while elapsed < 12*3600:
            info = get_simulation_info(sim_id)
            state = info.get('state')
            if state == 'complete':
                logger.info("Simulation %s complete", sim_id)
                return
            if state in ('error','complete-with-errors'):
                logger.error("Simulation %s error state: %s", sim_id, state)
                raise RuntimeError(f"Simulation {sim_id} failed with state {state}")
            time.sleep(interval)
            elapsed += interval
        raise RuntimeError(f"Simulation {sim_id} did not complete in time.")



    def parse_results(
        self,
        turbine_res: Dict[str, Any],
        x: Sequence[float],
        y: Sequence[float]
    ) -> SimulationResult:
        """
        Parse WakeBlaster API results into a SimulationResult.

        Parameters
        ----------
        turbine_res : dict
            Dictionary of turbine-level results.
        x : sequence of float
            Turbine easting coordinates.
        y : sequence of float
            Turbine northing coordinates.

        Returns
        -------
        SimulationResult
            Parsed results including WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, and P_ilk.
        """
        num_wt = len(x)
        num_wd = len(self.site.ds.wd)-1
        num_ws = len(self.site.ds.ws)

        flow_cases = self.flow_cases.copy()
        flow_cases = flow_cases.rename(columns={
            'WindSpeed_avg':'ws','WindDirection':'wd','weighting':'P'
        })
        
        
        pwr = turbine_res['power'].copy()

        pwr['ws'] = self.flow_cases['WindSpeed_avg'].where(pwr['flow_case_id'] == self.flow_cases['flow_case_id'])
        pwr['wd'] = self.flow_cases['WindDirection'].where(pwr['flow_case_id'] == self.flow_cases['flow_case_id'])
        pwr['P'] = self.flow_cases['weighting'].where(pwr['flow_case_id'] == self.flow_cases['flow_case_id'])
        for df in [pwr]:
            del df['index']
            del df['flow_case_id']
        P_ilk = self.flow_cases['weighting'].values.reshape(self.num_wd, self.num_ws)
        P_ilk = np.tile(P_ilk[np.newaxis, :, :], (num_wt, 1, 1))
        
        pwr = pwr.drop(columns=['ws', 'wd', 'P'])
        try:
            pwr.columns = pwr.columns.astype(int)
        except Exception as e:
            logger.error("An error occurred during conversion of the columns to int:", e)
        pwr = pwr.reindex(sorted(pwr.columns), axis=1)
        power_ilk  = np.array(pwr).reshape(num_wd, num_ws, num_wt).transpose(2, 0, 1)

        hub_wind_speed = turbine_res['hub_wind_speed']
        for col in ['index', 'flow_case_id']:
            if col in hub_wind_speed.columns:
                hub_wind_speed = hub_wind_speed.drop(columns=[col])
        try:
            hub_wind_speed.columns = hub_wind_speed.columns.astype(int)
        except Exception as e:
             logger.error("An error occurred during conversion of the columns of hub_wind_speed:", e)
        hub_wind_speed = hub_wind_speed.reindex(sorted(hub_wind_speed.columns), axis=1)
        WS_eff_ilk = np.array(hub_wind_speed).reshape(num_wd, num_ws, num_wt).transpose(2, 0, 1)
        
        # Modify turbulence_intensity (TI)
        ti = turbine_res['turbulence_intensity']
        for col in ['index', 'flow_case_id']:
            if col in ti.columns:
                ti = ti.drop(columns=[col])
        try:
            ti.columns = ti.columns.astype(int)
        except Exception as e:
            logger.error("An error occurred during conversion of the columns of turbulence_intensity:", e)
        ti = ti.reindex(sorted(ti.columns), axis=1)
        TI_eff_ilk = np.array(ti).reshape(num_wd, num_ws, num_wt).transpose(2, 0, 1)

        
        
        ct_ilk = np.zeros((num_wt, num_wd, num_ws))
        ds = self.site.ds
        wd_array = ds.coords["wd"].values[:-1]
        ws_array = ds.coords["ws"].values

        local_wind = self.site.local_wind(
            x=x, y=y,
            h=[self.windTurbines.hub_height()]*num_wt,
            wd=self.site.ds.coords['wd'].values[:-1],
            ws=self.site.ds.coords['ws'].values
        )

        return WakeBlasterResult(
            self,
            localWind=local_wind,
            type_i=self.windTurbines.types(),
            WS_eff_ilk=WS_eff_ilk,
            TI_eff_ilk=TI_eff_ilk,
            power_ilk=power_ilk,
            ct_ilk=ct_ilk,
            P_ilk=P_ilk
        )


    def calculate_flow_plane(
        self,
        wind_speed: float,
        wind_direction: float,
        height: float
    ) -> str:
        """
        Trigger a single flow case simulation to generate a flow plane log for the specified
        wind speed, wind direction, and height.
    
        Parameters
        ----------
        wind_speed : float
            Wind speed for the simulation.
        wind_direction : float
            Wind direction for the simulation.
        height : float
            Height at which to log the flow plane.
    
        Returns
        -------
        str
            Path to the downloaded Surfer grid file containing the flow plane output.
        """
        # Prepare single-case DataFrame
        flow_case = pd.DataFrame({
            "WindSpeed_avg": [wind_speed],
            "WindDirection": [wind_direction],
            "weighting": [8766.0],  # Full-year equivalent hours
        })
    
        # Configure flow plane logging
        config: Any = self.wakeblasterconfig.simulation_config.copy()
        config["flow_plane_logger"] = {
            "enabled": True,
            "grid_height": height,
            "single_height": True,
        }
    
        logging.debug(
            "Flow plane simulation config: %s\nFlow case:\n%s",
            config,
            flow_case.to_dict(orient="records")
        )
    
        # Start simulation
        simulation = et.WakeBlasterNewSimulation(
            "PyWake_farm",
            "0",
            flow_case,
            config
        )
        logging.info(
            "Started flow plane simulation (ID: %s) for wind_speed=%s, wind_direction=%s",
            simulation.id,
            wind_speed,
            wind_direction
        )
    
        # Await results
        self.wait_for_results(simulation.id)
    
        # Download and return grid file path
        return self.download_flow_plane(simulation.id, height=height)

    def download_flow_plane(
        self,
        sim_id: str,
        height: float
    ) -> str:
        """
        Download the Surfer grid file for a completed flow plane simulation.
    
        Parameters
        ----------
        sim_id : str
            Identifier of the completed simulation.
        height : float
            Height at which the flow plane was calculated.
    
        Returns
        -------
        str
            Local file path to the downloaded Surfer grid file.
    
        Raises
        ------
        RuntimeError
            If the download fails.
        """
        file_path = f"flow_plane_{sim_id}_h{height}.grd"
        logging.info("Downloading flow plane grid file: %s", file_path)
    
        try:
            # Use the WakeBlaster SDK client to fetch the file
            wb.download_surfer_grid(sim_id, height, file_path)  # type: ignore[attr-defined]
        except Exception as exc:
            logging.exception(
                "Failed to download grid file for simulation '%s' at height %.2f",
                sim_id,
                height
            )
            raise RuntimeError(
                f"Download failed for simulation '{sim_id}' at height {height}"
            ) from exc
    
        return file_path

