import json
import os
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
from numpy import newaxis as na
import xarray as xr
import matplotlib.pyplot as plt

import datetime as dt                   # Used to add today's date to the WTG file
import xml.etree.ElementTree as ET      # Used to write the wtg file

import uuid

from py_wake.deficit_models.deficit_model import WakeDeficitModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_farm_models.wind_farm_model import SimulationResult
from py_wake.superposition_models import SquaredSum
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
from typing import Any, Dict, Optional, Sequence, Tuple
from py_wake.wind_turbines import WindTurbines
from py_wake.site import Site, UniformSite

from osgeo import gdal
gdal.DontUseExceptions()

logger = logging.getLogger("__main__")
logger.setLevel('INFO')

# Keep a reference to the original implementation once
if not hasattr(FlowMap, "_orig_plot_wake_map"):
    FlowMap._orig_plot_wake_map = FlowMap.plot_wake_map


def _resolve_ti(TI, i, j):
    if TI is None:
        return float("nan")
    # scalar
    if getattr(TI, "ndim", 0) == 0:
        return float(TI)
    # 1D over wd
    if TI.ndim == 1:
        return float(TI[i])
    # 2D over wd,ws
    if TI.ndim == 2:
        return float(TI[i, j])
    raise ValueError("WakeBlaster: TI has unsupported dimensions; expected scalar, 1D(wd) or 2D(wd,ws).")


def _infer_ws_bin_edges(ws_vals: np.ndarray) -> np.ndarray:
    """
    Infer wind-speed bin edges from representative wind-speed values.

    The incoming `ws_vals` are treated as bin representatives (typically bin centers),
    and symmetric bin edges are reconstructed. This avoids truncating probability mass
    when converting Weibull distributions to discrete flow cases.
    """
    ws_vals = np.asarray(ws_vals, dtype=float)
    if ws_vals.ndim != 1 or len(ws_vals) == 0:
        raise ValueError("WakeBlaster: ws_vals must be a non-empty 1D array")
    if len(ws_vals) == 1:
        width = 1.0
        lo = max(0.0, ws_vals[0] - 0.5 * width)
        hi = ws_vals[0] + 0.5 * width
        return np.asarray([lo, hi], dtype=float)

    d = np.diff(ws_vals)
    if not np.all(d > 0):
        raise ValueError("WakeBlaster: ws_vals must be strictly increasing")

    edges = np.empty(len(ws_vals) + 1, dtype=float)
    edges[1:-1] = 0.5 * (ws_vals[:-1] + ws_vals[1:])
    edges[0] = max(0.0, ws_vals[0] - 0.5 * d[0])
    edges[-1] = ws_vals[-1] + 0.5 * d[-1]
    return edges


def custom_plot_wake_map(self, ax=None, cmap='Blues_r', colorbar=True, title=None, **kwargs):
    """
    Custom plot_wake_map method that loads the Surfer grid file stored in self.attrs['_grid_file']
    and plots it as an image.
    """
    # Only use the WakeBlaster image path if we actually have a grid file
    grid_file = self.attrs.get('_grid_file', None)
    if not grid_file:
        # Fallback: use the original PyWake plotting for NOJ/Bastankhah/etc.
        return FlowMap._orig_plot_wake_map(self, ax=ax, cmap=cmap, **kwargs)

    if ax is None:
        ax = plt.gca()

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

    # Respect caller’s title; only set if `title` is provided or none exists
    if title is not None:
        ax.set_title(title)
    elif not ax.get_title():
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


def _call_with_optional_type(func: Any, turbine_type: Optional[int] = None) -> Any:
    """Call a WindTurbines accessor with or without a type argument."""
    if turbine_type is None:
        return func()
    attempts = (
        lambda: func(turbine_type),
        lambda: func(type=turbine_type),
        lambda: func(types=turbine_type),
    )
    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return func()


def _call_curve_with_optional_type(func, ws, turbine_type):
    if turbine_type is None:
        return func(ws)

    attempts = (
        lambda: func(ws, turbine_type),
        lambda: func(ws, type=turbine_type),
        lambda: func(ws, types=turbine_type),
    )

    for attempt in attempts:
        try:
            return attempt()
        except TypeError:
            pass

    # fallback for single-type turbines
    return func(ws)


def _scalarize(value: Any, cast=float) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return cast(arr.item())
    if arr.size == 1:
        return cast(arr.reshape(-1)[0])
    return cast(arr.reshape(-1)[0])


def _canonical_h(h: Any) -> float:
    return round(float(h), 6)


def _format_h_for_name(h: Any) -> str:
    h_val = float(h)
    if float(h_val).is_integer():
        return f"{int(round(h_val))}m"
    return f"{str(h_val).replace('.', 'p')}m"


def _safe_design_basename(name: str) -> str:
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(name))
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_') or "WTG"


def _get_ws_tab_for_type(wind_turbines: Any, turbine_type: Optional[int] = None) -> np.ndarray:
    if turbine_type is None:
        pcf = getattr(wind_turbines, "powerCtFunction", None)
        if isinstance(pcf, PowerCtTabular):
            return np.asarray(pcf.ws_tab, dtype=float)

    for attr in ("powerCtFunctions", "powerCtFunction_lst", "powerCtFunctionList", "powerCtFunction"):
        funcs = getattr(wind_turbines, attr, None)
        if funcs is not None and turbine_type is not None:
            try:
                pcf = funcs.windTurbineFunction_lst[int(turbine_type)]
                if isinstance(pcf, PowerCtTabular):
                    return np.asarray(pcf.ws_tab, dtype=float)
            except Exception:
                pass

    pcf = getattr(wind_turbines, "powerCtFunction", None)
    if isinstance(pcf, PowerCtTabular):
        return np.asarray(pcf.ws_tab, dtype=float)

    raise TypeError("Expected PowerCtTabular instance for powerCtFunction")


def export_wtg_file(
    filename: str,
    wind_turbines: Any,
    turbine_type: Optional[int] = None,
    hub_height_override: Optional[float] = None,
    design_name: Optional[str] = None,
) -> None:
    """
    Export the turbine design in WAsP WTG file format.

    Parameters
    ----------
    filename : str
        File path for the WTG output.
    wind_turbines : Any
        Turbine object.
    turbine_type : int, optional
        Type index for mixed-type WindTurbines objects.
    hub_height_override : float, optional
        Explicit hub height to write into the WTG.
    design_name : str, optional
        Explicit description/name for the exported design.
    """
    raw_name = _call_with_optional_type(wind_turbines.name, turbine_type)
    turbine_name = str(_scalarize(raw_name, cast=str))
    if design_name is not None:
        turbine_name = str(design_name)

    hub_ht = (
        float(hub_height_override)
        if hub_height_override is not None
        else float(_scalarize(_call_with_optional_type(wind_turbines.hub_height, turbine_type)))
    )
    rotor_dia = float(_scalarize(_call_with_optional_type(wind_turbines.diameter, turbine_type)))

    ws_array = _get_ws_tab_for_type(wind_turbines, turbine_type)
    max_ws = float(ws_array.max())
    ws_range = np.arange(0, max_ws + 1, 1)

    power_crv = np.asarray(_call_curve_with_optional_type(wind_turbines.power, ws_range, turbine_type), dtype=float)
    ct_crv = np.asarray(_call_curve_with_optional_type(wind_turbines.ct, ws_range, turbine_type), dtype=float)

    _wtg_template = Path(__file__).resolve().parent.parent / "examples" / "data" / "WTG_Template.wtg"
    if not _wtg_template.exists():
        raise FileNotFoundError(f"WTG template not found at {_wtg_template}")
    tree = ET.parse(str(_wtg_template))
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
    logger.info("WTG file exported to %s", filename)


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
            logger.error(msg)
            raise RuntimeError(msg)
        except json.JSONDecodeError:
            msg = f"Configuration file '{filename}' is not valid JSON."
            logger.error(msg)
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
    wake superposition is handled intrinsically by WakeBlaster.
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
        unwaked_power_ilk: np.ndarray,
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
            unwaked_power_ilk=unwaked_power_ilk,
            **kwargs_ilk
        )
        self.windFarmModel = windFarmModel
        self.localWind = localWind

    def aep_ilk(self, normalize_probabilities=False, with_wake_loss=True):
        """
        Compute AEP for each turbine, wind direction and wind speed.

        Parameters
        ----------
        normalize_probabilities : bool, default False
            If True, normalize probabilities so they sum to 1 before computing AEP.
            This gives the AEP as if only the simulated flow cases existed for the
            full year.
        with_wake_loss : bool, default True
            If True, use the wake-affected power.
            If False, use the unwaked power based on free-flow wind speed, as calculated by Wakeblaster.

        Returns
        -------
        np.ndarray
            AEP for each turbine (i), wind direction (l) and wind speed (k) in GWh.
        """
        return self.aep(normalize_probabilities=normalize_probabilities, with_wake_loss=with_wake_loss).ilk()

    def aep(self, normalize_probabilities=False, with_wake_loss=True,
            hours_pr_year=365 * 24, linear_power_segments=False):
        """
        Compute Annual Energy Production (AEP) in GWh.

        Parameters
        ----------
        normalize_probabilities : bool, default False
            If True, normalize the flow-case probabilities before computing AEP.
            This gives the AEP as if the simulated flow cases represented a full year.
        with_wake_loss : bool, default True
            If True, use the wake-affected power.
            If False, use the unwaked power based on free-flow wind speed, as calculated by Wakeblaster.

        Returns
        -------
        xr.DataArray
            AEP of the wind farm in GWh. To get total AEP, call .sum() on the result.
        """
        if normalize_probabilities:
            norm = self.P.ilk().sum((1, 2))[:, na, na]
        else:
            norm = 1
        if with_wake_loss:
            power_ilk = self.Power.ilk()
        else:
            power_ilk = self.unwaked_power.ilk()

        if linear_power_segments:
            s = "The linear_power_segments method "
            assert all([n in self for n in ['Weibull_A', 'Weibull_k', 'Sector_frequency']]), \
                s + "requires a site with weibull information"
            assert normalize_probabilities is False, \
                s + "cannot be combined with normalize_probabilities"
            assert np.all(self.Power.isel(ws=0) == 0) and np.all(self.Power.isel(ws=-1) == 0), \
                s + "requires first wind speed to have no power (just below cut-in)"
            assert np.all(self.Power.isel(ws=-1) == 0), \
                s + "requires last wind speed to have no power (just above cut-out)"
            weighted_power = weibull.WeightedPower(
                self.ws.values,
                self.Power.ilk(),
                self.Weibull_A.ilk(),
                self.Weibull_k.ilk())
            aep = weighted_power * self.Sector_frequency.ilk() * hours_pr_year * 1e-9
            ws = (self.ws.values[1:] + self.ws.values[:-1]) / 2
            return xr.DataArray(aep, [('wt', self.wt.values), ('wd', self.wd.values), ('ws', ws)])
        else:
            weighted_power = power_ilk * self.P.ilk() / norm
        if 'time' in self.dims and weighted_power.shape[2] == 1:
            weighted_power = weighted_power[:, :, 0]

        return xr.DataArray(weighted_power * hours_pr_year * 1e-9,
                            self.Power.coords,
                            name='AEP [GWh]',
                            attrs={'Description': 'Annual energy production [GWh]'})

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

        # normalize list/array inputs to scalars
        wd = float(np.atleast_1d(wd)[0])
        ws = float(np.atleast_1d(ws)[0])

        # 1) Run simulation if no grid file provided
        if grid_file is None:
            grid_file = self.windFarmModel.calculate_flow_plane(ws, wd, height)
            logger.debug("Generated flow plane file: %s", grid_file)

        # 2) Load Surfer grid data
        grid_data = load_surfer_grid(grid_file)
        x_coords = np.array(grid_data["x"])        # (nx,)
        y_coords = np.array(grid_data["y"])        # (ny,)
        wake_loss = np.array(grid_data["wake_loss"])  # (ny, nx)

        # 3) Construct meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)      # (ny, nx)

        # 4) Compute effective wind speed and turbulence
        WS_eff_jlk = (ws * wake_loss)[..., np.newaxis]
        TI_da = self.windFarmModel.site.ds["TI"] if "TI" in self.windFarmModel.site.ds else None

        if TI_da is None:
            ambient_TI = 0.1
        else:
            # select TI for closest wind direction
            if "wd" in TI_da.dims:
                wd_vals = np.asarray(TI_da["wd"].values, dtype=float)
                wd_idx = int(np.argmin(np.abs(wd_vals - wd)))
                TI_da = TI_da.isel(wd=wd_idx)

            # if point-based site: take the first point
            if "i" in TI_da.dims:
                TI_da = TI_da.isel(i=0)

            ambient_TI = float(np.asarray(TI_da).squeeze())
            TI_eff_jlk = np.full_like(WS_eff_jlk, ambient_TI)

        # 5) Compute local wind at grid points
        num_pts = X.size
        Xf = X.ravel()
        Yf = Y.ravel()
        hf = np.full(num_pts, height, dtype=float)

        helper_site = UniformSite(p_wd=[1.0], ti=ambient_TI)

        localWind = helper_site.local_wind(
            x=Xf,
            y=Yf,
            h=hf,
            wd=np.array([wd], dtype=float),
            ws=np.array([ws], dtype=float)
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
        logger.info("Flow map generated for ws=%.1f, wd=%.1f", ws, wd)

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
        turbulenceModel: Optional[Any] = None,
        wtg_filename: Optional[str] = None,
        auto_export_wtg: bool = True,
        wtg_map: Optional[Dict[Any, str]] = None,
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

        # Non-breaking defaults:
        # - wtg_filename preserves the existing single-design path
        # - wtg_map enables mixed type / hub-height layouts
        self.wtg_filename = wtg_filename
        self.auto_export_wtg = auto_export_wtg
        self.wtg_map: Dict[Any, str] = dict(wtg_map or {})
        self._last_site_name: Optional[str] = None
        self._last_type_i: Optional[np.ndarray] = None
        self._last_h_i: Optional[np.ndarray] = None
        self._explicit_time_series: bool = False
        self._time_values: Optional[np.ndarray] = None

        set_client_id("PyWake")
        api_key = wakeblasterconfig.api['key']
        api_url = wakeblasterconfig.api['url']
        self.depends = et.WakeBlasterDependencies(api_key, api_url)

    def _normalize_type_i(self, n_wt: int, type: Optional[Sequence[int]]) -> np.ndarray:
        if type is None:
            return np.zeros(n_wt, dtype=int)
        type_i = np.asarray(type, dtype=int)
        if type_i.ndim == 0:
            type_i = np.full(n_wt, int(type_i), dtype=int)
        if len(type_i) != n_wt:
            raise ValueError(f"type must have length {n_wt}, got {len(type_i)}")
        return type_i

    def _default_h_i(self, type_i: np.ndarray) -> np.ndarray:
        try:
            hh = self.windTurbines.hub_height(type_i)
        except TypeError:
            try:
                hh = self.windTurbines.hub_height(type=type_i)
            except TypeError:
                hh = self.windTurbines.hub_height()
        hh_arr = np.asarray(hh, dtype=float)
        if hh_arr.ndim == 0:
            hh_arr = np.full(len(type_i), float(hh_arr), dtype=float)
        elif len(hh_arr) != len(type_i):
            hh_arr = np.asarray([float(np.asarray(v).reshape(-1)[0]) for v in hh_arr], dtype=float)
            if len(hh_arr) != len(type_i):
                raise ValueError("Unable to resolve hub heights per turbine from wind_turbines")
        return hh_arr

    def _normalize_h_i(
        self,
        n_wt: int,
        type_i: np.ndarray,
        h: Optional[Sequence[float]],
    ) -> np.ndarray:
        if h is None:
            h_i = self._default_h_i(type_i)
        else:
            h_i = np.asarray(h, dtype=float)
            if h_i.ndim == 0:
                h_i = np.full(n_wt, float(h_i), dtype=float)
            if len(h_i) != n_wt:
                raise ValueError(f"h must have length {n_wt}, got {len(h_i)}")
        return h_i.astype(float)

    def _find_wtg_map_entry(self, turbine_type: int, hub_height: float) -> Optional[str]:
        if not self.wtg_map:
            return None
        key_candidates = [
            (int(turbine_type), _canonical_h(hub_height)),
            (int(turbine_type), float(hub_height)),
            (int(turbine_type), int(round(float(hub_height)))) if float(hub_height).is_integer() else None,
            int(turbine_type),
            str(int(turbine_type)),
            (_canonical_h(hub_height), int(turbine_type)),
        ]
        for key in key_candidates:
            if key is not None and key in self.wtg_map:
                return self.wtg_map[key]
        return None

    def _default_design_id_and_filename(self, turbine_type: int, hub_height: float) -> Tuple[str, str]:
        raw_name = _call_with_optional_type(self.windTurbines.name, int(turbine_type))
        turbine_name = str(_scalarize(raw_name, cast=str))
        base = _safe_design_basename(f"{turbine_name}_{_format_h_for_name(hub_height)}")
        return base, f"{base}.wtg"

    def _ensure_design_file(
        self,
        turbine_type: int,
        hub_height: float,
        unique_pairs: Sequence[Tuple[int, float]],
    ) -> Tuple[str, str]:
        mapped = self._find_wtg_map_entry(turbine_type, hub_height)
        if mapped is not None:
            wtg_filename = mapped
            design_id = Path(wtg_filename).stem
            if not os.path.exists(wtg_filename):
                raise FileNotFoundError(f"Configured WTG file not found: {wtg_filename}")
            logger.info("Using mapped WTG file %s for type=%s, h=%s", wtg_filename, turbine_type, hub_height)
            return design_id, wtg_filename

        if self.wtg_filename is not None:
            if len(unique_pairs) > 1:
                raise ValueError(
                    "Single wtg_filename cannot represent multiple turbine designs. "
                    "Pass wtg_map for mixed type/h layouts or omit wtg_filename to auto-export."
                )
            wtg_filename = self.wtg_filename
            design_id = Path(wtg_filename).stem
            if not os.path.exists(wtg_filename):
                raise FileNotFoundError(f"Configured WTG file not found: {wtg_filename}")
            logger.info("Using provided WTG file %s", wtg_filename)
            return design_id, wtg_filename

        design_id, wtg_filename = self._default_design_id_and_filename(turbine_type, hub_height)
        if os.path.exists(wtg_filename):
            logger.info("WTG file %s found.", wtg_filename)
            return design_id, wtg_filename

        if not self.auto_export_wtg:
            raise FileNotFoundError(
                f"WTG file {wtg_filename} not found and auto_export_wtg is False."
            )

        logger.info("WTG file %s not found; exporting...", wtg_filename)
        export_wtg_file(
            wtg_filename,
            self.windTurbines,
            turbine_type=int(turbine_type),
            hub_height_override=float(hub_height),
            design_name=design_id,
        )
        return design_id, wtg_filename

    def _prepare_designs(self, type_i: np.ndarray, h_i: np.ndarray) -> np.ndarray:
        pair_seq = [(int(t), _canonical_h(h)) for t, h in zip(type_i, h_i)]
        unique_pairs = list(dict.fromkeys(pair_seq))
        pair_to_design: Dict[Tuple[int, float], str] = {}
        loaded_designs = set()

        for turbine_type, hub_height in unique_pairs:
            design_id, wtg_filename = self._ensure_design_file(turbine_type, hub_height, unique_pairs)
            pair_to_design[(turbine_type, hub_height)] = design_id
            if design_id not in loaded_designs:
                self.depends.load_turbine_design(design_id, wtg_filename)
                loaded_designs.add(design_id)

        return np.asarray([pair_to_design[p] for p in pair_seq], dtype=object)

    def _site_ti_value(self, wd_val: float, ws_val: Optional[float] = None) -> float:
        ds = self.site.ds
        if "TI" not in ds.variables:
            return float("nan")
        TI_da = ds["TI"]
        if "i" in TI_da.dims:
            TI_da = TI_da.isel(i=0)
        if "wd" in TI_da.dims:
            wd_coords = np.asarray(TI_da["wd"].values, dtype=float)
            wd_idx = int(np.argmin(np.abs(wd_coords - float(wd_val))))
            TI_da = TI_da.isel(wd=wd_idx)
        if ws_val is not None and "ws" in TI_da.dims:
            ws_coords = np.asarray(TI_da["ws"].values, dtype=float)
            ws_idx = int(np.argmin(np.abs(ws_coords - float(ws_val))))
            TI_da = TI_da.isel(ws=ws_idx)
        return float(np.asarray(TI_da).squeeze())

    def __call__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        h: Optional[Sequence[float]] = None,
        type: Optional[Sequence[int]] = None,
        wd: Optional[Sequence[float]] = None,
        ws: Optional[Sequence[float]] = None,
        time: Any = False,
        TI: Optional[Sequence[float]] = None,
        L: Optional[Sequence[float]] = None,
        return_simulationResult: bool = True,
        simulation_id: Optional[str] = None
    ) -> Optional[SimulationResult]:
        """
        Run farm-level simulation and return SimulationResult if requested.

        Parameters
        ----------
        x : array_like
            Turbine eastings in meters.
        y : array_like
            Turbine northings in meters. Must have the same length as x.
        h : float or array_like, optional
            Hub height per turbine. If None, hub heights are derived from wind_turbines
            using the provided type values.
        type : int or array_like, optional
            Turbine type index per turbine. Scalar values are broadcast to all turbines.
            If None, type 0 is used for all turbines.
        wd : array_like, optional
            Wind directions in degrees. If omitted together with ws and time=False,
            flow cases are derived from the site climate.
        ws : array_like, optional
            Wind speeds in m/s. If omitted together with wd and time=False,
            flow cases are derived from the site climate.
        time : bool or array_like
            If False (default), wd and ws form a Cartesian grid (standard AEP mode).
            If True, wd and ws are paired samples with implicit timestamps 0,1,...,n.
            If array_like, same as True but the array is used as the time coordinate.
        TI : scalar or array_like, optional
            Turbulence intensity, scalar or matching wd/ws.
        L : scalar or array_like, optional
            Monin-Obukhov length for stability, scalar or matching wd/ws.
        return_simulationResult : bool, default True
            If True, parse and return a WakeBlasterResult. If False, run the simulation
            without constructing a SimulationResult object.
        simulation_id : str, optional
            Simulation ID of existing simulation to retrieve results from instead of starting new simulation.
            Flow cases from this simulation must match those that would be generated from the provided input parameters.

        Returns
        -------
        SimulationResult or None
            Parsed simulation result when return_simulationResult=True, otherwise None.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")

        self.num_wt = len(x)
        type_i = self._normalize_type_i(self.num_wt, type)
        h_i = self._normalize_h_i(self.num_wt, type_i, h)
        self._last_type_i = type_i
        self._last_h_i = h_i
        self._last_x = np.asarray(x, dtype=float)
        self._last_y = np.asarray(y, dtype=float)

        # Normalize time to PyWake convention: False | True | array_like
        if time is False:
            self._explicit_time_series = False
            self._time_values = None
        elif time is True:
            self._explicit_time_series = True
            self._time_values = None  # will use np.arange later
        else:
            # array_like
            self._explicit_time_series = True
            self._time_values = np.asarray(np.atleast_1d(time))

        self.flow_cases = self.create_flowcases(wd=wd, ws=ws, time=time, TI=TI, L=L)

        # derive dimensions from actual flow_cases (robust across sites)
        self.num_wd = int(self.flow_cases["WindDirection"].nunique())
        self.num_ws = int(self.flow_cases["WindSpeed_avg"].nunique())

        run_id = uuid.uuid4().hex[:12]
        site_name = f"PyWake_farm_{run_id}"
        self._last_site_name = site_name

        design_ids = self._prepare_designs(type_i, h_i)

        layout = pd.DataFrame({
            "Easting": x,
            "Northing": y,
            "ID": range(self.num_wt),
            "DesignID": design_ids
        })
        turbines = [
            {
                "id": str(row.ID),
                "easting": float(row.Easting),
                "northing": float(row.Northing),
                "design_id": str(row.DesignID),
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
                if key == 'resource':
                    # RSF file: read as plain text
                    with open(inputs[key], 'r') as fp:
                        wind_resource_grid = fp.read()
                else:
                    # JSON inputs
                    with open(inputs[key], 'r') as fp:
                        val = json.load(fp)
                    if key == 'associated_WS':
                        associated_WS = val
                    elif key == 'regional_roughness':
                        regional_roughness = val

        self.depends.set_farm(
            site_name,
            et.WindFarm(
                farm_def,
                wind_resource_grid=wind_resource_grid,
                associated_WS=associated_WS,
                regional_roughness=regional_roughness
            )
        )

        # Need to make sure that the right stability model is used when L is provided
        # Build per-run config to avoid mutating the base configuration.
        sim_cfg = deepcopy(self.wakeblasterconfig.simulation_config)
        if L is not None:
            wake_model_cfg = sim_cfg.setdefault("wake_model", {})
            wake_model_cfg["type"] = "3D"
            eddy_viscosity_cfg = wake_model_cfg.setdefault("eddy_viscosity", {})
            eddy_viscosity_cfg["type"] = "CombinedProfile"
            eddy_viscosity_cfg["stability_model"] = "BusingerDyer"

        if simulation_id is None:
            sim = et.WakeBlasterNewSimulation(
                site_name,
                "0",
                self.flow_cases,
                sim_cfg
            )
            logger.info("Simulation started (ID: %s)", sim.id)
            self.wait_for_results(sim.id)
            _, turbine_res = sim.get_results(
                [],
                ["power", "turbulence_intensity", "hub_wind_speed", "unwaked_power"]
            )
        else:
            logger.info("Retrieving results for existing simulation ID: %s", simulation_id)
            sim = et.WakeBlasterSimulation(simulation_id, self.flow_cases)

            if sim.are_results_ready():
                _, turbine_res = sim.get_results(
                    [],
                    ["power", "turbulence_intensity", "hub_wind_speed", "unwaked_power"]
                )
            else:
                logger.warning("Results for simulation ID %s are not ready yet.", simulation_id)
                return None

        if return_simulationResult:
            return self.parse_results(turbine_res, x, y, type_i=type_i, h_i=h_i)
        return None

    def create_flowcases(
        self,
        wd: Optional[Sequence[float]] = None,
        ws: Optional[Sequence[float]] = None,
        time: Any = False,
        TI: Optional[Sequence[float]] = None,
        L: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """Create flow cases for the WakeBlaster simulation.

        Parameters
        ----------
        wd : array_like, optional
            Wind directions in degrees. If None, derived from site climate.
        ws : array_like, optional
            Wind speeds in m/s. If None, derived from site climate.
        time : bool or array_like, default False
            False: wd/ws form a Cartesian grid (or fall back to site climate).
            True: wd/ws are paired samples with implicit timestamps 0, 1, ..., n-1, matching flow case ids.
            array_like: wd/ws are paired samples with explicit timestamps.
        TI : float or array_like, optional
            Turbulence intensity. Scalar (applied to all cases) or array matching wd/ws.
        L : float or array_like, optional
            Monin-Obukhov length for atmospheric stability. Scalar or array matching wd/ws.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for wind direction, wind speed, probability/hours,
            and optionally TI and L for each flow case.
        """
        ds = self.site.ds
        records = []

        # Determine whether we are in time-series (paired) mode
        _is_time_series = time is not False  # True or array_like

        # Explicit cases from user-supplied wd/ws (grid or paired)
        if wd is not None or ws is not None or _is_time_series:
            if wd is None or ws is None:
                raise ValueError("When specifying explicit cases, both wd and ws must be provided.")

            wd_vals = np.asarray(np.atleast_1d(wd), dtype=float)
            ws_vals = np.asarray(np.atleast_1d(ws), dtype=float)
            l_records = []  # store L values separately if provided, to be added as a column later

            if _is_time_series:
                # time=True or time=array_like: wd[i] paired with ws[i]
                if time is not True:
                    time_vals = np.atleast_1d(time)
                    if not (len(wd_vals) == len(ws_vals) == len(time_vals)):
                        raise ValueError("For time-series mode, wd, ws and time must have the same length.")
                else:
                    # time=True: implicit timestamps, just validate wd/ws lengths match
                    if len(wd_vals) != len(ws_vals):
                        raise ValueError("For time-series mode (time=True), wd and ws must have the same length.")

                ti_vals = None if TI is None else np.asarray(np.atleast_1d(TI), dtype=float)
                if ti_vals is not None and ti_vals.size not in (1, len(wd_vals)):
                    raise ValueError("TI must be scalar or have the same length as wd/ws in time-series mode.")

                l_vals = None if L is None else np.asarray(np.atleast_1d(L), dtype=float)
                if l_vals is not None and l_vals.size not in (1, len(wd_vals)):
                    raise ValueError("L must be scalar or have the same length as wd/ws in time-series mode.")
                weight = 1.0 / len(wd_vals)  # Equal weighting for time-series cases
                for idx, (wd_val, ws_val) in enumerate(zip(wd_vals, ws_vals)):
                    if ti_vals is None:
                        ti_val = self._site_ti_value(wd_val, ws_val)
                    elif ti_vals.size == 1:
                        ti_val = float(ti_vals[0])
                    else:
                        ti_val = float(ti_vals[idx])
                    if l_vals is not None:
                        l_val = float(l_vals[0]) if l_vals.size == 1 else float(l_vals[idx])
                    if not np.isfinite(ti_val) or ti_val <= 0:
                        ti_val = 0.1
                    records.append([float(ws_val), float(wd_val), weight, float(ti_val)])
                    if l_vals is not None:
                        l_records.append(l_val)
            else:
                # time=False: Create flow cases as cartesian product of wd and ws.
                # Weights are derived from the site climate, matching PyWake's P_ilk behaviour
                ti_vals = None if TI is None else np.asarray(np.atleast_1d(TI), dtype=float)
                if ti_vals is not None and ti_vals.size not in (1, len(wd_vals), len(ws_vals), len(wd_vals) * len(ws_vals)):
                    raise ValueError(
                        "TI must be scalar, len(wd), len(ws), or len(wd)*len(ws) for explicit wd/ws mode."
                    )
                l_vals = np.asarray(np.atleast_1d(L), dtype=float) if L is not None else None
                if l_vals is not None and l_vals.size not in (1, len(wd_vals), len(ws_vals), len(wd_vals) * len(ws_vals)):
                    raise ValueError(
                        "L must be scalar, len(wd), len(ws), or len(wd)*len(ws) for explicit wd/ws mode."
                    )

                # Query site probability at all turbine positions, matching PyWake's
                # calc_wt_interaction which calls site.local_wind with every turbine's
                # (x, y, h).  P_ilk has shape (n_wt, n_wd, n_ws); we average over
                # turbines to get a single (n_wd, n_ws) weight matrix per flow case.
                # For uniform sites this is identical to a single-point query.
                _x = self._last_x if hasattr(self, '_last_x') else np.array([0.0])
                _y = self._last_y if hasattr(self, '_last_y') else np.array([0.0])
                _h = self._last_h_i if hasattr(self, '_last_h_i') else np.array([70.0])
                try:
                    lw_prob = self.site.local_wind(
                        x=_x, y=_y, h=_h, wd=wd_vals, ws=ws_vals
                    )
                    # P_ilk shape: (n_wt, n_wd, n_ws) -> mean over turbines -> (n_wd, n_ws)
                    P_lk = np.asarray(lw_prob.P_ilk, dtype=float).mean(axis=0)
                except Exception:
                    # Fallback: uniform weights if site query fails
                    P_lk = np.ones((len(wd_vals), len(ws_vals)), dtype=float)
                    P_lk /= P_lk.sum()

                flat_ti_idx = 0
                flat_l_idx = 0
                for i, wd_val in enumerate(wd_vals):
                    for j, ws_val in enumerate(ws_vals):
                        if ti_vals is None:
                            ti_val = self._site_ti_value(wd_val, ws_val)
                        elif ti_vals.size == 1:
                            ti_val = float(ti_vals[0])
                        elif ti_vals.size == len(wd_vals):
                            ti_val = float(ti_vals[i])
                        elif ti_vals.size == len(ws_vals):
                            ti_val = float(ti_vals[j])
                        else:
                            ti_val = float(ti_vals[flat_ti_idx])
                        if l_vals is not None:
                            if l_vals.size == 1:
                                l_val = float(l_vals[0])
                            elif l_vals.size == len(wd_vals):
                                l_val = float(l_vals[i])
                            elif l_vals.size == len(ws_vals):
                                l_val = float(l_vals[j])
                            else:
                                l_val = float(l_vals[flat_l_idx])
                        flat_ti_idx += 1
                        flat_l_idx += 1
                        if not np.isfinite(ti_val) or ti_val <= 0:
                            ti_val = 0.1
                        records.append([float(ws_val), float(wd_val), float(P_lk[i, j]), float(ti_val)])
                        if l_vals is not None:
                            l_records.append(l_val)

            flow_cases = pd.DataFrame(records, columns=["WindSpeed_avg", "WindDirection", "weighting", "TI"])
            if len(l_records):
                flow_cases["MoninObukhovLength"] = l_records
            flow_cases["WindSpeed_std"] = flow_cases["TI"] * flow_cases["WindSpeed_avg"]
            flow_cases = flow_cases.drop(columns=["TI"]).reset_index(drop=True)
            flow_cases["flow_case_id"] = [
                f"{idx:05d}WS{row.WindSpeed_avg}dir{row.WindDirection}"
                for idx, row in flow_cases.iterrows()
            ]
            return flow_cases

        # Default climate-driven cases from site
        wd_vals = ds.coords["wd"].values
        drop_last_wd = len(wd_vals) > 1 and (wd_vals[-1] == 360 or wd_vals[-1] == wd_vals[0] + 360)
        wd_idx = slice(None, -1) if drop_last_wd else slice(None)

        ws_vals = ds["ws"].values

        has_weibull = all(k in ds.variables for k in ["Sector_frequency", "Weibull_A", "Weibull_k"])
        has_prob = "P" in ds.variables

        if has_weibull:
            sf = ds["Sector_frequency"].isel(wd=wd_idx)
            A = ds["Weibull_A"].isel(wd=wd_idx)
            k = ds["Weibull_k"].isel(wd=wd_idx)

            # point-based RSF: choose one representative point
            if "i" in sf.dims:
                sf = sf.isel(i=0)
            if "i" in A.dims:
                A = A.isel(i=0)
            if "i" in k.dims:
                k = k.isel(i=0)

            TI_da = ds["TI"] if "TI" in ds.variables else None
            if TI_da is not None:
                if "wd" in TI_da.dims:
                    TI_da = TI_da.isel(wd=wd_idx)
                if "i" in TI_da.dims:
                    TI_da = TI_da.isel(i=0)
                TI_arr = TI_da.values
            else:
                TI_arr = None

            sf = np.asarray(sf.values, dtype=float)
            A = np.asarray(A.values, dtype=float)
            k = np.asarray(k.values, dtype=float)

            ws_edges = _infer_ws_bin_edges(ws_vals)

            for i, wd_val in enumerate(wd_vals[wd_idx]):
                g = []
                for j in range(len(ws_vals)):
                    lo = ws_edges[j]
                    hi = ws_edges[j + 1]
                    gij = weibull.cdf(hi, A[i], k[i]) - weibull.cdf(lo, A[i], k[i])
                    g.append(float(gij))

                for j, ws_val in enumerate(ws_vals):
                    p = float(sf[i]) * float(g[j])
                    weighting = float(p)
                    ti_val = _resolve_ti(TI_arr, i, j)
                    if not np.isfinite(ti_val) or ti_val <= 0:
                        ti_val = 0.1
                    records.append([float(ws_val), float(wd_val), weighting, float(ti_val)])
        elif has_prob:
            P = ds["P"].values
            TI_arr = ds["TI"].values if "TI" in ds.variables else None

            if P.ndim == 2:
                P2 = P[wd_idx, :]
                for i, wd_val in enumerate(wd_vals[wd_idx]):
                    for j, ws_val in enumerate(ws_vals):
                        weighting = float(P2[i, j])
                        ti_val = _resolve_ti(TI_arr, i, j)
                        if not np.isfinite(ti_val) or ti_val <= 0:
                            ti_val = 0.1
                        records.append([float(ws_val), float(wd_val), weighting, float(ti_val)])

            # If P is 1D over wd (happens in UniformSite with only one wind speed), expand to ws
            elif P.ndim == 1 and P.shape[0] == len(wd_vals):
                P2 = P[wd_idx]
                for i, wd in enumerate(wd_vals[wd_idx]):
                    for j, ws in enumerate(ws_vals):
                        weighting = float(P2[i])
                        ti_val = _resolve_ti(TI_arr, i, j)
                        records.append([float(ws), float(wd), weighting, float(ti_val)])

            else:
                raise ValueError("WakeBlaster create_flowcases: 'P' must be 2D over ('wd','ws') or 1D over 'wd' for this site.")
        else:
            raise ValueError(
                "WakeBlaster create_flowcases: Unsupported site dataset. "
                "Expected either Weibull fields (Sector_frequency, Weibull_A, Weibull_k) "
                "or probability table field 'P' with coords wd/ws."
            )

        flow_cases = pd.DataFrame(records, columns=["WindSpeed_avg", "WindDirection", "weighting", "TI"])
        if flow_cases["TI"].isna().all():
            flow_cases["TI"] = 0.1
        flow_cases["WindSpeed_std"] = flow_cases["TI"] * flow_cases["WindSpeed_avg"]
        flow_cases = flow_cases.drop(columns=["TI"]).reset_index(drop=True)
        flow_cases["flow_case_id"] = [
            f"{idx:05d}WS{row.WindSpeed_avg}dir{row.WindDirection}"
            for idx, row in flow_cases.iterrows()
        ]

        weighting_sum = float(flow_cases["weighting"].sum())
        logger.debug("WakeBlaster flow case weighting sum = %.6f", weighting_sum)
        if not np.isclose(weighting_sum, 1.0, rtol=0.0, atol=1e-3):
            logger.debug(
                "WakeBlaster: flow case weighting sum differs from 1.0 "
                "(%.6f vs %.6f )",
                weighting_sum,
                float(1.0),
            )
        return flow_cases

    def wait_for_results(self, sim_id: str) -> None:
        """
        Poll simulation until completion or timeout (12h).
        """
        elapsed = 0
        interval = 30
        while elapsed < 12 * 3600:
            info = get_simulation_info(sim_id)
            state = info.get('state')
            if state == 'complete':
                logger.info("Simulation %s complete", sim_id)
                return
            if state in ('error', 'complete-with-errors'):
                logger.error("Simulation %s error state: %s", sim_id, state)
                raise RuntimeError(f"Simulation {sim_id} failed with state {state}")
            time.sleep(interval)
            elapsed += interval
        raise RuntimeError(f"Simulation {sim_id} did not complete in time.")

    def parse_results(
        self,
        turbine_res: Dict[str, pd.DataFrame],
        x: Sequence[float],
        y: Sequence[float],
        type_i: Optional[Sequence[int]] = None,
        h_i: Optional[Sequence[float]] = None,
    ) -> SimulationResult:
        """
        Parse WakeBlaster API outputs into a WakeBlasterResult.

        The parser supports two result interpretations:

        1) Grid mode (wd x ws):
           Used when no explicit time series was requested and no duplicate
           (WindDirection, WindSpeed_avg) pairs exist in flow cases.

        2) Time-series mode (flow-case axis):
           Used when explicit time mode was requested (time=True or array_like),
           or when duplicate (WindDirection, WindSpeed_avg) pairs are detected.
           In the duplicate case, parsing is auto-switched to time-series mode
           to avoid ambiguous reshaping onto a unique wd/ws grid.

        Parameters
        ----------
        turbine_res : dict[str, pd.DataFrame]
            Raw result tables returned by WakeBlaster.
        x : array_like
            Turbine eastings used to reconstruct local wind coordinates.
        y : array_like
            Turbine northings used to reconstruct local wind coordinates.
        type_i : array_like, optional
            Turbine type index per turbine. If None, uses the cached value from
            the latest model call.
        h_i : array_like, optional
            Hub height per turbine. If None, uses the cached value from the
            latest model call (or model defaults if not cached).

        Returns
        -------
        SimulationResult
            WakeBlasterResult containing simulation results.
            Dimensions are wt,wd,ws in grid mode and wt,time in time-series mode.
        """
        num_wt = self.num_wt
        num_wd = self.num_wd
        num_ws = self.num_ws

        # Determine whether results must be interpreted as a true time/case series
        # (explicit time input or duplicate wd/ws flow cases).
        flow_cases = self.flow_cases.copy().reset_index(drop=True)
        has_duplicate_wd_ws = flow_cases.duplicated(subset=["WindDirection", "WindSpeed_avg"]).any()
        use_time_series = bool(self._explicit_time_series or has_duplicate_wd_ws)
        if has_duplicate_wd_ws and not self._explicit_time_series:
            msg = (
                "WakeBlaster: duplicate wd/ws flow cases detected without explicit time axis; "
                "switching result parsing to time-series mode."
            )
            logger.warning(msg)

        # Keep flow-case ordering stable by aligning result tables on flow_case_id when available.
        ordered_ids = flow_cases["flow_case_id"].tolist() if "flow_case_id" in flow_cases.columns else None

        def _prepare_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
            tbl = df.copy()
            drop_cols = [c for c in ["index"] if c in tbl.columns]
            if drop_cols:
                tbl = tbl.drop(columns=drop_cols)
            if ordered_ids is not None and "flow_case_id" in tbl.columns:
                tbl = tbl.set_index("flow_case_id").reindex(ordered_ids)
                if tbl.isna().all(axis=1).any():
                    raise ValueError(f"WakeBlaster parse_results: missing {label} rows after flow_case_id alignment")
                tbl = tbl.reset_index(drop=True)
            elif "flow_case_id" in tbl.columns:
                tbl = tbl.drop(columns=["flow_case_id"])
            try:
                tbl.columns = tbl.columns.astype(int)
            except Exception as e:
                logger.error("An error occurred during conversion of the columns of %s: %s", label, e)
            tbl = tbl.reindex(sorted(tbl.columns), axis=1)
            return tbl

        pwr = _prepare_table(turbine_res['power'], 'power')
        unwaked_pwr = _prepare_table(turbine_res['unwaked_power'], 'unwaked_power')
        hub_wind_speed = _prepare_table(turbine_res['hub_wind_speed'], 'hub_wind_speed')
        ti = _prepare_table(turbine_res['turbulence_intensity'], 'turbulence_intensity')

        if use_time_series:
            # Time-series mode: keep results in flow-case order, allowing duplicates
            n_case = len(flow_cases)
            if len(pwr) != n_case:
                raise ValueError(
                    f"WakeBlaster parse_results: expected {n_case} result rows in time-series mode, got {len(pwr)}"
                )

            # One row per flow case -> transpose to turbine-major and keep a singleton ws-axis.
            power_ilk = np.asarray(pwr, dtype=float).T[:, :, np.newaxis]
            unwaked_power_ilk = np.asarray(unwaked_pwr, dtype=float).T[:, :, np.newaxis]
            WS_eff_ilk = np.asarray(hub_wind_speed, dtype=float).T[:, :, np.newaxis]
            TI_eff_ilk = np.asarray(ti, dtype=float).T[:, :, np.newaxis]

            weighting = np.asarray(flow_cases['weighting'], dtype=float)
            P_ilk = np.tile(weighting[np.newaxis, :, np.newaxis], (num_wt, 1, 1))
            ct_ilk = np.zeros((num_wt, n_case, 1))  # The WakeBlaster API currently does not return Ct

            wd_array = np.asarray(flow_cases['WindDirection'], dtype=float)
            ws_array = np.asarray(flow_cases['WindSpeed_avg'], dtype=float)

            if self._explicit_time_series and self._time_values is not None and len(self._time_values) == n_case:
                time_array = np.asarray(self._time_values)
            else:
                # For implicit time series (or mismatched explicit time), use flow case index.
                time_array = np.arange(n_case)

            ti_input_ilk = None
            if "WindSpeed_std" in flow_cases.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ti_flow = np.asarray(flow_cases["WindSpeed_std"] / flow_cases["WindSpeed_avg"], dtype=float)
                ti_flow = np.nan_to_num(ti_flow, nan=0.1, posinf=0.1, neginf=0.1)
                ti_input_ilk = ti_flow[np.newaxis, :, np.newaxis]

            L_input_ilk = None
            if "MoninObukhovLength" in flow_cases.columns:
                L_flow = np.asarray(flow_cases["MoninObukhovLength"], dtype=float)
                if np.isfinite(L_flow).any():
                    L_input_ilk = L_flow[np.newaxis, :, np.newaxis]
        else:
            # Grid mode: reshape results onto (wt, wd, ws), with unique wd/ws pairs.
            P_ilk = flow_cases['weighting'].values.reshape(num_wd, num_ws)
            P_ilk = np.tile(P_ilk[np.newaxis, :, :], (num_wt, 1, 1))

            power_ilk = np.array(pwr).reshape(num_wd, num_ws, num_wt).transpose(2, 0, 1)
            unwaked_power_ilk = np.array(unwaked_pwr).reshape(num_wd, num_ws, num_wt).transpose(2, 0, 1)
            WS_eff_ilk = np.array(hub_wind_speed).reshape(num_wd, num_ws, num_wt).transpose(2, 0, 1)
            TI_eff_ilk = np.array(ti).reshape(num_wd, num_ws, num_wt).transpose(2, 0, 1)

            ct_ilk = np.zeros((num_wt, num_wd, num_ws))
            wd_array = np.sort(flow_cases["WindDirection"].unique())
            ws_array = np.sort(flow_cases["WindSpeed_avg"].unique())
            if "WindSpeed_std" in flow_cases.columns:
                ti_array = np.sort((flow_cases["WindSpeed_std"] / flow_cases["WindSpeed_avg"]).unique())
            else:
                ti_array = None

            # Preserve flow-case TI/L in the result dataset on the (wd, ws) grid.
            ti_input_ilk = None
            if "WindSpeed_std" in flow_cases.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ti_flow = flow_cases["WindSpeed_std"] / flow_cases["WindSpeed_avg"]
                ti_grid = (
                    flow_cases.assign(TI=ti_flow)
                    .pivot_table(index="WindDirection", columns="WindSpeed_avg", values="TI", aggfunc="first")
                    .reindex(index=wd_array, columns=ws_array)
                )
                ti_input = np.asarray(ti_grid.values, dtype=float)
                if np.isfinite(ti_input).any():
                    if ti_array is not None and len(ti_array) > 0:
                        fallback_ti = float(np.asarray(ti_array, dtype=float).reshape(-1)[0])
                    else:
                        fallback_ti = 0.1
                    ti_input = np.nan_to_num(ti_input, nan=fallback_ti, posinf=fallback_ti, neginf=fallback_ti)
                    ti_input_ilk = ti_input[np.newaxis, :, :]

            L_input_ilk = None
            if "MoninObukhovLength" in flow_cases.columns:
                L_grid = (
                    flow_cases
                    .pivot_table(index="WindDirection", columns="WindSpeed_avg", values="MoninObukhovLength", aggfunc="first")
                    .reindex(index=wd_array, columns=ws_array)
                )
                L_input = np.asarray(L_grid.values, dtype=float)
                if np.isfinite(L_input).any():
                    L_input_ilk = L_input[np.newaxis, :, :]

        if type_i is None:
            type_i = self._last_type_i if self._last_type_i is not None else np.zeros(num_wt, dtype=int)
        type_i = np.asarray(type_i, dtype=int)

        if h_i is None:
            h_i = self._last_h_i if self._last_h_i is not None else self._default_h_i(type_i)
        h_i = np.asarray(h_i, dtype=float)

        if use_time_series:
            if ti_input_ilk is None:
                local_ti = None
            else:
                local_ti = np.asarray(ti_input_ilk[0, :, 0], dtype=float)
            local_wind = self.site.local_wind(
                x=x,
                y=y,
                h=h_i,
                wd=wd_array,
                ws=ws_array,
                time=time_array,
                TI=local_ti
            )
        else:
            local_wind = self.site.local_wind(
                x=x, y=y,
                h=h_i,
                wd=wd_array,
                ws=ws_array,
                TI=ti_array
            )

        result_kwargs: Dict[str, Any] = {"P_ilk": P_ilk}
        if ti_input_ilk is not None:
            result_kwargs["TI_ilk"] = ti_input_ilk
        if L_input_ilk is not None:
            result_kwargs["L_ilk"] = L_input_ilk

        result = WakeBlasterResult(
            self,
            localWind=local_wind,
            type_i=type_i,
            WS_eff_ilk=WS_eff_ilk,
            TI_eff_ilk=TI_eff_ilk,
            power_ilk=power_ilk,
            unwaked_power_ilk=unwaked_power_ilk,
            ct_ilk=ct_ilk,
            **result_kwargs
        )

        # Add descriptions to result variables
        result["Power"].attrs["Description"] = "Turbine power output [W]"
        result["WS_eff"].attrs["Description"] = "Effective wind speed at hub height [m/s]"
        result["TI_eff"].attrs["Description"] = "Effective turbulence intensity"
        result["CT"].attrs["Description"] = "Thrust coefficient"
        result["unwaked_power"].attrs["Description"] = "Turbine power output without wake interaction [W]"
        if "TI" in result:
            result["TI"].attrs["Description"] = "Ambient input turbulence intensity"
        if "L" in result:
            result["L"].attrs["Description"] = "Monin-Obukhov length [m]"
        if "time" in result.coords:
            if self._explicit_time_series:
                result.coords["time"].attrs["Description"] = "Explicit time values for each flow case"
            else:
                result.coords["time"].attrs["Description"] = "Flow case index (auto-generated time axis due to duplicate wd/ws cases)"

        return result

    def calculate_flow_plane(
        self,
        wind_speed: float,
        wind_direction: float,
        height: float
    ) -> str:
        """
        Trigger a single flow case simulation to generate a flow plane log for the specified
        wind speed, wind direction, and height.
        """
        flow_case = pd.DataFrame({
            "WindSpeed_avg": [wind_speed],
            "WindDirection": [wind_direction],
            "weighting": [365 * 24],
        })

        config: Any = self.wakeblasterconfig.simulation_config.copy()
        config["flow_plane_logger"] = {
            "enabled": True,
            "grid_height": height,
            "single_height": True,
        }

        logger.debug(
            "Flow plane simulation config: %s\nFlow case:\n%s",
            config,
            flow_case.to_dict(orient="records")
        )

        site_name = self._last_site_name or "PyWake_farm"
        simulation = et.WakeBlasterNewSimulation(
            site_name,
            "0",
            flow_case,
            config
        )
        logger.info(
            "Started flow plane simulation (ID: %s) for wind_speed=%s, wind_direction=%s",
            simulation.id,
            wind_speed,
            wind_direction
        )

        self.wait_for_results(simulation.id)
        return self.download_flow_plane(simulation.id, height=height)

    def download_flow_plane(
        self,
        sim_id: str,
        height: float
    ) -> str:
        """
        Download the Surfer grid file for a completed flow plane simulation.
        """
        file_path = f"flow_plane_{sim_id}_h{height}.grd"
        logger.info("Downloading flow plane grid file: %s", file_path)

        try:
            wb.download_surfer_grid(sim_id, height, file_path)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.exception(
                "Failed to download grid file for simulation '%s' at height %.2f",
                sim_id,
                height
            )
            raise RuntimeError(
                f"Download failed for simulation '{sim_id}' at height {height}"
            ) from exc

        return file_path
