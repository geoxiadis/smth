import os

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import xarray as xr
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap
from zoneinfo import ZoneInfoNotFoundError

from py_wake.shadow_models.solar_position import solar_position


class ShadowBase:
    """Base class for shadow calculation results"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.mode = dataset.attrs["mode"]
        self.freq = dataset.attrs["freq"]

    def sel(self, **kwargs):
        """
        Select a subset of the data using coordinate indexing.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to xarray.Dataset.sel()

        Returns
        -------
        ShadowBase
            New instance with selected data
        """
        dataset = self.dataset.sel(**kwargs)
        dataset = dataset.expand_dims(
            [dim for dim in self.dataset.dims if dim not in dataset.dims]
        )
        return self.__class__(dataset)

    def isel(self, **kwargs):
        """
        Select a subset of the data using integer indexing.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to xarray.Dataset.isel()

        Returns
        -------
        ShadowBase
            New instance with selected data
        """
        original_dims = list(self.dataset.sizes.keys())
        dataset = self.dataset.isel(**kwargs)

        squeezed_dims = [d for d in original_dims if d not in dataset.sizes]
        if squeezed_dims:
            dataset = dataset.expand_dims(squeezed_dims)

        if list(dataset.sizes.keys()) != original_dims:
            dataset = dataset.transpose(
                *[d for d in original_dims if d in dataset.sizes])

        return self.__class__(dataset)

    def get_rotor_shadow(self):
        """
        Return rotor shadow data.

        Returns
        -------
        xarray.DataArray
            Rotor shadow data
        """
        return self.dataset["Rotor shadow"]

    def get_tower_shadow(self):
        """
        Return tower shadow data.

        Returns
        -------
        xarray.DataArray
            Tower shadow data
        """
        return self.dataset["Tower shadow"]

    def get_combined_shadow(self):
        """
        Return combined shadow data based on the current mode.

        Returns
        -------
        xarray.DataArray
            Combined shadow data

        Raises
        ------
        ValueError
            If the shadow mode is invalid or unable to determine appropriate shadow data
        """
        if self.mode == "combined":
            return self.dataset["Combined shadow"]
        elif self.mode == "both":
            return np.maximum(self.get_rotor_shadow(), self.get_tower_shadow())
        elif self.mode == "rotor":
            return self.get_rotor_shadow()
        elif self.mode == "tower":
            return self.get_tower_shadow()
        else:
            raise ValueError("Invalid shadow mode or missing shadow data")

    def collapse_time(self):
        """
        Collapse time dimension to a single time step.

        Returns
        -------
        ShadowBase
            Self with collapsed time dimension
        """
        self.dataset = self.dataset.sum("time", keep_attrs=True, keepdims=True)
        return self

    def collapse_turbine(self):
        """
        Collapse turbine dimension to a single turbine.
        For shadow-related variables, applies logical_or to maintain boolean format.
        Other variables are set to NaN while maintaining dimensions.

        Returns
        -------
        ShadowBase
            Self with collapsed turbine dimension
        """

        coords = {}
        for name, coord in self.dataset.coords.items():
            if name == "wt":
                coords[name] = coord.isel(wt=slice(0, 1))
            else:
                coords[name] = coord

        new_ds = xr.Dataset(coords=coords)

        shadow_vars = ["Rotor shadow", "Tower shadow", "Combined shadow"]

        for var_name, var in self.dataset.data_vars.items():
            if var_name in shadow_vars and "wt" in var.dims:
                combined_var = var.max("wt", keepdims=True)
                new_ds[var_name] = combined_var

            elif "wt" in var.dims:
                if np.issubdtype(var.dtype, np.integer):
                    first_var = var.isel(wt=slice(0, 1)).astype(float)
                else:
                    first_var = var.isel(wt=slice(0, 1))

                new_ds[var_name] = first_var * np.nan
            else:
                new_ds[var_name] = var

        new_ds.attrs.update(self.dataset.attrs)

        self.dataset = new_ds
        return self

    def collapse_shadow_type(self):
        """
        Collapse shadow type dimension to a single combined shadow type.

        Returns
        -------
        ShadowBase
            Self with collapsed shadow types

        Raises
        ------
        ValueError
            If not in "both" mode where both rotor and tower shadows exist
        """
        if self.mode == "both":
            self.dataset["Combined shadow"] = np.maximum(
                self.get_rotor_shadow(), self.get_tower_shadow()
            )
            self.dataset = self.dataset.drop_vars(
                ["Rotor shadow", "Tower shadow"])
            self.dataset.attrs["mode"] = "combined"
            self.mode = "combined"
            return self
        else:
            raise ValueError(
                "Cannot collapse shadow type for single shadow mode")

    def save(self, filepath):
        """
        Save shadow results to a NetCDF file.
        Parameters
        ----------
        filepath : str or path-like object
        Path to save the NetCDF file. Extension .nc will be added if not present.
        """
        dataset_to_save = self.dataset.copy()
        dataset_to_save.attrs["save_time"] = np.datetime64("now").astype(str)
        dataset_to_save.attrs["freq"] = self.freq
        dataset_to_save.attrs["mode"] = self.mode
        if "time" in dataset_to_save.coords:
            try:
                times = pd.DatetimeIndex(dataset_to_save.time.values)
                dataset_to_save = dataset_to_save.assign_coords(
                    time=times.to_numpy())
            except Exception as e:
                warnings.warn(
                    f"Could not convert time coordinate: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                try:
                    dataset_to_save = dataset_to_save.assign_coords(
                        time=[np.datetime64(t)
                              for t in dataset_to_save.time.values]
                    )
                except BaseException:
                    raise ValueError(
                        "Cannot convert time data to a serializable format"
                    )

        filepath = str(filepath)
        if not filepath.endswith(".nc"):
            filepath = f"{filepath}.nc"
        dataset_to_save.to_netcdf(filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load shadow results from a NetCDF file.

        Parameters
        ----------
        filepath : str
            Path to the NetCDF file to load

        Returns
        -------
        ShadowBase
            New instance with loaded data

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        return cls(xr.open_dataset(filepath))

    def __repr__(self):
        """String representation of the shadow base object"""
        return self.dataset.__repr__().replace(
            "xarray.Dataset", f"xarray.{self.__class__.__name__}"
        )

    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset if not found in this class"""
        try:
            return getattr(self.dataset, name)
        except AttributeError:
            raise AttributeError(
                f"Neither {
                    self.__class__.__name__} nor its dataset has attribute '{name}'"
            )

    def __getitem__(self, item):
        return self.dataset[item]


class ShadowMap(ShadowBase):
    """
    Class for grid-based shadow calculation results with visualization capabilities.

    Extends ShadowBase to provide mapping and visualization of shadow patterns.
    """

    def plot(
        self,
        norm=None,
        cmap="viridis",
        figsize=(10, 8),
        title="Shadow Map",
        show=True,
        colorbar_label="Shadow Hours",
        save_path=None,
        source_style=None,
        levels=None,
    ):
        """
        Plot the accumulated shadow hours on a map with discrete color levels.

        Parameters
        ----------
        norm : matplotlib.colors.Normalize, optional
            Normalization for the colormap. If None, uses BoundaryNorm with levels.
        cmap : str or matplotlib.colors.Colormap, default "viridis"
            Colormap to use for shadow hours visualization.
        figsize : tuple(float, float), default (10, 8)
            Figure size in inches (width, height).
        title : str, default "Shadow Map"
            Title for the plot.
        show : bool, default True
            Whether to show the plot immediately.
        colorbar_label : str, default "Shadow Hours"
            Label for the colorbar.
        save_path : str, optional
            If provided, saves the figure to this path.
        source_style : dict, optional
            Dictionary of style parameters for source markers.
            Example: {'color': 'red', 's': 100, 'marker': '*', 'label': 'Turbines'}
        levels : array-like, optional
            Custom levels for discrete coloring. If None, levels are calculated
            using [0, 1, 2, 5] pattern followed by doubling values.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """

        shadow_hours = np.sum(
            np.sum(self.get_combined_shadow(), axis=-1) / self.freq, axis=0
        ).squeeze()
        max_val = np.max(shadow_hours.values)

        if levels is None:
            levels = [0, 1, 2, 5]
            current = levels[-1]
            while current * 2 <= max_val:
                current *= 2
                levels.append(current)
        else:
            levels = np.unique(np.array(levels))

        fig, ax = plt.subplots(figsize=figsize)

        cmap_obj = plt.get_cmap(cmap)
        if norm is None:
            norm = BoundaryNorm(levels, cmap_obj.N)

        im = ax.pcolormesh(
            self.dataset.x,
            self.dataset.y,
            shadow_hours,
            cmap=cmap_obj,
            norm=norm,
            shading="nearest",
        )

        plt.colorbar(im, ax=ax, label=colorbar_label,
                     ticks=levels, extend="max")

        default_style = {"color": "k", "s": 50,
                         "marker": "x", "label": "Source"}
        if source_style is not None:
            default_style.update(source_style)

        if self.mode != "combined":
            ax.scatter(self.src_x, self.src_y, **default_style)
            ax.legend()

        ax.set(xlabel="Longitude", ylabel="Latitude",
               title=title, aspect="equal")

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()

        return fig, ax

    def animate(
        self,
        ax=None,
        step=1,
        interval=100,
        show=False,
        save_path=None,
        custom_colors=None,
        title="Shadow Animation",
        legend_position="bottom",
        xlabel="Longitude",
        ylabel="Latitude",
    ):
        """
        Generate and optionally save animations of shadow movement over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes to use for animation. If None, the current axes
            (``plt.gca()``) are used.
        step : int, default 1
            Step size for time series sampling. Higher values reduce frame count.
        interval : int, default 100
            Interval between animation frames in milliseconds.
        show : bool
            Whether to display the plot
        save_path : str, optional
            Path to save the animation. File extension determines format (.mp4, .gif, etc.).
        custom_colors : list of RGBA tuples, optional
            Custom colors for [no shadow, rotor shadow, tower shadow, overlapping shadow].
        title : str, default "Shadow Animation"
            Title for the animation plot.
        legend_position : str, default 'bottom'
            Position for the legend: 'bottom', 'right', or 'none' to hide.
        xlabel : str, default 'Longitude'
            Label for the x-axis. Set to e.g. 'Easting [m]' when the output
            coordinates are given in a projected CRS such as UTM.
        ylabel : str, default 'Latitude'
            Label for the y-axis. Set to e.g. 'Northing [m]' when the output
            coordinates are given in a projected CRS such as UTM.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object that must be stored in a variable to prevent garbage collection.
        """

        # Convert time values and create indices for animation frames
        try:
            timezone = self.dataset.attrs.get("tz", "UTC")
            times = pd.DatetimeIndex(
                self.dataset.coords["time"].values, tz="UTC")
            try:
                times = times.tz_convert(timezone)
            except ZoneInfoNotFoundError:
                raise ValueError(f"'{timezone}'")
            times = times[::step]
            indices = np.arange(0, len(self.dataset.coords["time"]), step)
        except (KeyError, AttributeError, ValueError) as e:
            raise ValueError(f"Error processing time data: {e}")

        # Use the provided axes or fall back to the current axes
        ax = ax or plt.gca()
        fig = ax.figure

        # Define color scheme
        if custom_colors is None:
            if self.mode == "combined":
                colors = [
                    (0, 0, 0, 0),  # Transparent for no shadow
                    (0.5, 0, 0.5, 0.7),  # Purple for combined shadow
                ]
            else:
                colors = [
                    (0, 0, 0, 0),  # Transparent for no shadow
                    (1, 0, 0, 0.7),  # Red for rotor shadow
                    (0, 0, 1, 0.7),  # Blue for tower shadow
                    (0.5, 0, 0.5, 0.7),  # Purple for overlapping shadow
                ]
        else:
            colors = custom_colors

        # Create colormap and normalization
        cmap = ListedColormap(colors)
        bounds = (
            [-0.5, 0.5, 1.5] if self.mode == "combined" else [-0.5, 0.5, 1.5, 2.5, 3.5]
        )
        norm = BoundaryNorm(bounds, cmap.N)

        # Create legend elements
        legend_patches = []
        legend_labels = []

        if self.mode == "combined":
            legend_patches.append(plt.Rectangle(
                (0, 0), 1, 1, fc=colors[1], ec="none"))
            legend_labels.append("Shadow")
        else:
            if self.mode in ["both", "rotor"]:
                legend_patches.append(
                    plt.Rectangle((0, 0), 1, 1, fc=colors[1], ec="none")
                )
                legend_labels.append("Rotor shadow")

            if self.mode in ["both", "tower"]:
                legend_patches.append(
                    plt.Rectangle((0, 0), 1, 1, fc=colors[2], ec="none")
                )
                legend_labels.append("Tower shadow")

            if self.mode == "both":
                legend_patches.append(
                    plt.Rectangle((0, 0), 1, 1, fc=colors[3], ec="none")
                )
                legend_labels.append("Overlapping shadow")

        # Plot source locations if available
        try:
            if hasattr(self, "src_x") and hasattr(self, "src_y"):
                ax.scatter(
                    self.src_x, self.src_y, color="k", label="Sources", marker="2", s=50
                )
        except Exception:
            pass

        timestamp = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        existing_handles, existing_labels = ax.get_legend_handles_labels()
        legend_elements = legend_patches + existing_handles
        legend_labels = legend_labels + existing_labels

        if legend_position != "none" and legend_elements:
            if legend_position == "bottom":
                ax.legend(
                    legend_elements,
                    legend_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    ncol=min(len(legend_elements),
                             fig.get_size_inches()[0] // 2),
                )
            elif legend_position == "right":
                ax.legend(
                    legend_elements,
                    legend_labels,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                )

        x, y = self.dataset.x.values, self.dataset.y.values
        template = np.zeros((len(y), len(x)), dtype=int)

        mesh = ax.pcolormesh(x, y, template, cmap=cmap,
                             norm=norm, shading="nearest")

        ax.set(
            xlim=(x.min(), x.max()),
            ylim=(y.min(), y.max()),
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        ax.grid(alpha=0.3)

        def update(frame):
            """Update function for animation frames"""
            idx = indices[frame]

            if self.mode == "combined":
                shadow_data = (
                    np.max(self.dataset["Combined shadow"].isel(
                        time=idx), axis=0)
                    .squeeze()
                    .values.astype(int)
                )
            else:
                rotor = np.zeros_like(template, dtype=bool)
                tower = np.zeros_like(template, dtype=bool)

                if self.mode in ["both", "rotor"]:
                    rotor = (
                        np.max(self.dataset["Rotor shadow"].isel(
                            time=idx), axis=0)
                        .squeeze()
                        .values
                    )

                if self.mode in ["both", "tower"]:
                    tower = (
                        np.max(self.dataset["Tower shadow"].isel(
                            time=idx), axis=0)
                        .squeeze()
                        .values
                    )

                shadow_data = np.zeros_like(template, dtype=int)
                shadow_data[np.logical_and(rotor, ~tower)] = 1  # Rotor only
                shadow_data[np.logical_and(tower, ~rotor)] = 2  # Tower only
                # Both shadows
                shadow_data[np.logical_and(rotor, tower)] = 3

            mesh.set_array(shadow_data)

            time_str = pd.to_datetime(str(times[frame])).strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            timestamp.set_text(f"Time: {time_str}")

            return mesh, timestamp

        plt.tight_layout()
        anim = FuncAnimation(
            fig, update, frames=len(indices), interval=interval, blit=True
        )

        if save_path:
            writer = "pillow" if save_path.endswith(
                (".gif", ".png")) else "ffmpeg"
            anim.save(save_path, writer=writer)

        if show:
            plt.show()

        return anim


class ShadowResult(ShadowBase):
    """Class for receptor-based shadow calculation results"""

    def plot(
        self,
        figsize=(10, 6),
        title="Shadow at Receptors",
        show=True,
        save_path=None,
        cmap="jet",
        marker_size=100,
        annotate=True,
        aspect="equal",
    ):
        """
        Plot a visualization of shadow hours at receptor locations and wind turbine positions.

        Parameters
        ----------
        figsize : tuple
            Figure size as (width, height) in inches
        title : str
            Title of the plot
        show : bool
            Whether to display the plot
        save_path : str or None
            Path to save the figure; if None, figure is not saved
        colormap : str
            Matplotlib colormap name to use for shadow hour visualization
        marker_size : int
            Size of the receptor markers in the scatter plot
        annotate : bool
            Whether to add text annotations showing shadow hours and receptor/turbine numbers
        aspect : str or float
            Aspect ratio of the plot ('equal', 'auto', or a float)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object
        ax : matplotlib.axes.Axes
            The matplotlib axes object
        """
        fig, ax = plt.subplots(figsize=figsize)

        receptor_x = self.dataset["rec_x"].values.squeeze()
        receptor_y = self.dataset["rec_y"].values.squeeze()
        receptor_ids = self.dataset.rec.values

        turbine_x = self.dataset["src_x"].values.squeeze()
        turbine_y = self.dataset["src_y"].values.squeeze()
        turbine_ids = self.dataset.wt.values

        shadow_hours = np.sum(
            np.sum(self.get_combined_shadow(), axis=-1) / self.freq, axis=0
        )

        sc = ax.scatter(
            receptor_x,
            receptor_y,
            c=shadow_hours,
            cmap=cmap,
            s=marker_size,
            edgecolor="black",
            label="Receptors",
        )

        ax.scatter(
            turbine_x,
            turbine_y,
            marker="2",
            color="red",
            s=marker_size * 1.5,
            label="Wind Turbines",
        )

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        ax.set(title=title, xlabel="Longitude",
               ylabel="Latitude", aspect=aspect)

        if annotate:
            for i, (x, y, hours, rec_id) in enumerate(
                zip(receptor_x, receptor_y, shadow_hours, receptor_ids)
            ):
                if x > x_min + 0.85 * x_range:
                    xytext = (-5, 0)
                    ha = "right"
                else:
                    xytext = (5, 5)
                    ha = "left"

                if y > y_min + 0.9 * y_range:
                    xytext = (xytext[0], -5)
                    va = "top"
                else:
                    va = "bottom"

                text = f"rec{rec_id}: {hours:.1f}h"

                ax.annotate(
                    text,
                    xy=(x, y),
                    xytext=xytext,
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                )

            if not np.isnan([turbine_x, turbine_y]).any():
                turbine_x = np.atleast_1d(turbine_x)
                turbine_y = np.atleast_1d(turbine_y)

                for i, (x, y, wt_id) in enumerate(
                        zip(turbine_x, turbine_y, turbine_ids)):
                    ax.annotate(
                        f"wt{wt_id}",
                        xy=(x, y),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Shadow Hours")

        ax.legend(loc="best")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()

        return fig, ax

    def calendar(
        self,
        hour_range=None,
        figsize=None,
        title="Shadow Calendar",
        show=True,
        save_path=None,
        receptor_labels=None,
        turbine_labels=None,
    ):
        """
        Create calendar visualizations of shadow events at receptors

        Parameters
        ----------
        hour_range : tuple of (start_hour, end_hour) or None
            Custom hour range to display (e.g., (6, 18) for daylight hours only)
        figsize : tuple of (width, height) or None
            Figure dimensions in inches
        title : str, default "Shadow Calendar"
            Title for the overall figure
        show : bool
            Whether to display the plot
        save_path : str or None
            If provided, saves the figure to this path
        receptor_labels : list of str or None
            Custom labels for each receptor point
        turbine_labels : list of str or None
            Custom labels for each turbine
        """
        if self.mode == "combined":
            shadow_data = self.dataset["Combined shadow"]
        elif self.mode == "rotor":
            shadow_data = self.get_rotor_shadow()
        elif self.mode == "tower":
            shadow_data = self.get_tower_shadow()
        elif self.mode == "both":
            rotor_shadow = self.get_rotor_shadow()
            tower_shadow = self.get_tower_shadow()
        else:
            raise ValueError("Invalid shadow mode")

        times = pd.DatetimeIndex(
            self.dataset.coords["time"].values, tz="UTC"
        ).tz_convert(self.dataset.attrs["tz"])

        rec_lon = self.dataset.get("rec_lon", self.dataset["rec_x"]).values
        rec_lat = self.dataset.get("rec_lat", self.dataset["rec_y"]).values

        solar_position_kwargs = {}
        if "solar_position_method" in self.dataset.attrs:
            solar_position_kwargs["method"] = self.dataset.attrs[
                "solar_position_method"]
        _, rec_celestial_coords = solar_position(
            times, rec_lat, rec_lon, **solar_position_kwargs)

        dates, times_of_day = times.date, times.time
        unique_dates = np.unique(dates)
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}

        start_hour, end_hour = hour_range if hour_range else (0, 24)
        if not (0 <= start_hour <= 24 and 0 <= end_hour <= 24):
            raise ValueError("Hours must be between 0 and 24")
        if start_hour > end_hour:
            raise ValueError(
                "Start hour must be less than or equal to end hour")

        minutes_per_day = (end_hour - start_hour) * 60
        minute_offset = start_hour * 60

        minute_indices = np.array(
            [(t.hour * 60 + t.minute - minute_offset) for t in times_of_day]
        )
        valid_time_mask = (minute_indices >= 0) & (
            minute_indices < minutes_per_day)

        n_turbines = len(self.dataset.wt)
        n_receptors = len(self.dataset.rec)

        if turbine_labels is not None and len(turbine_labels) != n_turbines:
            raise ValueError(
                "Length of turbine_labels must match number of turbines"
            )
        if receptor_labels is not None and len(receptor_labels) != n_receptors:
            raise ValueError(
                "Length of receptor_labels must match number of receptors"
            )

        n_cols = min(n_receptors, 3) if n_receptors > 2 else n_receptors
        n_rows = (n_receptors + n_cols - 1) // n_cols
        figsize = figsize or (5 * n_cols, 4 * n_rows + 1.5)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle(title, fontsize=16)

        turbine_colors = plt.cm.tab10(np.linspace(0, 1, n_turbines))
        legend_items = {}
        x_date = mdates.date2num(pd.to_datetime(unique_dates))
        x_min, x_max = x_date[0], x_date[-1]
        y_minutes = np.arange(minutes_per_day)
        date_grid, minute_grid = np.meshgrid(x_date, y_minutes)

        for receptor_idx in range(n_receptors):
            ax = axes[receptor_idx // n_cols, receptor_idx % n_cols]
            ax.set_title(
                receptor_labels[receptor_idx]
                if receptor_labels
                else f"Receptor {receptor_idx + 1}"
            )

            for turbine_idx in range(n_turbines):
                shadow_2d = np.zeros(
                    (minute_grid.shape[0],
                     minute_grid.shape[1]), dtype=np.uint8
                )
                valid_times = np.where(valid_time_mask)[0]

                if valid_times.size > 0:
                    date_indices = np.array(
                        [date_to_idx[d] for d in dates[valid_times]]
                    )
                    minutes = minute_indices[valid_times]

                    if self.mode == "combined":
                        shadow_mask = shadow_data[
                            turbine_idx, receptor_idx, valid_times
                        ]
                        if np.any(shadow_mask):
                            shadow_2d[
                                minutes[shadow_mask.astype(bool)],
                                date_indices[shadow_mask.astype(bool)],
                            ] = 1
                    elif self.mode in ["rotor", "tower"]:
                        shadow_mask = shadow_data[
                            turbine_idx, receptor_idx, valid_times
                        ]
                        if np.any(shadow_mask):
                            shadow_2d[
                                minutes[shadow_mask.astype(bool)],
                                date_indices[shadow_mask.astype(bool)],
                            ] = 1
                    else:  # mode == "both"
                        rotor_mask = rotor_shadow[
                            turbine_idx, receptor_idx, valid_times
                        ]
                        tower_mask = tower_shadow[
                            turbine_idx, receptor_idx, valid_times
                        ]

                        if np.any(rotor_mask) or np.any(tower_mask):
                            for mask, value in [
                                (rotor_mask & tower_mask, 3),
                                (tower_mask & ~rotor_mask, 2),
                                (rotor_mask & ~tower_mask, 1),
                            ]:
                                if mask.any():
                                    shadow_2d[
                                        minutes[mask.astype(bool)],
                                        date_indices[mask.astype(bool)],
                                    ] = value

                    if np.any(shadow_2d > 0):
                        base_rgba = turbine_colors[turbine_idx]
                        if self.mode in ["combined", "rotor", "tower"]:
                            alphas = [0, 0.7]  # Single shadow type
                            cmap = ListedColormap(
                                [(0, 0, 0, 0), (*base_rgba[:3], alphas[1])]
                            )
                            vmax = 1
                        else:  # mode == "both"
                            alphas = [0, 0.2, 0.55, 1.0]
                            cmap = ListedColormap(
                                [
                                    (0, 0, 0, 0)
                                    if i == 0
                                    else (*base_rgba[:3], alphas[i])
                                    for i in range(4)
                                ]
                            )
                            vmax = 3

                        ax.pcolormesh(
                            date_grid,
                            minute_grid,
                            shadow_2d,
                            cmap=cmap,
                            shading="nearest",
                            vmin=0,
                            vmax=vmax,
                        )

                        turbine_name = (
                            turbine_labels[turbine_idx]
                            if turbine_labels
                            else f"Turbine {turbine_idx + 1}"
                        )
                        if turbine_name not in legend_items:
                            legend_items[turbine_name] = {}

                        if self.mode in ["combined", "rotor", "tower"]:
                            shadow_type = "Shadow"
                            legend_items[turbine_name][shadow_type] = mpatches.Patch(
                                color=base_rgba[:3], alpha=alphas[1]
                            )
                        else:
                            for shadow_type, value, alpha in [
                                ("Both", 3, 1.0),
                                ("Tower", 2, 0.55),
                                ("Rotor", 1, 0.2),
                            ]:
                                legend_items[turbine_name][
                                    shadow_type
                                ] = mpatches.Patch(color=base_rgba[:3], alpha=alpha)

            altitude = rec_celestial_coords[receptor_idx, :, 1]
            transitions = np.diff((altitude > 0).astype(int))

            for transition_value, line_style in [(1, "-"), (-1, "--")]:
                indices = np.where(transitions == transition_value)[0]
                crossing_data = []

                for idx in indices:
                    if (idx + 1 >= len(times) or
                            not valid_time_mask[idx] or not valid_time_mask[idx + 1]):
                        continue

                    t1, t2 = pd.to_datetime(
                        times[idx]), pd.to_datetime(times[idx + 1])
                    a1, a2 = altitude[idx], altitude[idx + 1]
                    if np.isnan(a1) or np.isnan(a2):
                        continue

                    t_frac = -a1 / (a2 - a1)
                    t_cross = pd.Timestamp.fromtimestamp(
                        t1.timestamp() + t_frac * (t2.timestamp() - t1.timestamp()), tz=t1.tzinfo,)

                    cross_minutes = (
                        t_cross.hour * 60 + t_cross.minute + t_cross.second / 60 - minute_offset)
                    if 0 <= cross_minutes < minutes_per_day:
                        crossing_data.append((t_cross.date(), cross_minutes))

                if crossing_data:

                    x_dates, y_minutes = zip(*sorted(crossing_data))
                    x_dates = mdates.date2num(x_dates)

                    i, segments = 0, []
                    while i < len(x_dates):
                        seg_x, seg_y = [x_dates[i]], [y_minutes[i]]
                        j = i + 1
                        while (j < len(x_dates) and abs(y_minutes[j] - y_minutes[j - 1]) < max(
                                minutes_per_day / 3, 240) and (x_dates[j] - x_dates[j - 1]) < 1.5):
                            seg_x.append(x_dates[j])
                            seg_y.append(y_minutes[j])
                            j += 1
                        segments.append((seg_x, seg_y))
                        i = j

                    for seg_x, seg_y in segments:
                        if len(seg_x) > 1:
                            ax.plot(seg_x, seg_y, color="black", linestyle=line_style,
                                    linewidth=1, alpha=0.7, zorder=10)

            date_range = (
                pd.to_datetime(unique_dates[-1]) -
                pd.to_datetime(unique_dates[0])
            ).days

            if date_range <= 7:
                ax.xaxis.set_major_locator(mdates.DayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            elif date_range <= 60:
                ax.xaxis.set_major_locator(
                    mdates.WeekdayLocator(byweekday=mdates.MO))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            else:
                ax.xaxis.set_major_locator(
                    mdates.MonthLocator(interval=1 if date_range <= 365 else 3)
                )
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            hour_interval = 2 if minutes_per_day > 480 else 1
            ax.set_ylim(0, minutes_per_day)
            ax.set_yticks(
                np.arange(0, minutes_per_day + 1, hour_interval * 60))
            ax.set_yticklabels(
                [
                    f"{(h % 24):02d}:00"
                    for h in range(start_hour, end_hour + 1, hour_interval)
                ]
            )
            ax.grid(True, linestyle=":", alpha=0.7)

            if x_min == x_max:
                ax.set_xlim(x_min - 1, x_max + 1)
            else:
                ax.set_xlim(x_min, x_max)

            if receptor_idx % n_cols == 0:
                ax.set_ylabel("Time of Day")
            if receptor_idx // n_cols == (n_rows - 1):
                ax.set_xlabel("Date")

        for i in range(n_receptors, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        if legend_items:
            turbine_names = sorted(
                list(legend_items.keys()),
                key=lambda x: int(
                    x.split()[-1]) if x.split()[-1].isdigit() else x,
            )

            shadow_types = (
                ["Shadow"] if self.mode == "combined" else [
                    "Both", "Tower", "Rotor"]
            )

            handles, labels = [], []
            for turbine in turbine_names:
                for shadow_type in shadow_types:
                    if shadow_type in legend_items[turbine]:
                        handles.append(legend_items[turbine][shadow_type])
                        labels.append(f"{turbine} ({shadow_type})")

            if handles:
                fig.legend(
                    handles=handles,
                    labels=labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0),
                    ncol=len(turbine_names),
                    frameon=True,
                )

        plt.tight_layout(rect=[0, 0.15, 1, 0.95])

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()

        return fig, axes
