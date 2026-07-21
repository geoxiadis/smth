import os
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from py_wake import np
from py_wake.utils.maps import dk_coast
from py_wake.utils.plotting import setup_plot
from py_wake.wind_turbines._wind_turbines import WindTurbines
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.tests import ptf

xl = 'X (øst) koordinat UTM 32 Euref89'
yl = 'Y (nord) koordinat UTM 32 Euref89'


class DKWindTurbines():
    def __init__(self, update_cache=False):
        folder = Path(__file__).parent
        f = folder / 'dk_turbines.h5'
        if update_cache or not f.exists():
            urllib.request.urlretrieve("https://ens.dk/media/3531/download", folder / 'dk_turbines.xlsx')
            df = pd.read_excel(folder / 'dk_turbines.xlsx')
            i = [df.iloc[i, 0] == 'Møllenummer (GSRN)' for i in range(1, 20)].index(True)
            header = [(h, str(h).replace('\n', ''))[int('\n' in str(h))] for h in df.iloc[i + 1, :]]
            df_dk_wt_data = df.iloc[i + 2:]
            df_dk_wt_data.columns = header
            df_dk_wt_data.to_hdf(f, key="data")
            (folder / 'dk_turbines.xlsx').unlink()
            self.dataframe = df_dk_wt_data
        else:
            self.dataframe = pd.read_hdf(f)
        self.dk_map = dk_coast(crs='EPSG:25832')
        self.reset_filter()

        self.wf_filters = {
            'Rødsand1': dict(xlim=(670000, 700000), ylim=(6040000, 6055000), filter=lambda df: df['Type af placering'] == 'HAV'),
            'Rødsand2': dict(xlim=(640000, 670000), ylim=(6040000, 6055000), filter=lambda df: df['Type af placering'] == 'HAV'),
            'Hornsrev1': dict(xlim=(400000, 430000), ylim=(6140000, 6180000), filter=lambda df: df['Rotor-diameter (m)'] == 80),
            'Hornsrev2': dict(xlim=(400000, 430000), ylim=(6140000, 6180000), filter=lambda df: df['Rotor-diameter (m)'] == 93),
            'Hornsrev3': dict(xlim=(400000, 430000), ylim=(6140000, 6180000), filter=lambda df: df['Rotor-diameter (m)'] == 164),
            'Anholt': dict(xlim=(620000, 700000), ylim=(6250000, 6300000)),
            'Middelgrunden': dict(xlim=(730000, 800000), ylim=(6150000, 6300000)),
            'Vesterhav syd': dict(xlim=(425000, 440000), ylim=(6200000, 6230000)),
            'Vesterhav nord': dict(xlim=(425000, 445000), ylim=(6260000, 6300000)),
            'Krigers Flak': dict(xlim=(730000, 780000), ylim=(6000000, 6120000)),
        }

    def __len__(self):
        return len(self())

    def set_filter(self, xlim=None, ylim=None, filter=None):
        df = self.dataframe
        if filter is None:
            filter = np.ones(len(df), dtype=bool)
        if callable(filter):
            filter = filter(df)
        if xlim:
            filter &= (df[xl] > xlim[0]) & (df[xl] < xlim[1])
        if ylim:
            filter &= (df[yl] > ylim[0]) & (df[yl] < ylim[1])
        self.filter = filter
        return self.dataframe[self.filter]

    def reset_filter(self):
        self.filter = np.ones(len(self.dataframe), dtype=bool)

    def plot(self, ax=None):
        ax = ax or plt.gca()
        self.dk_map.plot(ax=ax)
        df = self.dataframe[self.filter]
        xy = zip(*[(turbine[xl], turbine[yl]) for _, turbine in list(df.iterrows())])
        ax.plot(*xy, '.k')
        ax.set_xlim([df[xl].min() - 10000, df[xl].max() + 10000])
        ax.set_ylim([df[yl].min() - 10000, df[yl].max() + 10000])

    def __call__(self):
        return self.dataframe[self.filter]

    def get_WindTurbines(self, ti=0.075):
        df = self.dataframe[self.filter]

        xy = np.array(df[[xl, yl]].astype(float).values).T
        D = df['Rotor-diameter (m)'].values
        H = df['Navhøjde (m)'].values
        P = df['Kapacitet (kW)'].values
        name = df['Typebetegnelse'].values
        type_id = np.array([f'{n}_{p}_{d}_{h}' for n, d, h, p in zip(name, D, H, P)])
        wt_phdn = np.column_stack((P, H, D, name))
        type = np.zeros(len(df), dtype=int)
        wt_id = df['Møllenummer (GSRN)'].values.astype(str)

        wt_lst = []
        for t, (tid, idx) in enumerate(zip(*np.unique(type_id, return_index=True))):
            p, h, d, n = wt_phdn[idx]
            wt_lst.append(
                GenericWindTurbine(
                    name=n,
                    diameter=float(d),
                    hub_height=float(h),
                    power_norm=float(p),
                    turbulence_intensity=ti))
            type[np.where(type_id == tid)[0]] = t
        return WindTurbines.from_WindTurbine_lst(wt_lst), type, xy, wt_id

    def get_production(self, update_cache=False):
        folder = Path(__file__).parent
        f = folder / 'dk_wind_farm_production.nc'

        if update_cache or not f.exists():
            da = xr.open_dataarray(ptf('dk_data/dk_wind_farm_production.nc',
                                       known_hash='f98583dcb51f67dd4034ad783249ecf8c759d6cc574fab312700e82d04abb255'))
            # The monthly production is temporary not available 05/2026
            # urllib.request.urlretrieve("https://ens.dk/media/3533/download", 'production.xlsx')
            # df_production = pd.read_excel(
            #     'production.xlsx', sheet_name=None, header=8, dtype={
            #         'Møllenummer (identikationsnummer)': str})
            # for k, df in df_production.items():
            #     df_production[k].columns = ['id', 'start', 'end', 'jan', 'feb', 'mar',
            #                                 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'total']
            #     df_production[k].drop_duplicates(subset=['id'], keep='first', inplace=True)
            # da = xr.concat([xr.DataArray(df.iloc[:, 3:-1].values, dims=('id', 'time'),
            #                              coords={'id': df['id'].values.astype(str),
            #                                      'time': pd.to_datetime([f"{k[11:]}-{month}" for month in range(1, 13)])})
            #                 for k, df in df_production.items()], dim='time', join='outer')
            #
            da.to_netcdf(f)
        else:
            da = xr.load_dataarray(f)
        return da.sel(id=self()['Møllenummer (GSRN)'].values.astype(str))

    def get_wind_farm(self, name):
        assert name in self.wf_filters, f"Wind farm {name} not found. Available wind farms: {list(self.wf_filters.keys())}"
        wf = DKWindTurbines()
        wf.set_filter(**wf.wf_filters[name])
        return wf
