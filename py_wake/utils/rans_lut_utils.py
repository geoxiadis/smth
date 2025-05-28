import xarray as xr
import numpy as np
from numpy import newaxis as na
from pathlib import Path
from py_wake.tests import ptf


def convert_ranslut(path_LUT=Path(ptf('ranslut/V80_ranslut_demo.nc', 'bb368d8235ea3c300bc9d38a9fc3e3424217dae110800839ab7ffc79ffe48476')),
                    Dref=80.0, zRef=70.0, rename=True, nondim=False, cover_rotor_area=False):
    """
    Convert RANS LUT for testing
   """
    lut = xr.load_dataset(path_LUT)
    if cover_rotor_area:
        lut['z'] = [-0.5 - 0.1 / Dref, 0.5 + 0.1 / Dref]
    if not nondim:
        lut['x'] = lut['x'] * Dref
        lut['y'] = lut['y'] * Dref
        lut['z'] = lut['z'] * Dref + zRef
    if rename:
        lut = rename_lut(lut)
    return lut


def rename_lut(lut):
    lutvars = ['x', 'y', 'z', 'ct', 'ti', 'yaw', 'type_i']
    renamevars = ['dw_ijlk', 'hcw_ijlk', 'z_ijlk', 'ct_ilk', 'TI_eff_ilk', 'yaw_ilk', 'type_i_ilk']
    d = {}
    for i in range(len(lutvars)):
        if lutvars[i] in lut.coords:
            d[lutvars[i]] = renamevars[i]
    return lut.rename(d)


def cxzy_to_cxyzg(cxyz, o3, o2, pos):
    # Global poly center and orientation
    o1 = np.cross(o2, o3)
    cxyzg = cxyz[:, :, :, 0, na] * o1[:, na, na, :] + cxyz[:, :, :, 1, na] * o2[:, na, na, :] + cxyz[:, :, :, 2, na] * o3[:, na, na, :] + pos[:, na, na, :]
    return cxyzg


def merge_luts(LUTs, types=None, outfile='LUT_merged.nc'):
    """
    When merging LUTs, we need to be carefull about merging the dimensions
    in order to not have huge LUT files that become memory intensive.
    Therefore, we use the same dx and dy (PyWake interpolates in [m])
    but we include all z levels; all through interpolation.
    The reference xy-grid is the LUT for the smallest diameter
    It should also be noted that the output LUT file coords are now dimensional [m].
    """
    Nluts = len(LUTs)
    if types is None:
        types = np.linspace(0, Nluts - 1, Nluts, dtype=int)
    Drefs = []
    zRefs = []
    names = []
    deficits = []
    maxct = []
    cts = []
    heights = []
    datas = []
    # Read LUTs
    for i in range(Nluts):
        data = xr.open_dataset(LUTs[i])
        Dref = data.diameter
        zRef = data.hub_height
        data['x'] = data['x'] * Dref
        data['y'] = data['y'] * Dref
        data['z'] = data['z'] * Dref + zRef
        datas.append(data)
        Drefs.append(Dref)
        zRefs.append(zRef)
        names.append(data.name)
        maxct.append(data.ct.max())
        cts = cts + list(data.ct.values)
        heights = heights + list(data.z.values)

    # Find smallest turbine based on diameter
    imin = np.argmin(Drefs)
    ctall = np.sort(np.unique(np.round(np.asarray(cts) * 1e10) / 1e10))
    heightsall = np.sort(np.unique(heights))
    # Interpolate other LUT files to LUT file of the smallest diameter
    xref = datas[imin].x
    yref = datas[imin].y
    # To avoid Nan values we extrapolate bounds
    kwargs = {"fill_value": "extrapolate"}
    for i in range(len(types)):
        datas[i] = datas[i].interp(x=xref, y=yref, z=heightsall, ct=ctall, ti=datas[0].ti, kwargs=kwargs)
    # Find smallest turbine based on diameter
    imin = np.argmin(Drefs)
    ctall = np.sort(np.unique(np.round(np.asarray(cts) * 1e10) / 1e10))
    heightsall = np.sort(np.unique(heights))
    # Interpolate other LUT files to LUT file of the smallest diameter
    xref = datas[imin].x
    yref = datas[imin].y
    for i in range(len(types)):
        datas[i] = datas[i].interp(x=xref, y=yref, z=heightsall, ct=ctall, ti=datas[0].ti, kwargs=kwargs)
    # Merge
    dataset = xr.concat(datas, dim="type_i").assign_coords(type_i=types)
    # Add meta data
    dataset.attrs = {}
    dataset.attrs['names'] = names
    dataset.attrs['diameters'] = Drefs
    dataset.attrs['hub_heights'] = zRefs
    encoding = {}
    vars = ['deficits', 'added_ti']
    for var in vars:
        encoding[var] = {'dtype': 'float32'}
    vars = ['x', 'y', 'z', 'ct', 'ti']
    for var in vars:
        encoding[var] = {'dtype': 'float64'}
    encoding['type_i'] = {'dtype': 'int64'}
    # Save
    dataset.to_netcdf(outfile, encoding=encoding)
