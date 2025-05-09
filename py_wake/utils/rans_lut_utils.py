import xarray as xr
import numpy as np
from numpy import newaxis as na
from py_wake.tests.test_files import tfp


def fugalut2ranslut(path_fuga_LUT=tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc', Dref=80.0, zRef=70.0, rename=True, nondim=False, cover_rotor_area=False):
    """
    Create a simple LUT file for testing from Fuga
    """
    lut = xr.load_dataset(path_fuga_LUT)
    lut = lut.rename(UL='deficits')
    cts = [0.1, 0.818]
    TIs = [0.075, 10.0]
    deficit = lut.deficits.transpose('x', 'y', 'z')
    if cover_rotor_area:
        # Move levels to cover rotor area
        deficit['z'] = np.array([zRef - 0.5 * Dref - 0.1, zRef + 0.5 * Dref + 0.1])
    # To avoid huge upstream blockage and distances tests from failing
    deficit[0, :, :] = 0.0
    deficit[-1, :, :] = 0.0
    deficit[:, -1, :] = 0.0
    # To avoid test_PropagateUpDownIterative_wake_deficitModels test failure
    deficit[:, 60:-1, :] = 0.0
    # Mirror deficit at y=0 since Fuga only stores one side of the wake
    deficit2 = deficit.copy()
    deficit['y'] = -deficit['y']
    deficit = xr.merge([deficit, deficit2])
    # Add ct and ti dimensions
    subluts_ti = []
    for TI in TIs:
        subluts_ct = []
        for ct in cts:
            subluts_ct.append(-deficit * ct)
        subluts_ti.append(xr.concat(subluts_ct, dim='ct').assign_coords(ct=cts))
    da = xr.concat(subluts_ti, dim='ti').assign_coords(ti=TIs)
    if nondim:
        da['x'] = da['x'] / Dref
        da['y'] = da['y'] / Dref
        da['z'] = (da['z'] - zRef) / Dref
    da = da.deficits
    if rename:
        da = rename_lut(da)
    da = da.astype('float64')
    # Create a dummy added_ti lut. To be replaced with a RANS wake added TI
    da_ti = -da.copy() * 0.1
    # Set uptream wake added TI to zero to avoid huge upstream blockage turb test from failing
    da_ti[:, :, 0:513, :, :] = 0.0
    ds = xr.Dataset(data_vars={'deficits': da, 'added_ti': da_ti})
    # Add meta data
    ds.attrs['name'] = 'V80'
    ds.attrs['diameter'] = Dref
    ds.attrs['hub_height'] = zRef
    return ds


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
    for i in range(len(types)):
        datas[i] = datas[i].interp(x=xref, y=yref, z=heightsall, ct=ctall, ti=datas[0].ti)
    # Find smallest turbine based on diameter
    imin = np.argmin(Drefs)
    ctall = np.sort(np.unique(np.round(np.asarray(cts) * 1e10) / 1e10))
    heightsall = np.sort(np.unique(heights))
    # Interpolate other LUT files to LUT file of the smallest diameter
    xref = datas[imin].x
    yref = datas[imin].y
    for i in range(len(types)):
        datas[i] = datas[i].interp(x=xref, y=yref, z=heightsall, ct=ctall, ti=datas[0].ti)
    # Merge
    dataset = xr.concat(datas, dim="type_i").assign_coords(type_i=types)
    # Add meta data
    dataset.attrs = {}
    dataset.attrs['names'] = names
    dataset.attrs['diameters'] = Drefs
    dataset.attrs['hub_heights'] = zRefs
    encoding = {}
    vars = ['x', 'y', 'z', 'ct', 'ti', 'deficits', 'added_ti']
    # for var in vars:
    #     encoding[var] = {'dtype': 'float32'}
    # encoding['type_i'] = {'dtype': 'int32'}
    for var in vars:
        encoding[var] = {'dtype': 'float64'}
    encoding['type_i'] = {'dtype': 'int64'}
    # Save
    dataset.to_netcdf(outfile, encoding=encoding)
