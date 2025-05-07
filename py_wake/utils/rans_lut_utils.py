import xarray as xr
from py_wake.tests.test_files import tfp


def fugalut2ranslut(path_fuga_LUT=tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc', Dref=80.0, zRef=70.0):
    """
    Create a simple LUT file for testing from Fuga
    """
    lut = xr.load_dataset(path_fuga_LUT)
    lut = lut.rename(UL='deficits')
    cts = [0.1, 0.818]
    TIs = [0.075, 10.0]
    deficit = lut.deficits.transpose('x', 'y', 'z')
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
    da['x'] = da['x'] / Dref
    da['y'] = da['y'] / Dref
    da['z'] = (da['z'] - zRef) / Dref
    da = da.deficits
    da = da.astype('float64')
    # Create a dummy added_ti lut. To be replaced with a RANS wake added TI
    da_ti = da.copy()
    wake_added_ti = 0.05
    da_ti[:, :, :, :, :] = wake_added_ti
    return da, da_ti
