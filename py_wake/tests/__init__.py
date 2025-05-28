from pathlib import Path
import tempfile

import numpy
import pooch

from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
import xarray as xr
npt = numpy.testing
WindFarmModel.verbose = False


def ptf(filename, known_hash=None):
    # Download a file and save it locally, returning the path to it.
    # Running this again will not cause a download. Pooch will check the hash
    # (checksum) of the downloaded file against the given value to make sure
    # it's the right file (not corrupted or outdated).

    pct_encoding = {'/': '%2F', '.': '%2E', '_': '%5F', '-': '%2D'}
    for k, v in pct_encoding.items():
        filename = filename.replace(k, v)
    ret = pooch.retrieve(url=f"https://gitlab.windenergy.dtu.dk/api/v4/projects/3385/repository/files/{filename}/raw",
                         path=Path.home() / 'PyWakeTestFiles',
                         known_hash=known_hash)
    return ret
