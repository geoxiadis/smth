import os

import geopandas as gpd
from py_wake.tests import ptf
import zipfile


def dk_coast(crs=None):
    f = ptf('maps/dk_coast.zip', known_hash='c6b90f62fddec4134762d41777dbfcc50122d8d1a0d2fa0e49a0601382aaecff')
    with zipfile.ZipFile(f, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(f))
    f = os.path.join(os.path.dirname(f), 'dk_coast/dk_coast.shp')
    dk = gpd.read_file(f)
    if crs:
        dk = dk.to_crs(crs)
    return dk


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        dk = dk_coast()
        dk.plot()
        plt.show()


main()
