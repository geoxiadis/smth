import numpy as np
import xarray as xr
import yaml

from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.site import XRSite
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import CrespoHernandez
from py_wake.wind_farm_models import All2AllIterative

wind_resource = yaml.load(
    open("wr.yaml"),
    Loader=yaml.FullLoader,
)["wind_resource"]
A = wind_resource["weibull_a"]
k = wind_resource["weibull_k"]
freq = wind_resource["sector_probability"]
wd = np.array(wind_resource["wind_direction"])
ws = np.array(wind_resource["wind_speed"])
TI = wind_resource["turbulence_intensity"]["data"]
SITE = XRSite(
    ds=xr.Dataset(
        data_vars={
            "Sector_frequency": ("wd", freq["data"]),
            "Weibull_A": ("wd", A["data"]),
            "Weibull_k": ("wd", k["data"]),
            "TI": ("ws", TI),
        },
        coords={"wd": wd, "ws": ws},
    )
)
WTURBINE = DTU10MW()

params = {
    "A": 0.28169340293751577,
    "ceps": 0.1420674010683264,
    "cti1": 1.2015732556452348,
    "cti2": 3.148364906110177,
    "ctlim": 0.7750556898490065,
}
turbulenceModel = CrespoHernandez()
def_args = {
    "A": params["A"],
    "cTI": [params["cti1"], params["cti2"]],
    "ceps": params["ceps"],
    "ctlim": params["ctlim"],
    "ct2a": ct2a_mom1d,
    "rotorAvgModel": GaussianOverlapAvgModel(),
}
wake_deficitModel = TurboGaussianDeficit(**def_args)
wake_deficitModel.WS_key = "WS_jlk"

wfm = All2AllIterative(
    SITE,
    WTURBINE,
    wake_deficitModel=wake_deficitModel,
    blockage_deficitModel=SelfSimilarityDeficit2020(),
    turbulenceModel=turbulenceModel,
    superpositionModel=LinearSum(),
    use_aa=True,
)

nx = 15
ny = 7
np.random.seed(42)  # For reproducibility
x = []
y = []
min_distance = WTURBINE.diameter() * 5

while len(x) < nx * ny:
    candidate_x = np.random.uniform(
        -WTURBINE.diameter() * 3 * nx, WTURBINE.diameter() * 3 * nx
    )
    candidate_y = np.random.uniform(
        -WTURBINE.diameter() * 3 * ny, WTURBINE.diameter() * 3 * ny
    )

    if all(
        np.sqrt((candidate_x - xi) ** 2 + (candidate_y - yi) ** 2) >= min_distance
        for xi, yi in zip(x, y)
    ):
        x.append(candidate_x)
        y.append(candidate_y)

x = np.array(x)
y = np.array(y)

import time

s = time.time()
sim_res = wfm(x, y, ws=ws, wd=wd, verbose=True, n_cpu=4)
aep = sim_res.aep().sum()
print("Time taken:", time.time() - s)
print("AEP:", aep)

flow_map = sim_res.flow_map(wd=270, ws=10.0)
flow_map.plot_wake_map()
import matplotlib.pyplot as plt

plt.show()
