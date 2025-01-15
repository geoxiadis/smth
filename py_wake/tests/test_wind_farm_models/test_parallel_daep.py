import numpy.testing as npt
from py_wake.utils.gradients import cs
from py_wake.examples.data.hornsrev1 import V80
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.site.xrsite import XRSite
from py_wake.utils.gradients import autograd
import numpy as np
import xarray as xr
import numpy as np  # fmt: skip
np.random.seed(42)


def get_xrsite_ds():
    x_dim, y_dim, h_dim, time_dim = 100, 40, 8, 48
    h = np.array([30.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0])
    x = np.linspace(6.947e5, 8.932e5, x_dim)
    y = np.linspace(8.449e5, 9.225e5, y_dim)
    time = np.arange(48)
    WS = np.random.uniform(3.0, 15.0, size=(y_dim, x_dim, h_dim, time_dim))
    WD = np.random.uniform(250.0, 300.0, size=(y_dim, x_dim, h_dim, time_dim))
    P = 1.0
    TI = 0.06
    ds = xr.Dataset(
        {
            "WS": (["y", "x", "h", "time"], WS),
            "WD": (["y", "x", "h", "time"], WD),
            "P": ([], P),
            "TI": ([], TI),
        },
        coords={
            "h": (["h"], h),
            "x": (["x"], x),
            "y": (["y"], y),
            "time": (["time"], time),
        },
    )
    return ds


def test_parallel_daep_with_time_site():
    ds = get_xrsite_ds()
    ds = ds.transpose("x", "y", "h", "time")
    T = np.arange(len(ds.time))
    site = XRSite(ds)
    wfm = Nygaard_2022(site, V80())

    test_x, test_y = [
        ds.x.mean().item(),
        ds.x.mean().item(),
        ds.x.mean().item(),
    ], [
        ds.y.mean().item(),
        ds.y.mean().item() + 250,
        ds.y.mean().item() + 500,
    ]

    def aep_func(x, y, n_cpu=1):  # full=False, **kwargs
        _f = np.zeros_like(T)
        sim_res = wfm(x, y, time=T, ws=_f, wd=_f, n_cpu=n_cpu)
        return sim_res.aep().sum().values

    def aep_jac(x, y, n_cpu=1, **kwargs):
        _f = np.zeros_like(T)
        jx, jy = wfm.aep_gradients(
            gradient_method=autograd,
            wrt_arg=["x", "y"],
            x=x,
            y=y,
            ws=_f,
            wd=_f,
            time=T,
            n_cpu=n_cpu,
        )
        return np.array([np.atleast_2d(jx), np.atleast_2d(jy)])

    aep1 = aep_func(test_x, test_y, n_cpu=1)
    aep2 = aep_func(test_x, test_y, n_cpu=2)
    npt.assert_almost_equal(
        aep1, aep2, decimal=6, err_msg="AEP is not the same for n_cpu=1 and n_cpu=2"
    )

    cs_grad1 = cs(aep_func, True, argnum=[0, 1])(test_x, test_y, n_cpu=1)
    ad_grad1 = aep_jac(test_x, test_y, n_cpu=1)
    npt.assert_almost_equal(
        np.stack(cs_grad1, axis=0).reshape(ad_grad1.shape),
        ad_grad1,
        decimal=6,
        err_msg="Gradient is not the same for n_cpu=1 compared to complex step",
    )

    cs_grad2 = cs(aep_func, True, argnum=[0, 1])(test_x, test_y, n_cpu=2)
    ad_grad2 = aep_jac(test_x, test_y, n_cpu=2)
    npt.assert_almost_equal(
        cs_grad1,
        cs_grad2,
        decimal=6,
    )
    npt.assert_almost_equal(
        ad_grad1,
        ad_grad2,
        decimal=6,
        err_msg="[AD] Gradient is not the same for n_cpu=1 and n_cpu=2",
    )
    npt.assert_almost_equal(
        np.stack(cs_grad2, axis=0).reshape(ad_grad2.shape),
        ad_grad2,
        decimal=6,
        err_msg="[CS vs AD] Gradient is not the same for n_cpu=1 and n_cpu=2",
    )
