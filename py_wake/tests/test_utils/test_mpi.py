import time
import numpy as np
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.flow_map import XYGrid
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


if MPI:
    TOLS = {"rtol": 1e-6, "atol": 1e-6}

    def test_mpi_wind_farm_model():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        site = Hornsrev1Site()
        x, y = site.initial_position.T
        windTurbines = V80()
        wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

        np.random.seed(0)
        n = 2000
        use_time = False
        wss = np.random.uniform(0, 15, n) if use_time else None
        wds = np.random.uniform(0, 360, n) if use_time else None
        if rank == 0:
            start = time.time()
        aep = wf_model(x=x, y=y, ws=wss, wd=wds, time=use_time, n_cpu=4).aep()
        jac_full = (
            wf_model.aep_gradients(x=x, y=y, ws=wss, wd=wds, time=use_time, n_cpu=4) * 1e6
        )  # fmt: skip
        if rank == 0:
            elapsed = time.time() - start
            print(f"WFM AEP+Grad runtime: {elapsed:.4f} seconds")

            # serial run
            aep_reff = wf_model(x=x, y=y, ws=wss, wd=wds, time=use_time, n_cpu=1).aep()
            jac_reff = (
                wf_model.aep_gradients(x=x, y=y, ws=wss, wd=wds, time=use_time, n_cpu=1) * 1e6
            )  # fmt: skip

            np.testing.assert_allclose(aep.sum().values, aep_reff.sum().values, **TOLS)
            np.testing.assert_allclose(jac_full, jac_reff, **TOLS)

    def test_mpi_flow_map():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        site = Hornsrev1Site()
        x, y = site.initial_position.T
        windTurbines = V80()
        wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)
        sim_res = wf_model(x=x, y=y, wd=np.arange(0, 360, 30), ws=[9, 10, 11])
        grid = XYGrid(
            x=np.linspace(x.min() - 500, x.max() + 500, 100),
            y=np.linspace(y.min() - 500, y.max() + 500, 100),
        )
        if rank == 0:
            start = time.time()
        # MPI parallel flow map (n_cpu > 1 enables MPI when running with mpirun)
        flow_map_mpi = sim_res.flow_map(grid=grid, n_cpu=4)
        if rank == 0:
            elapsed = time.time() - start
            print(f"WFM flow_map runtime: {elapsed:.4f} seconds")

            flow_map_serial = sim_res.flow_map(grid=grid, n_cpu=1)
            np.testing.assert_allclose(
                flow_map_mpi.WS_eff.values, flow_map_serial.WS_eff.values, **TOLS
            )
            np.testing.assert_allclose(
                flow_map_mpi.TI_eff.values, flow_map_serial.TI_eff.values, **TOLS
            )

    if __name__ == "__main__":
        test_mpi_wind_farm_model()
        test_mpi_flow_map()
