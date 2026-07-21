import atexit
import gc
import multiprocessing
import platform
from itertools import starmap


def load_MPI():
    try:  # pragma: no cover
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        assert comm.Get_size() > 1
    except (ImportError, AssertionError):
        MPI = None
    return MPI


pool_dict = {}


def get_pool(processes=multiprocessing.cpu_count()):
    if processes not in pool_dict:
        # close pools
        for pool in pool_dict.values():
            pool.close()
        pool_dict.clear()

        if platform.system() == 'Darwin':  # pragma: no cover
            pool_dict[processes] = multiprocessing.get_context('fork').Pool(processes)
        else:
            pool_dict[processes] = multiprocessing.Pool(processes)
    return pool_dict[processes]


# class gc_func():
#     seems to have no effect
#     def __init__(self, func):
#         self.func = func
#
#     def __call__(self, *args, **kwargs):
#         import time
#         t = time.perf_counter()
#         r = self.func(*args, **kwargs)
#         print('f', time.perf_counter() - t)
#         gc.collect()
#         print('gc', time.perf_counter() - t)
#         return r


def get_pool_map(n_cpu=multiprocessing.cpu_count(), starmap=False, verbose=True):
    pool = get_pool(n_cpu)

    def pool_map(func, iterable, chunksize=None):
        iterable = get_tqdm_iterable(iterable, verbose=verbose)
        if starmap:
            return pool.starmap(func, iterable, chunksize)
        else:
            return pool.map(func, iterable, chunksize)
    return pool_map


def close_pools():  # pragma: no cover
    for k, pool in pool_dict.items():
        pool.close()


def get_n_cpu(n_cpu):
    MPI = load_MPI()
    if MPI is None:
        n_cpu = n_cpu or multiprocessing.cpu_count()
    else:  # pragma: no cover
        comm = MPI.COMM_WORLD
        n_cpu = min(n_cpu or comm.Get_size(), comm.Get_size())
    return n_cpu


def get_map_func(n_cpu, starmap=False, verbose=True, desc='', unit='it', leave=True):
    if n_cpu is None:
        n_cpu = get_n_cpu(n_cpu)

    if n_cpu > 1:
        n_cpu = get_n_cpu(n_cpu)
        MPI = load_MPI()
        if MPI and n_cpu > 1:  # pragma: no cover
            map_func = get_mpi_map_func(n_cpu, starmap=starmap)
        else:
            map_func = get_pool_map(n_cpu, starmap=starmap, verbose=verbose)
    else:
        if starmap:
            from itertools import starmap
            mapf = starmap
        else:
            mapf = map

        def map_func(f, iter):
            return mapf(f, get_tqdm_iterable(iter, desc=desc, unit=unit, verbose=verbose, leave=leave))
    return map_func


def get_tqdm_iterable(iterable, desc='', unit='it', total=None, verbose=True, leave=True):
    from tqdm import tqdm
    total = getattr(iterable, '__len__', lambda: total)()
    return tqdm(iterable, desc=desc, unit=unit, total=total, disable=not verbose, leave=leave)


atexit.register(close_pools)


def get_mpi_map_func(n_cpu, starmap=False):  # pragma: no cover
    MPI = load_MPI()
    assert MPI is not None, "mpi4py is not installed"
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def mpi_map_func(f, iter_args):
        iter_args = list(iter_args)
        # Distribute work across MPI ranks
        local_args = [iter_args[i] for i in range(rank, len(iter_args), n_cpu)]
        local_results = (
            [f(arg) for arg in local_args]
            if not starmap
            else [f(*arg) for arg in local_args]
        )
        # Gather results from all ranks
        all_results = comm.allgather(local_results)
        # Reconstruct results in original order
        results = [None] * len(iter_args)
        for r, res_list in enumerate(all_results):
            for idx, res in enumerate(res_list):
                results[r + idx * n_cpu] = res
        return results

    return mpi_map_func
