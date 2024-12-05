import multiprocessing
import atexit
import platform
import gc
import os
from functools import partial
import numpy as np

pool_dict = {}

def get_pool(processes=multiprocessing.cpu_count()):
    """Get or create a process pool with the specified number of processes"""
    if processes not in pool_dict:
        # Close existing pools before creating new one
        for pool in pool_dict.values():
            pool.close()
        pool_dict.clear()

        # Use fork on Darwin for better compatibility
        if platform.system() == 'Darwin':  # pragma: no cover
            pool_dict[processes] = multiprocessing.get_context('fork').Pool(processes)
        else:
            pool_dict[processes] = multiprocessing.Pool(processes)
    return pool_dict[processes]

def get_distributed_map(backend='multiprocessing', processes=None):
    """Get appropriate mapping function based on backend
    
    Parameters
    ----------
    backend : str
        'multiprocessing', 'mpi', or 'ray'
    processes : int
        Number of processes to use
    
    Returns
    -------
    map_func : callable
        Mapping function that distributes work across processes
    """
    if backend == 'multiprocessing':
        return get_pool_map(processes)
    
    elif backend == 'mpi':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        def mpi_map(func, iterable, chunksize=None):
            if rank == 0:
                # Calculate chunk sizes for even distribution
                n_items = len(iterable)
                chunk_sizes = [n_items // size + (1 if i < n_items % size else 0)
                             for i in range(size)]
                chunks = []
                start = 0
                for chunk_size in chunk_sizes:
                    chunks.append(iterable[start:start + chunk_size])
                    start += chunk_size
            else:
                chunks = None
                
            local_chunk = comm.scatter(chunks, root=0)
            local_result = list(map(gc_func(func), local_chunk))
            
            results = comm.gather(local_result, root=0)
            if rank == 0:
                # Flatten results
                return [item for sublist in results for item in sublist]
            return None
            
        return mpi_map
        
    elif backend == 'ray':
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=processes)
            
        def ray_map(func, iterable, chunksize=None):
            # Determine optimal chunk size if not specified
            if chunksize is None:
                chunksize = max(1, len(iterable) // (processes * 4))
                
            # Create chunks
            chunks = [iterable[i:i + chunksize] 
                     for i in range(0, len(iterable), chunksize)]
            
            @ray.remote
            def process_chunk(chunk):
                return [gc_func(func)(x) for x in chunk]
            
            # Submit chunks to Ray
            futures = [process_chunk.remote(chunk) for chunk in chunks]
            
            # Get results and flatten
            chunk_results = ray.get(futures)
            return [item for chunk in chunk_results for item in chunk]
            
        return ray_map
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

class gc_func:
    """Wrapper to ensure garbage collection after function execution"""
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        r = self.func(*args, **kwargs)
        gc.collect()
        return r

def get_pool_map(processes=multiprocessing.cpu_count()):
    """Get a pool.map function that uses specified number of processes"""
    pool = get_pool(processes)
    
    def gc_map(func, iterable, chunksize=None):
        return pool.map(gc_func(func), iterable, chunksize)
    return gc_map

def get_pool_starmap(processes=multiprocessing.cpu_count()):
    """Get a pool.starmap function that uses specified number of processes"""
    pool = get_pool(processes)
    
    def gc_map(func, iterable, chunksize=None):
        return pool.starmap(gc_func(func), iterable, chunksize)
    return gc_map

def close_pools():
    """Close all process pools"""
    for pool in pool_dict.values():
        pool.close()
        pool.terminate()

atexit.register(close_pools)
