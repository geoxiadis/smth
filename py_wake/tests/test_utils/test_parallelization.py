import time

import numpy as np
from memory_profiler import _get_memory

from py_wake.utils.parallelization import get_map_func, get_pool_map
from py_wake.utils.profiling import get_memory_usage, timeit


# def test_gc_function():
#     outcommented as the gcfunc seems to have no effect
#     from py_wake.utils.parallelization import gc_func
#
#     def f():
#         np.full((1, 1024**2, 128), 1.)  # allocate 1gb
#
#     mem_before = get_memory_usage()
#     gc_func(f)()
#
#     # assert memory increase is less than 5mb (on linux an increase occurs)
#     assert get_memory_usage() - mem_before < 5
#
#     mem_before = get_memory_usage()
#     f()
#
#     # assert memory increase is less than 5mb (on linux an increase occurs)
#     assert get_memory_usage() - mem_before < 5


def f(x):
    time.sleep(0.1)
    return x * 2


def test_get_map_func():
    map_func = get_map_func(n_cpu=1, starmap=True, verbose=False)
    r = list(map_func(lambda x, y: x + y, [(1, 2), (3, 4), (5, 6)]))
    assert r == [3, 7, 11]

    map_func = get_map_func(n_cpu=1, starmap=False, verbose=False)
    r = list(map_func(lambda x: x[0] + x[1], [(1, 2), (3, 4), (5, 6)]))
    assert r == [3, 7, 11]

    map_func = get_map_func(n_cpu=1, starmap=False, verbose=False)
    t = timeit(lambda x: list(map_func(f, x)))(np.arange(6))[1]
    assert np.min(t) > 0.6

    map_func = get_map_func(n_cpu=2, starmap=False, verbose=False)
    t = timeit(lambda x: list(map_func(f, x)), min_runs=3)(np.arange(6))[1]
    assert np.min(t) < 0.5

    map_func = get_map_func(n_cpu=None, starmap=False, verbose=False)
    t = timeit(lambda x: list(map_func(f, x)), min_runs=3)(np.arange(6))[1]
    assert np.min(t) < 0.5
