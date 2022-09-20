import cupy as cp

from subprocess import Popen, PIPE
import xml.etree.ElementTree as ET


def get_gpu_info(*args, **kwargs):
    p = Popen(["nvidia-smi", "-q", "-x"], stdout=PIPE)
    outs, errors = p.communicate()

    xml = ET.fromstring(outs)
    driver_version = xml.findall("driver_version")[0].text
    cuda_version = xml.findall("cuda_version")[0].text
    datas = []
    for gpu_id, gpu in enumerate(xml.iter("gpu")):
        gpu_data = {}
        name = [x for x in gpu.iter("product_name")][0].text
        memory_usage = gpu.findall("fb_memory_usage")[0]
        total_memory = memory_usage.findall("total")[0].text

        gpu_data["name"] = name
        gpu_data["total_memory"] = total_memory
        gpu_data["driver_version"] = driver_version
        gpu_data["cuda_version"] = cuda_version
        datas.append(gpu_data)
    return datas[0]


gpu_info = get_gpu_info()

gpu_name = gpu_info['name']


mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


def print_gpu_mem():
    print("%7s %7s" % ('Used', 'Total'))
    fmt = "%4d MB %4d MB"
    print(fmt % (mempool.used_bytes() / 1024**2, mempool.total_bytes() / 1024**2))


def free_gpu_mem(verbose=True):
    if verbose:
        print_gpu_mem()
        print("Free all blocks")
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    if verbose:
        print_gpu_mem()
