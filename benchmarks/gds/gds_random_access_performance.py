import os
import numpy as np
import cupy as cp

from contextlib import ContextDecorator
from time import perf_counter

import cucim.clara.filesystem as fs

KVIKIO_AVAILABLE = True
try:
    import kvikio
except ImportError:
    KVIKIO_AVAILABLE = False


class Timer(ContextDecorator):
    def __init__(self, message):
        self.message = message

    def elapsed_time(self):
        self.end = perf_counter()
        return self.end - self.start

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if not self.end:
            self.elapsed_time()
        print("{} : {}".format(self.message, self.end - self.start))


samples = 5
data_folder = "."
seed = 12345
rng = np.random.default_rng(seed)

cp_arr = cp.zeros(2**30 * 10, dtype=cp.uint8)  # 10 GB

filesize = 2
patch_size = 2**14  # 16384
offsets = np.arange(0, 2**30*filesize - 2 * patch_size, patch_size)
offsets_size = offsets.shape[0]

# Shuffle offsets
rng.shuffle(offsets)
delta = rng.integers(0, patch_size, size=offsets_size)
offsets = offsets + delta

wfp = open("gds_performance_random.csv", "a")

file_size_list = [f"{filesize}.0"]
for file_size_item in file_size_list:
    file_name = f"{data_folder}/data_{file_size_item}.blob"
    file_size = os.path.getsize(file_name)

    print(f"## File Size : {file_size_item} GB ({file_size})")

    # kvikio
    if KVIKIO_AVAILABLE:
        time_sum = 0
        for count in range(samples):
            offset_list = np.copy(offsets)
            fs.discard_page_cache(file_name)
            with Timer(f"  Read {file_name} with kvikio") as timer:
                fd = kvikio.CuFile(file_name, "r")
                for offset in offset_list:
                    read_count = fd.read(cp_arr, patch_size, offset)
                fd.close()
                elapsed_time = timer.elapsed_time()
                time_sum += elapsed_time
            output_text = f"{file_size_item}, kvikio, {count}, {elapsed_time}, {patch_size}\n"
            wfp.write(output_text)
        output_text = f"{file_size_item}, kvikio, avg_{samples}, {time_sum / samples}, {patch_size}\n"
        wfp.write(output_text)

    # GDS
    time_sum = 0
    for count in range(samples):
        offset_list = np.copy(offsets)
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with GDS") as timer:
            fd = fs.open(file_name, "r")
            for offset in offset_list:
                read_count = fd.pread(cp_arr, patch_size, offset)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, GDS, {count}, {elapsed_time}, {patch_size}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, GDS, avg_{samples}, {time_sum / samples}, {patch_size}\n"
    wfp.write(output_text)

    # POSIX
    time_sum = 0
    for count in range(samples):
        offset_list = np.copy(offsets)
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with Posix") as timer:
            fd = fs.open(file_name, "rnp")
            for offset in offset_list:
                read_count = fd.pread(cp_arr, patch_size, offset)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, Posix, {count}, {elapsed_time}, {patch_size}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, Posix, avg_{samples}, {time_sum / samples}, {patch_size}\n"
    wfp.write(output_text)

    # Posix+ODIRECT
    time_sum = 0
    for count in range(samples):
        offset_list = np.copy(offsets)
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with Posix+ODIRECT") as timer:
            fd = fs.open(file_name, "rp")
            for offset in offset_list:
                read_count = fd.pread(cp_arr, patch_size, offset)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, Posix+ODIRECT, {count}, {elapsed_time}, {patch_size}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, Posix+ODIRECT, avg_{samples}, {time_sum / samples}, {patch_size}\n"
    wfp.write(output_text)

    # MMAP
    time_sum = 0
    for count in range(samples):
        offset_list = np.copy(offsets)
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with MMAP") as timer:
            fd = fs.open(file_name, "rm")
            for offset in offset_list:
                read_count = fd.pread(cp_arr, patch_size, offset)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, MMAP, {count}, {elapsed_time}, {patch_size}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, MMAP, avg_{samples}, {time_sum / samples}, {patch_size}\n"
    wfp.write(output_text)
wfp.close()
