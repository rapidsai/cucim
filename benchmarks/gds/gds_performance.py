import os
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
        self.end = None

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


samples = 50
data_folder = "."

file_size_list = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8",
                  "0.9", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]

cp_arr = cp.zeros(2**30 * 10, dtype=cp.uint8)  # 10 GB

wfp = open("gds_performance.csv", "a")

for file_size_item in file_size_list:
    file_name = f"{data_folder}/data_{file_size_item}.blob"
    file_size = os.path.getsize(file_name)

    print(f"## File Size : {file_size_item} GB ({file_size})")

    # kvikio
    if KVIKIO_AVAILABLE:
        time_sum = 0
        for count in range(samples):
            fs.discard_page_cache(file_name)
            with Timer(f"  Read {file_name} with kvikio") as timer:
                fd = kvikio.CuFile(file_name, "r")
                read_count = fd.read(cp_arr, file_size, 0)
                fd.close()
                elapsed_time = timer.elapsed_time()
                time_sum += elapsed_time
            output_text = f"{file_size_item}, kvikio, {count}, {elapsed_time}\n"
            wfp.write(output_text)
        output_text = f"{file_size_item}, kvikio, avg_{samples}, {time_sum / samples}\n"
        wfp.write(output_text)

    # GDS
    time_sum = 0
    for count in range(samples):
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with GDS") as timer:
            fd = fs.open(file_name, "r")
            read_count = fd.pread(cp_arr, file_size, 0, 0)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, GDS, {count}, {elapsed_time}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, GDS, avg_{samples}, {time_sum / samples}\n"
    wfp.write(output_text)

    # POSIX
    time_sum = 0
    for count in range(samples):
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with Posix") as timer:
            fd = fs.open(file_name, "rnp")
            read_count = fd.pread(cp_arr, file_size, 0)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, Posix, {count}, {elapsed_time}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, Posix, avg_{samples}, {time_sum / samples}\n"
    wfp.write(output_text)

    # Posix+ODIRECT
    time_sum = 0
    for count in range(samples):
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with Posix+ODIRECT") as timer:
            fd = fs.open(file_name, "rp")
            read_count = fd.pread(cp_arr, file_size, 0)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, Posix+ODIRECT, {count}, {elapsed_time}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, Posix+ODIRECT, avg_{samples}, {time_sum / samples}\n"
    wfp.write(output_text)

    # MMAP
    time_sum = 0
    for count in range(samples):
        fs.discard_page_cache(file_name)
        with Timer(f"  Read {file_name} with MMAP") as timer:
            fd = fs.open(file_name, "rm")
            read_count = fd.pread(cp_arr, file_size, 0)
            fs.close(fd)
            elapsed_time = timer.elapsed_time()
            time_sum += elapsed_time
        output_text = f"{file_size_item}, MMAP, {count}, {elapsed_time}\n"
        wfp.write(output_text)
    output_text = f"{file_size_item}, MMAP, avg_{samples}, {time_sum / samples}\n"
    wfp.write(output_text)
wfp.close()
