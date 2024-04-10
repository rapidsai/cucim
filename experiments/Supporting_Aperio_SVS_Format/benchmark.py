#
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import concurrent.futures
from contextlib import ContextDecorator
from datetime import datetime
from itertools import repeat
from time import perf_counter

from cucim import CuImage
from cucim.clara.filesystem import discard_page_cache  # noqa: F401
from openslide import OpenSlide


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


def load_tile_openslide(slide, start_loc, patch_size):
    _ = slide.read_region(start_loc, 0, [patch_size, patch_size])


def load_tile_openslide_chunk(inp_file, start_loc_list, patch_size):
    with OpenSlide(inp_file) as slide:
        for start_loc in start_loc_list:
            _ = slide.read_region(start_loc, 0, [patch_size, patch_size])


def load_tile_cucim(slide, start_loc, patch_size):
    _ = slide.read_region(start_loc, [patch_size, patch_size], 0)


def load_tile_cucim_chunk(inp_file, start_loc_list, patch_size):
    try:
        slide = CuImage(inp_file)
        for start_loc in start_loc_list:
            _ = slide.read_region(start_loc, [patch_size, patch_size], 0)
    except Exception as e:
        print(e)


def load_tile_openslide_chunk_mp(inp_file, start_loc_list, patch_size):
    with OpenSlide(inp_file) as slide:
        for start_loc in start_loc_list:
            _ = slide.read_region(start_loc, 0, [patch_size, patch_size])


def load_tile_cucim_chunk_mp(inp_file, start_loc_list, patch_size):
    slide = CuImage(inp_file)
    for start_loc in start_loc_list:
        _ = slide.read_region(start_loc, [patch_size, patch_size], 0)


def experiment_thread(
    cache_strategy, input_file, num_threads, start_location, patch_size
):
    import psutil

    print("  ", psutil.virtual_memory())
    for num_workers in (1, 3, 6, 9, 12):  # range(1, num_threads + 1):
        openslide_time = 1
        cucim_time = 1
        rasterio_time = 1

        # discard_page_cache(input_file)
        with OpenSlide(input_file) as slide:
            width, height = slide.dimensions

            start_loc_data = [
                (sx, sy)
                for sy in range(start_location, height, patch_size)
                for sx in range(start_location, width, patch_size)
            ]
            chunk_size = len(start_loc_data) // num_workers
            start_loc_list_iter = [
                start_loc_data[i : i + chunk_size]
                for i in range(0, len(start_loc_data), chunk_size)
            ]
            with Timer("  Thread elapsed time (OpenSlide)") as timer:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    executor.map(
                        load_tile_openslide_chunk,
                        repeat(input_file),
                        start_loc_list_iter,
                        repeat(patch_size),
                    )
                openslide_time = timer.elapsed_time()
        print("  ", psutil.virtual_memory())

        cache_size = psutil.virtual_memory().available // 1024 // 1024 // 20
        cache = CuImage.cache(
            cache_strategy, memory_capacity=cache_size, record_stat=True
        )
        cucim_time = 0
        # discard_page_cache(input_file)
        slide = CuImage(input_file)
        start_loc_data = [
            (sx, sy)
            for sy in range(start_location, height, patch_size)
            for sx in range(start_location, width, patch_size)
        ]
        chunk_size = len(start_loc_data) // num_workers
        start_loc_list_iter = [
            start_loc_data[i : i + chunk_size]
            for i in range(0, len(start_loc_data), chunk_size)
        ]
        with Timer("  Thread elapsed time (cuCIM)") as timer:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(
                    load_tile_cucim_chunk,
                    repeat(input_file),
                    start_loc_list_iter,
                    repeat(patch_size),
                )
            cucim_time = timer.elapsed_time()
        print(f"  hit: {cache.hit_count}   miss: {cache.miss_count}")
        print("  ", psutil.virtual_memory())

        output_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},thread,{cache_strategy},{input_file},{start_location},{patch_size},{num_workers},{openslide_time},{cucim_time},{rasterio_time},{openslide_time / cucim_time},{rasterio_time / cucim_time},{cache_size},{cache.hit_count},{cache.miss_count}\n"  # noqa: E501
        with open("experiment.txt", "a+") as f:
            f.write(output_text)
        print(output_text)


def experiment_process(
    cache_strategy, input_file, num_processes, start_location, patch_size
):
    import psutil

    print("  ", psutil.virtual_memory())
    for num_workers in (1, 3, 6, 9, 12):  # range(1, num_processes + 1):
        openslide_time = 1
        cucim_time = 1
        rasterio_time = 1

        # discard_page_cache(input_file)
        with OpenSlide(input_file) as slide:
            width, height = slide.dimensions

            start_loc_data = [
                (sx, sy)
                for sy in range(start_location, height, patch_size)
                for sx in range(start_location, width, patch_size)
            ]
            chunk_size = len(start_loc_data) // num_workers
            start_loc_list_iter = [
                start_loc_data[i : i + chunk_size]
                for i in range(0, len(start_loc_data), chunk_size)
            ]

            with Timer("  Process elapsed time (OpenSlide)") as timer:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    executor.map(
                        load_tile_openslide_chunk_mp,
                        repeat(input_file),
                        start_loc_list_iter,
                        repeat(patch_size),
                    )
                openslide_time = timer.elapsed_time()
        print("  ", psutil.virtual_memory())

        cache_size = psutil.virtual_memory().available // 1024 // 1024 // 20
        if cache_strategy == "shared_memory":
            cache_size = cache_size * num_workers
        cache = CuImage.cache(
            cache_strategy, memory_capacity=cache_size, record_stat=True
        )
        cucim_time = 0
        # discard_page_cache(input_file)
        slide = CuImage(input_file)
        start_loc_data = [
            (sx, sy)
            for sy in range(start_location, height, patch_size)
            for sx in range(start_location, width, patch_size)
        ]
        chunk_size = len(start_loc_data) // num_workers
        start_loc_list_iter = [
            start_loc_data[i : i + chunk_size]
            for i in range(0, len(start_loc_data), chunk_size)
        ]

        with Timer("  Process elapsed time (cuCIM)") as timer:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(
                    load_tile_cucim_chunk_mp,
                    repeat(input_file),
                    start_loc_list_iter,
                    repeat(patch_size),
                )
            cucim_time = timer.elapsed_time()
        print("  ", psutil.virtual_memory())

        rasterio_time = 0
        start_loc_data = [
            (sx, sy)
            for sy in range(start_location, height, patch_size)
            for sx in range(start_location, width, patch_size)
        ]
        chunk_size = len(start_loc_data) // num_workers
        start_loc_list_iter = [
            start_loc_data[i : i + chunk_size]
            for i in range(0, len(start_loc_data), chunk_size)
        ]

        print("  ", psutil.virtual_memory())
        output_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},process,{cache_strategy},{input_file},{start_location},{patch_size},{num_workers},{openslide_time},{cucim_time},{rasterio_time},{openslide_time / cucim_time},{rasterio_time / cucim_time},{cache_size},{cache.hit_count},{cache.miss_count}\n"  # noqa: E501
        with open("experiment.txt", "a+") as f:
            f.write(output_text)
        print(output_text)


for i in range(10):
    experiment_thread(
        "per_process", "notebooks/input/TUPAC-TR-488.svs", 12, 120, 240
    )
    experiment_thread(
        "per_process", "notebooks/input/JP2K-33003-2.svs", 12, 128, 256
    )
    experiment_thread(
        "per_process", "notebooks/input/CMU-1-JP2K-33005.svs", 12, 120, 240
    )
    experiment_process(
        "per_process", "notebooks/input/TUPAC-TR-488.svs", 12, 120, 240
    )
    experiment_process(
        "per_process", "notebooks/input/JP2K-33003-2.svs", 12, 128, 256
    )
    experiment_process(
        "per_process", "notebooks/input/CMU-1-JP2K-33005.svs", 12, 120, 240
    )
