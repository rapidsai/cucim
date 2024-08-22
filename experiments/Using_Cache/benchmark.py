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

import numpy as np
import rasterio
from openslide import OpenSlide
from rasterio.windows import Window

from cucim import CuImage


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
        print(f"{self.message} : {self.end - self.start}")


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


identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def load_tile_rasterio(slide, start_loc, tile_size):
    _ = np.moveaxis(
        slide.read(
            [1, 2, 3],
            window=Window.from_slices(
                (start_loc[0], start_loc[0] + tile_size),
                (start_loc[1], start_loc[1] + tile_size),
            ),
        ),
        0,
        -1,
    )


def load_tile_rasterio_chunk(input_file, start_loc_list, patch_size):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    slide = rasterio.open(input_file, transform=identity, num_threads=1)
    for start_loc in start_loc_list:
        _ = np.moveaxis(
            slide.read(
                [1, 2, 3],
                window=Window.from_slices(
                    (start_loc[0], start_loc[0] + patch_size),
                    (start_loc[1], start_loc[1] + patch_size),
                ),
            ),
            0,
            -1,
        )


def load_tile_openslide_chunk_mp(inp_file, start_loc_list, patch_size):
    with OpenSlide(inp_file) as slide:
        for start_loc in start_loc_list:
            _ = slide.read_region(start_loc, 0, [patch_size, patch_size])


def load_tile_cucim_chunk_mp(inp_file, start_loc_list, patch_size):
    slide = CuImage(inp_file)
    for start_loc in start_loc_list:
        _ = slide.read_region(start_loc, [patch_size, patch_size], 0)


def load_tile_rasterio_chunk_mp(input_file, start_loc_list, patch_size):
    slide = rasterio.open(input_file, num_threads=1)
    for start_loc in start_loc_list:
        _ = np.moveaxis(
            slide.read(
                [1, 2, 3],
                window=Window.from_slices(
                    (start_loc[0], start_loc[0] + patch_size),
                    (start_loc[1], start_loc[1] + patch_size),
                ),
            ),
            0,
            -1,
        )


def experiment_thread(
    cache_strategy, input_file, num_threads, start_location, patch_size
):
    import psutil

    print("  ", psutil.virtual_memory())
    # range(1, num_threads + 1): # (num_threads,):
    for num_workers in range(1, num_threads + 1):
        openslide_time = 1
        cucim_time = 1
        rasterio_time = 1

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

        with Timer("  Thread elapsed time (rasterio)") as timer:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(
                    load_tile_rasterio_chunk,
                    repeat(input_file),
                    start_loc_list_iter,
                    repeat(patch_size),
                )
            rasterio_time = timer.elapsed_time()

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
    for num_workers in range(1, num_processes + 1):
        openslide_time = 1
        cucim_time = 1
        rasterio_time = 1

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

        with Timer("  Process elapsed time (rasterio)") as timer:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(
                    load_tile_rasterio_chunk_mp,
                    repeat(input_file),
                    start_loc_list_iter,
                    repeat(patch_size),
                )
            rasterio_time = timer.elapsed_time()

        print("  ", psutil.virtual_memory())
        output_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},process,{cache_strategy},{input_file},{start_location},{patch_size},{num_workers},{openslide_time},{cucim_time},{rasterio_time},{openslide_time / cucim_time},{rasterio_time / cucim_time},{cache_size},{cache.hit_count},{cache.miss_count}\n"  # noqa: E501
        with open("experiment.txt", "a+") as f:
            f.write(output_text)
        print(output_text)


experiment_thread("nocache", "notebooks/input/image.tif", 12, 0, 256)
experiment_process("nocache", "notebooks/input/image.tif", 12, 0, 256)
experiment_thread("per_process", "notebooks/input/image.tif", 12, 0, 256)
experiment_process("per_process", "notebooks/input/image.tif", 12, 0, 256)
experiment_thread("shared_memory", "notebooks/input/image.tif", 12, 0, 256)
experiment_process("shared_memory", "notebooks/input/image.tif", 12, 0, 256)

experiment_thread("nocache", "notebooks/input/image.tif", 12, 1, 256)
experiment_process("nocache", "notebooks/input/image.tif", 12, 1, 256)
experiment_thread("per_process", "notebooks/input/image.tif", 12, 1, 256)
experiment_process("per_process", "notebooks/input/image.tif", 12, 1, 256)
experiment_thread("shared_memory", "notebooks/input/image.tif", 12, 1, 256)
experiment_process("shared_memory", "notebooks/input/image.tif", 12, 1, 256)

experiment_thread("nocache", "notebooks/input/image2.tif", 12, 0, 256)
experiment_process("nocache", "notebooks/input/image2.tif", 12, 0, 256)
experiment_thread("per_process", "notebooks/input/image2.tif", 12, 0, 256)
experiment_process("per_process", "notebooks/input/image2.tif", 12, 0, 256)
experiment_thread("shared_memory", "notebooks/input/image2.tif", 12, 0, 256)
experiment_process("shared_memory", "notebooks/input/image2.tif", 12, 0, 256)

experiment_thread("nocache", "notebooks/input/image2.tif", 12, 1, 256)
experiment_process("nocache", "notebooks/input/image2.tif", 12, 1, 256)
experiment_thread("per_process", "notebooks/input/image2.tif", 12, 1, 256)
experiment_process("per_process", "notebooks/input/image2.tif", 12, 1, 256)
experiment_thread("shared_memory", "notebooks/input/image2.tif", 12, 1, 256)
experiment_process("shared_memory", "notebooks/input/image2.tif", 12, 1, 256)

experiment_thread("nocache", "notebooks/0486052bb.tiff", 12, 0, 1024)
experiment_process("nocache", "notebooks/0486052bb.tiff", 12, 0, 1024)
experiment_thread("per_process", "notebooks/0486052bb.tiff", 12, 0, 1024)
experiment_process("per_process", "notebooks/0486052bb.tiff", 12, 0, 1024)
experiment_thread("shared_memory", "notebooks/0486052bb.tiff", 12, 0, 1024)
experiment_process("shared_memory", "notebooks/0486052bb.tiff", 12, 0, 1024)

experiment_thread("nocache", "notebooks/0486052bb.tiff", 12, 1, 1024)
experiment_process("nocache", "notebooks/0486052bb.tiff", 12, 1, 1024)
experiment_thread("per_process", "notebooks/0486052bb.tiff", 12, 1, 1024)
experiment_process("per_process", "notebooks/0486052bb.tiff", 12, 1, 1024)
experiment_thread("shared_memory", "notebooks/0486052bb.tiff", 12, 1, 1024)
experiment_process("shared_memory", "notebooks/0486052bb.tiff", 12, 1, 1024)
