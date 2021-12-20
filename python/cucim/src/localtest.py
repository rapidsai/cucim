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

from rasterio.windows import Window
from openslide import OpenSlide
import rasterio
from itertools import repeat
from datetime import datetime
import concurrent.futures
import json
import os
import time
from contextlib import ContextDecorator
from time import perf_counter
from tifffile import TiffFile
import sys
import numpy as np
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
        print("{} : {}".format(self.message, self.end - self.start))


img = CuImage("notebooks/input/image.tif")

cache = CuImage.cache("per_process", memory_capacity=1024)

width, height = img.size("XY")
start_location = 1
patch_size = 224

start_loc_data = [(sx, sy)
                  for sy in range(start_location, height, patch_size)
                  for sx in range(start_location, width, patch_size)]
# start_loc_data = start_loc_data[len(
#     start_loc_data) // 2:len(start_loc_data)//2 + 100]\


with Timer("  Thread elapsed time (cuCIM)") as timer:
    region = img.read_region(device="cuda")
    # # a = img.read_region(num_workers=16)
    # # print(a)
    # batch_iter = img.read_region(
    #     iter(start_loc_data), (patch_size, patch_size), batch_size=2, device="cuda", num_workers=4)  # device="cuda",
    # d = next(batch_iter)
    # e = next(batch_iter)
    # next(batch_iter)
    # next(batch_iter)
    # next(batch_iter)
    # # c = 0
    # # for batch in batch_iter:
    # #     c += 1


sys.exit(0)


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
            region = slide.read_region(start_loc, 0, [patch_size, patch_size])


def load_tile_cucim(slide, start_loc, patch_size):
    _ = slide.read_region(start_loc, [patch_size, patch_size], 0)


def load_tile_cucim_chunk(inp_file, start_loc_list, patch_size):
    try:
        slide = CuImage(inp_file)
        for start_loc in start_loc_list:
            region = slide.read_region(start_loc, [patch_size, patch_size], 0)
    except Exception as e:
        print(e)


identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def load_tile_rasterio(slide, start_loc, tile_size):
    _ = np.moveaxis(slide.read([1, 2, 3],
                               window=Window.from_slices((start_loc[0], start_loc[0] + tile_size), (start_loc[1], start_loc[1] + tile_size))), 0, -1)


def load_tile_rasterio_chunk(input_file, start_loc_list, patch_size):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    slide = rasterio.open(input_file, transform=identity, num_threads=1)
    for start_loc in start_loc_list:
        _ = np.moveaxis(slide.read([1, 2, 3],
                                   window=Window.from_slices((start_loc[0], start_loc[0] + patch_size), (start_loc[1], start_loc[1] + patch_size))), 0, -1)


def load_tile_openslide_chunk_mp(inp_file, start_loc_list, patch_size):
    with OpenSlide(inp_file) as slide:
        for start_loc in start_loc_list:
            region = slide.read_region(start_loc, 0, [patch_size, patch_size])


def load_tile_cucim_chunk_mp(inp_file, start_loc_list, patch_size):
    slide = CuImage(inp_file)
    for start_loc in start_loc_list:
        region = slide.read_region(start_loc, [patch_size, patch_size], 0)


def load_tile_rasterio_chunk_mp(input_file, start_loc_list, patch_size):
    slide = rasterio.open(input_file, num_threads=1)
    for start_loc in start_loc_list:
        region = np.moveaxis(slide.read([1, 2, 3],
                                        window=Window.from_slices((start_loc[0], start_loc[0] + patch_size), (start_loc[1], start_loc[1] + patch_size))), 0, -1)


def experiment_thread(cache_strategy, input_file, num_threads, start_location, patch_size):
    import psutil
    print("  ", psutil.virtual_memory())
    # range(1, num_threads + 1): # range(1, num_threads + 1): # (num_threads,):
    for num_workers in (num_threads,):
        openslide_time = 1
        cucim_time = 1
        rasterio_time = 1

        with OpenSlide(input_file) as slide:
            width, height = slide.dimensions

        #     # start_loc_iter = ((w, h)
        #     #                   for h in range(start_location, height, patch_size)
        #     #                   for w in range(start_location, width, patch_size))
        #     # with Timer("  Thread elapsed time (OpenSlide)") as timer:
        #     #     with concurrent.futures.ThreadPoolExecutor(
        #     #         max_workers=num_workers
        #     #     ) as executor:
        #     #         executor.map(
        #     #             lambda start_loc: load_tile_openslide(
        #     #                 slide, start_loc, patch_size),
        #     #             start_loc_iter,
        #     #         )
        #     #     openslide_time = timer.elapsed_time()

        #     start_loc_data = [(sx, sy)
        #                       for sy in range(start_location, height, patch_size)
        #                       for sx in range(start_location, width, patch_size)]
        #     chunk_size = len(start_loc_data) // num_workers
        #     start_loc_list_iter = [start_loc_data[i:i + chunk_size]
        #                            for i in range(0, len(start_loc_data), chunk_size)]
        #     with Timer("  Thread elapsed time (OpenSlide)") as timer:
        #         with concurrent.futures.ThreadPoolExecutor(
        #             max_workers=num_workers
        #         ) as executor:
        #             executor.map(
        #                 load_tile_openslide_chunk,
        #                 repeat(input_file),
        #                 start_loc_list_iter,
        #                 repeat(patch_size)
        #             )
        #         openslide_time = timer.elapsed_time()
        # print("  ", psutil.virtual_memory())

        # # slide = CuImage(input_file)
        # # start_loc_iter = ((w, h)
        # #                   for h in range(start_location, height, patch_size)
        # #                   for w in range(start_location, width, patch_size))
        # # with Timer("  Thread elapsed time (cuCIM)") as timer:
        # #     with concurrent.futures.ThreadPoolExecutor(
        # #         max_workers=num_workers
        # #     ) as executor:
        # #         executor.map(
        # #             lambda start_loc: load_tile_cucim(slide, start_loc, patch_size),
        # #             start_loc_iter,
        # #         )
        # #     cucim_time = timer.elapsed_time()

        cache_size = psutil.virtual_memory().available // 1024 // 1024 // 20
        cache = CuImage.cache(
            cache_strategy, memory_capacity=cache_size, record_stat=True)
        cucim_time = 0
        slide = CuImage(input_file)
        start_loc_data = [(sx, sy)
                          for sy in range(start_location, height, patch_size)
                          for sx in range(start_location, width, patch_size)]
        chunk_size = len(start_loc_data) // num_workers
        start_loc_list_iter = [start_loc_data[i:i + chunk_size]
                               for i in range(0, len(start_loc_data), chunk_size)]

        # print(len(start_loc_data))

        with Timer("  Thread elapsed time (cuCIM)") as timer:
            batch_iter = slide.read_region(
                iter(start_loc_data), (patch_size, patch_size), batch_size=32, num_workers=num_workers)
            c = 0
            for batch in batch_iter:
                c += 1
            print(c)
            cucim_time = timer.elapsed_time()

        # with Timer("  Thread elapsed time (cuCIM)") as timer:
        #     with concurrent.futures.ThreadPoolExecutor(
        #         max_workers=num_workers
        #     ) as executor:
        #         executor.map(
        #             load_tile_cucim_chunk,
        #             repeat(input_file),
        #             start_loc_list_iter,
        #             repeat(patch_size)
        #         )
        #     cucim_time = timer.elapsed_time()
        print(f"  hit: {cache.hit_count}   miss: {cache.miss_count}")
        print("  ", psutil.virtual_memory())

        # start_loc_data = [(sx, sy)
        #                 for sy in range(start_location, height, patch_size)
        #                     for sx in range(start_location, width, patch_size)]
        # chunk_size = len(start_loc_data) // num_workers
        # start_loc_list_iter = [start_loc_data[i:i+chunk_size] for i in range(0, len(start_loc_data), chunk_size)]

        # with Timer("  Thread elapsed time (rasterio)") as timer:
        #     with concurrent.futures.ThreadPoolExecutor(
        #         max_workers=num_workers
        #     ) as executor:
        #         executor.map(
        #             load_tile_rasterio_chunk,
        #             repeat(input_file),
        #             start_loc_list_iter,
        #             repeat(patch_size)
        #         )
        #     rasterio_time = timer.elapsed_time()

        # rasterio_time = 0
        # slide = rasterio.open(input_file, num_threads=1)
        # start_loc_iter = ((w, h)
        #                   for h in range(start_location, height, patch_size)
        #                   for w in range(start_location, width, patch_size))
        # with Timer("  Thread elapsed time (rasterio)") as timer:
        #     with concurrent.futures.ThreadPoolExecutor(
        #         max_workers=num_workers
        #     ) as executor:
        #         executor.map(
        #             lambda start_loc: load_tile_rasterio(slide, start_loc, patch_size),
        #             start_loc_iter,
        #         )
        #     rasterio_time = timer.elapsed_time()
        print("  ", psutil.virtual_memory())
        output_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},thread,{cache_strategy},{input_file},{start_location},{patch_size},{num_workers},{openslide_time},{cucim_time},{rasterio_time},{openslide_time / cucim_time},{rasterio_time / cucim_time},{cache_size},{cache.hit_count},{cache.miss_count}\n"
        with open("experiment.txt", "a+") as f:
            f.write(output_text)
        print(output_text)


def experiment_process(cache_strategy, input_file, num_processes, start_location, patch_size):
    import psutil
    print("  ", psutil.virtual_memory())
    # range(1, num_processes + 1): #(num_processes,):
    for num_workers in range(1, num_processes + 1):
        openslide_time = 1
        cucim_time = 1
        rasterio_time = 1
        # (92344 x 81017)
        with OpenSlide(input_file) as slide:
            width, height = slide.dimensions

            start_loc_data = [(sx, sy)
                              for sy in range(start_location, height, patch_size)
                              for sx in range(start_location, width, patch_size)]
            chunk_size = len(start_loc_data) // num_workers
            start_loc_list_iter = [start_loc_data[i:i + chunk_size]
                                   for i in range(0, len(start_loc_data), chunk_size)]

            with Timer("  Process elapsed time (OpenSlide)") as timer:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    executor.map(
                        load_tile_openslide_chunk_mp,
                        repeat(input_file),
                        start_loc_list_iter,
                        repeat(patch_size)
                    )
                openslide_time = timer.elapsed_time()
        print("  ", psutil.virtual_memory())

        cache_size = psutil.virtual_memory().available // 1024 // 1024 // 20
        if cache_strategy == "shared_memory":
            cache_size = cache_size * num_workers
        cache = CuImage.cache(
            cache_strategy, memory_capacity=cache_size, record_stat=True)
        cucim_time = 0
        slide = CuImage(input_file)
        start_loc_data = [(sx, sy)
                          for sy in range(start_location, height, patch_size)
                          for sx in range(start_location, width, patch_size)]
        chunk_size = len(start_loc_data) // num_workers
        start_loc_list_iter = [start_loc_data[i:i + chunk_size]
                               for i in range(0, len(start_loc_data), chunk_size)]

        with Timer("  Process elapsed time (cuCIM)") as timer:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(
                    load_tile_cucim_chunk_mp,
                    repeat(input_file),
                    start_loc_list_iter,
                    repeat(patch_size)
                )
            cucim_time = timer.elapsed_time()
        print("  ", psutil.virtual_memory())

        rasterio_time = 0
        start_loc_data = [(sx, sy)
                          for sy in range(start_location, height, patch_size)
                          for sx in range(start_location, width, patch_size)]
        chunk_size = len(start_loc_data) // num_workers
        start_loc_list_iter = [start_loc_data[i:i + chunk_size]
                               for i in range(0, len(start_loc_data), chunk_size)]

        with Timer("  Process elapsed time (rasterio)") as timer:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(
                    load_tile_rasterio_chunk_mp,
                    repeat(input_file),
                    start_loc_list_iter,
                    repeat(patch_size)
                )
            rasterio_time = timer.elapsed_time()

        print("  ", psutil.virtual_memory())
        output_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},process,{cache_strategy},{input_file},{start_location},{patch_size},{num_workers},{openslide_time},{cucim_time},{rasterio_time},{openslide_time / cucim_time},{rasterio_time / cucim_time},{cache_size},{cache.hit_count},{cache.miss_count}\n"
        with open("experiment.txt", "a+") as f:
            f.write(output_text)
        print(output_text)


# experiment_thread("nocache", "notebooks/input/image.tif", 12, 0, 256)
# experiment_thread("nocache", "notebooks/input/image2.tif", 12, 0, 256)

# experiment_thread("nocache", "notebooks/input/image.tif", 12, 0, 256)
# experiment_process("nocache", "notebooks/input/image.tif", 12, 0, 256)
# experiment_thread("per_process", "notebooks/input/image.tif", 12, 0, 256)
# experiment_process("per_process", "notebooks/input/image.tif", 12, 0, 256)
# experiment_thread("shared_memory", "notebooks/input/image.tif", 12, 0, 256)
# experiment_process("shared_memory", "notebooks/input/image.tif", 12, 0, 256)

# experiment_thread("nocache", "notebooks/input/image.tif", 12, 1, 256)
# experiment_process("nocache", "notebooks/input/image.tif", 12, 1, 256)
# experiment_thread("per_process", "notebooks/input/image.tif", 12, 1, 256)
# experiment_process("per_process", "notebooks/input/image.tif", 12, 1, 256)
# experiment_thread("shared_memory", "notebooks/input/image.tif", 12, 1, 256)
# experiment_process("shared_memory", "notebooks/input/image.tif", 12, 1, 256)

# experiment_thread("nocache", "notebooks/input/image2.tif", 12, 0, 256)
# experiment_process("nocache", "notebooks/input/image2.tif", 12, 0, 256)
experiment_thread("per_process", "notebooks/input/image2.tif", 3, 1, 256)
# experiment_process("per_process", "notebooks/input/image2.tif", 12, 0, 256)
# experiment_thread("shared_memory", "notebooks/input/image2.tif", 12, 0, 256)
# experiment_process("shared_memory", "notebooks/input/image2.tif", 12, 0, 256)

# experiment_thread("nocache", "notebooks/input/image2.tif", 12, 1, 256)
# experiment_process("nocache", "notebooks/input/image2.tif", 12, 1, 256)
# experiment_thread("per_process", "notebooks/input/image2.tif", 12, 1, 256)
# experiment_process("per_process", "notebooks/input/image2.tif", 12, 1, 256)
# experiment_thread("shared_memory", "notebooks/input/image2.tif", 12, 1, 256)
# experiment_process("shared_memory", "notebooks/input/image2.tif", 12, 1, 256)


# experiment_thread("nocache", "notebooks/0486052bb.tiff", 12, 0, 1024)
# experiment_process("nocache", "notebooks/0486052bb.tiff", 12, 0, 1024)
# experiment_thread("per_process", "notebooks/0486052bb.tiff", 12, 0, 1024)
# experiment_process("per_process", "notebooks/0486052bb.tiff", 12, 0, 1024)
# experiment_thread("shared_memory", "notebooks/0486052bb.tiff", 12, 0, 1024)
# experiment_process("shared_memory", "notebooks/0486052bb.tiff", 12, 0, 1024)

# experiment_thread("nocache", "notebooks/0486052bb.tiff", 12, 1, 1024)
# experiment_process("nocache", "notebooks/0486052bb.tiff", 12, 1, 1024)
# experiment_thread("per_process", "notebooks/0486052bb.tiff", 12, 1, 1024)
# experiment_process("per_process", "notebooks/0486052bb.tiff", 12, 1, 1024)
# experiment_thread("shared_memory", "notebooks/0486052bb.tiff", 12, 1, 1024)
# experiment_process("shared_memory", "notebooks/0486052bb.tiff", 12, 1, 1024)

# experiment_thread("per_process", "notebooks/0486052bb.tiff", 12, 1, 1024)
# experiment_process("per_process", "notebooks/0486052bb.tiff", 12, 1, 1024)


# for t in range(100):
#     experiment_thread("nocache", "notebooks/input/image2.tif", 12, 1, 256)

sys.exit(0)

#
# Copyright (c) 2020, NVIDIA CORPORATION.
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


# class Timer(ContextDecorator):
#     def __init__(self, message):
#         self.message = message
#         self.end = None

#     def elapsed_time(self):
#         self.end = perf_counter()
#         return self.end - self.start

#     def __enter__(self):
#         self.start = perf_counter()
#         return self

#     def __exit__(self, exc_type, exc, exc_tb):
#         if not self.end:
#             self.elapsed_time()
#         print("{} : {}".format(self.message, self.end - self.start))


# img = CuImage("notebooks/input/TUPAC-TR-467.svs")

# # cache = CuImage.cache("per_process", memory_capacity=1024)

# with Timer("  Thread elapsed time (cuCIM)") as timer:
#     a = img.read_region(num_workers=8)
#     print(a.shape)


# with Timer("  Thread elapsed time (tifffile)") as timer:
#     with TiffFile("notebooks/input/TUPAC-TR-467.svs") as tif:
#         a = tif.asarray()
#         print(a.shape)
# del img

# sys.exit(0)


cache = CuImage.cache("per_process", memory_capacity=1024)

img = CuImage("notebooks/input/TUPAC-TR-467.svs")

locations = [[0, 0], [100, 0], [200, 0], [300, 0],
             [0, 200], [100, 200], [200, 200]]
locations = np.array(locations)

region = img.read_region(
    locations[0:2], (224, 224), batch_size=2, num_workers=1)

b = next(region)
print("1", b)
print("2", b.shape)
c = np.asarray(b)
print("3", c.shape)

sys.exit(0)


cache = CuImage.cache("per_process", memory_capacity=1024)

img = CuImage("notebooks/input/TUPAC-TR-467.svs")

locations = [[0, 0], [100, 0], [200, 0], [300, 0],
             [0, 200], [100, 200], [200, 200]]
locations = np.array(locations)

# # locations = [[0, 0], [100, 0], [200, 0]]
# # locations = np.array(locations)

# region = img.read_region(locations[0], (224, 224), batch_size=1, num_workers=1)
# print(region)
# c = region.read_region((0, 0), (10, 10))
# print(c, c.shape)


# # b = next(region)
# # c = b.read_region((0, 0), (10, 10))
# sys.exit(0)


region = img.read_region(locations, (224, 224),
                         batch_size=2, drop_last=True, num_workers=4)
print(dir(region))
img2 = np.asarray(region)
print("#", img2)
for batch in region:
    img2 = np.asarray(batch)
    print(img2.shape)
# #     for item in img:
# #         print(item.shape)

# region2 = img.read_region(locations[0], (224, 224), batch_size=1, num_workers=0)

# for batch in region:
#     img2 = np.asarray(batch)
#     print("@@@", img2.shape)

# for batch in region2:
#     img2 = np.asarray(batch)
#     print("@@@", img2.shape)


# region = img.read_region(locations, (224, 224), batch_size=1, num_workers=1)
# for batch in region:
#     img2 = np.asarray(batch)
#     print(img2.shape)

# for item in img2:
#     print(item.shape)


# region = img.read_region(locations[0], (224, 224), batch_size=4, num_workers=8)
# print(region.shape)


# a = iter(region)  # CuImageIterator(region)
# print("hoho")
# img2 = next(a)
# print(dir(a))
# a = iter(region)  # CuImageIterator(region)
# print("hoho2")
# img2 = next(a)
# print(dir(a))
# print("hoho3")
# img2 = next(a)
# print(dir(a))


sys.exit(0)


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


img = CuImage("notebooks/input/TUPAC-TR-467.svs")

# cache = CuImage.cache("per_process", memory_capacity=1024)

with Timer("  Thread elapsed time (cuCIM)") as timer:
    a = img.read_region(num_workers=6)
    print(a.shape)


with Timer("  Thread elapsed time (tifffile)") as timer:
    with TiffFile("notebooks/input/TUPAC-TR-467.svs") as tif:
        a = tif.asarray()
        print(a.shape)

sys.exit(0)


# cache = CuImage.cache("per_process", memory_capacity=1024)

img = CuImage("notebooks/input/TUPAC-TR-467.svs")
a = img.read_region(num_workers=16)
print("###finished")

sys.exit(0)

locations = [[0, 0], [100, 0], [200, 0], [300, 0],
             [0, 200], [100, 200], [200, 200], [300, 200]]
# locations = [[0, 0], [100, 0], [200, 0], [300, 0]]
locations = np.array(locations)

region = img.read_region(locations, (224, 224), batch_size=4, num_workers=8)
print(region.shape)

print("done!")


sys.exit(0)


cache = CuImage.cache("per_process", memory_capacity=1024)

img = CuImage("notebooks/input/TUPAC-TR-467.svs")


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


# with Timer("  Thread elapsed time (cuCIM)") as timer:
#     a = img.read_region(num_workers=16)
#     print(a.shape)


# with Timer("  Thread elapsed time (tifffile)") as timer:
#     with TiffFile("notebooks/input/TUPAC-TR-467.svs") as tif:
#         a = tif.asarray()
#         print(a.shape)

locations = [[0, 0], [100, 0], [200, 0], [300, 0],
             [0, 200], [100, 200], [200, 200], [300, 200]]
locations = np.array(locations)

region = img.read_region(locations, (224, 224), batch_size=4, num_workers=8)
print(region.shape)
# from cucim import CuImage

# # img = CuImage("notebooks/input/image.tif")
# # print(img.read_region((0, 0, 200, 200), (200, 200), num_workers=2).shape)

sys.exit(0)


input_file = "notebooks/input/image2.tif"

# img = CuImage(input_file)
# # True if image data is loaded & available.
# print(img.is_loaded)
# # A device type.
# print(img.device)
# # The number of dimensions.
# print(img.ndim)
# # A string containing a list of dimensions being requested.
# print(img.dims)
# # A tuple of dimension sizes (in the order of `dims`).
# print(img.shape)
# # Returns size as a tuple for the given dimension order.
# print(img.size('XYC'))
# # The data type of the image.
# print(img.dtype)
# # A channel name list.
# print(img.channel_names)
# # Returns physical size in tuple.
# print(img.spacing())
# # Units for each spacing element (size is same with `ndim`).
# print(img.spacing_units())
# # Physical location of (0, 0, 0) (size is always 3).
# print(img.origin)
# # Direction cosines (size is always 3x3).
# print(img.direction)
# # Coordinate frame in which the direction cosines are measured.
# # Available Coordinate frame is not finalized yet.
# print(img.coord_sys)
# # Returns a set of associated image names.
# print(img.associated_images)
# # Returns a dict that includes resolution information.
# print(json.dumps(img.resolutions, indent=2))
# # A metadata object as `dict`
# print(json.dumps(img.metadata, indent=2))
# # A raw metadata string.
# print(img.raw_metadata)


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


num_threads = os.cpu_count()

start_location = 0
tile_size = 256


def load_tile_openslide(slide, start_loc, tile_size):
    _ = slide.read_region(start_loc, 0, [tile_size, tile_size])


def load_tile_cucim(slide, start_loc, tile_size):
    _ = slide.read_region(start_loc, [tile_size, tile_size], 0)


openslide_tot_time = 0
cucim_tot_time = 0
for num_workers in (1,):  # range(1, num_threads + 1):
    # with OpenSlide(input_file) as slide:
    #     width, height = slide.dimensions

    #     count = 0
    #     for h in range(start_location, height, tile_size):
    #         for w in range(start_location, width, tile_size):
    #             count += 1
    #     start_loc_iter = ((w, h)
    #                       for h in range(start_location, height, tile_size)
    #                       for w in range(start_location, width, tile_size))
    #     with Timer("  Thread elapsed time (OpenSlide)") as timer:
    #         with concurrent.futures.ThreadPoolExecutor(
    #             max_workers=num_workers
    #         ) as executor:
    #             executor.map(
    #                 lambda start_loc: load_tile_openslide(
    #                     slide, start_loc, tile_size),
    #                 start_loc_iter,
    #             )
    #         openslide_time = timer.elapsed_time()
    #         openslide_tot_time += openslide_time

    cucim_time = 0
    slide = CuImage(input_file)
    width, height = slide.size('XY')
    start_loc_iter = ((w, h)
                      for h in range(start_location, height, tile_size)
                      for w in range(start_location, width, tile_size))
    with Timer("  Thread elapsed time (cuCIM)") as timer:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            executor.map(
                lambda start_loc: load_tile_cucim(slide, start_loc, tile_size),
                start_loc_iter,
            )
        cucim_time = timer.elapsed_time()
        cucim_tot_time += cucim_time
    # print("  Performance gain (OpenSlide/cuCIM): {}".format(
    #     openslide_time / cucim_time))

# print("Total time (OpenSlide):", openslide_tot_time)
print("Total time (cuCIM):", cucim_tot_time)
# print("Average performance gain (OpenSlide/cuCIM): {}".format(
#     openslide_tot_time / cucim_tot_time))
