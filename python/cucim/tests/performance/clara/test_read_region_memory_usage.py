#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import pytest

from ...util.io import open_image_cucim

# skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")


def test_read_region_cuda_memleak(testimg_tiff_stripe_4096x4096_256_jpeg):
    import GPUtil

    gpus = GPUtil.getGPUs()

    if len(gpus) == 0:
        pytest.skip("No gpu available")

    img = open_image_cucim(testimg_tiff_stripe_4096x4096_256_jpeg)

    gpu = gpus[0]
    mem_usage_history = [gpu.memoryUsed]

    for i in range(10):
        _ = img.read_region(device="cuda")
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]
        mem_usage_history.append(gpu.memoryUsed)

    print(mem_usage_history)

    # The difference in memory usage should be less than 180MB.
    # Note: Since we cannot measure GPU memory usage for a process,
    #       we use a rough number.
    #       (experimentally measured, assuming that each image load
    #        consumes around 50MB of GPU memory).
    assert mem_usage_history[5] - mem_usage_history[9] < 180.0


def test_read_region_cpu_memleak(testimg_tiff_stripe_4096x4096_256):
    import os

    import psutil

    process = psutil.Process(os.getpid())

    img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    mem_usage_history = [process.memory_info().rss]

    for i in range(10):
        _ = img.read_region()
        mem_usage_history.append(process.memory_info().rss)

    print(mem_usage_history)

    # Memory usage difference should be less than 1MB
    assert mem_usage_history[5] - mem_usage_history[9] < 2**20 * 1


def test_read_random_region_cpu_memleak(testimg_tiff_stripe_4096x4096_256):
    import gc
    import os
    import random

    import psutil

    process = psutil.Process(os.getpid())

    img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    iteration = 10000
    mem_usage_history = [process.memory_info().rss] * iteration
    level_count = img.resolutions["level_count"]

    memory_increment_count = 0

    for i in range(iteration):
        location = (
            random.randrange(-2048, 4096 + 2048),
            random.randrange(-2048, 4096 + 2048),
        )
        level = random.randrange(0, level_count)
        _ = img.read_region(location, (256, 256), level)
        if i == 0 or i == iteration - 1:
            gc.collect()
        mem_usage_history[i] = process.memory_info().rss
        if i > 0:
            if mem_usage_history[i] - mem_usage_history[i - 1] > 0:
                memory_increment_count += 1
                print(
                    f"mem increase (iteration: {i:3d}): "
                    f"{mem_usage_history[i] - mem_usage_history[i - 1]:4d} "
                    "bytes"
                )

    print(mem_usage_history)

    # The expected memory usage difference should be smaller than
    # <iteration> * 256 * 3 bytes
    # (one line of pixels in the tile image is 256 * 3 bytes)
    assert mem_usage_history[-1] - mem_usage_history[1] < iteration * 256 * 3
    # The memory usage increment should be less than 1% of the iteration count.
    assert memory_increment_count < iteration * 0.01


def test_tiff_iterator(testimg_tiff_stripe_4096x4096_256):
    """Verify that the iterator of read_region does not cause a memory leak.
    See issue gh-598: https://github.com/rapidsai/cucim/issues/598
    """

    import os
    import random

    import numpy as np
    import psutil

    slide_path = testimg_tiff_stripe_4096x4096_256

    level = 0
    size = (300, 300)
    batch_size = 64
    num_workers = 16

    # This may be a non-integer number, to have the last iteration return a
    # number of patches < batch_size
    num_batches = 4.3
    locations = [
        (random.randint(0, 1000), random.randint(2000, 3000))
        for _ in range(int(batch_size * num_batches))
    ]
    print(
        f"Number of locations: {len(locations)}, batch size: {batch_size}, "
        "number of workers: {num_workers}"
    )

    def get_total_mem_usage():
        current_process = psutil.Process(os.getpid())
        mem = current_process.memory_info().rss
        for child in current_process.children(recursive=True):
            mem += child.memory_info().rss
        return mem

    with open_image_cucim(slide_path) as slide:
        start_mem = None
        mem_usage_history = []
        for i in range(101):
            gen = slide.read_region(
                locations,
                size,
                level,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            for x in gen:
                _ = np.asarray(x)

            if i % 10 == 0:
                if start_mem is None:
                    start_mem = get_total_mem_usage()
                    print(
                        f"starting with: {(start_mem) // (1024 * 1024)}"
                        " MB of memory consumption"
                    )
                else:
                    memory_increase = (get_total_mem_usage() - start_mem) // (
                        1024 * 1024
                    )
                    mem_usage_history.append(memory_increase)
                    print(
                        f"mem increase (iteration: {i:3d}): "
                        "{memory_increase:4d} MB"
                    )
        # Memory usage difference should be less than 20MB
        assert mem_usage_history[-1] - mem_usage_history[1] < 20
