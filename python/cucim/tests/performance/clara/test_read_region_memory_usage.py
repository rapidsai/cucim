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

import pytest

from ...util.io import open_image_cucim


def test_read_region_cuda_memleak(testimg_tiff_stripe_4096x4096_256):
    import GPUtil
    gpus = GPUtil.getGPUs()

    if len(gpus) == 0:
        pytest.skip('No gpu available')

    img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    gpu = gpus[0]
    mem_usage_history = [gpu.memoryUsed]

    for i in range(5):
        _ = img.read_region(device='cuda')
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]
        mem_usage_history.append(gpu.memoryUsed)

    print(mem_usage_history)

    # Memory usage difference should be less than 40MB
    # Note: Since we cannot measure GPU memory usage for a process,
    #       we use a rough number.
    #       Actual CUDA memory used would be 48MB per iteration (4096x4096x3).
    assert mem_usage_history[4] - mem_usage_history[1] < 40.0


def test_read_region_cpu_memleak(testimg_tiff_stripe_4096x4096_256):
    import os

    import psutil
    process = psutil.Process(os.getpid())

    img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    mem_usage_history = [process.memory_info().rss]

    for i in range(5):
        _ = img.read_region()
        mem_usage_history.append(process.memory_info().rss)

    print(mem_usage_history)

    # Memory usage difference should be less than 1MB
    assert mem_usage_history[4] - mem_usage_history[1] < 2**20 * 1


def test_read_random_region_cpu_memleak(testimg_tiff_stripe_4096x4096_256):
    import os
    import random

    import psutil
    process = psutil.Process(os.getpid())

    img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    iteration = 1000
    mem_usage_history = [process.memory_info().rss] * iteration
    level_count = img.resolutions['level_count']

    for i in range(iteration):
        location = (random.randrange(-2048, 4096 + 2048),
                    random.randrange(-2048, 4096 + 2048))
        level = random.randrange(0, level_count)
        _ = img.read_region(location, (256, 256), level)
        mem_usage_history[i] = process.memory_info().rss

    print(mem_usage_history)

    # Memory usage difference should be smaller than (iteration) * 100 bytes
    assert mem_usage_history[-1] - mem_usage_history[1] < iteration * 100
