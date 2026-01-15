#
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import pytest

from ...util.io import open_image_cucim

# skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")


def test_read_region_cuda_memleak(testimg_tiff_stripe_4096x4096_256_jpeg):
    def get_used_gpu_memory_mib():
        """Get the used GPU memory in MiB."""
        dev = cp.cuda.Device()
        free, total = dev.mem_info
        memory_used = (total - free) / (2**20)
        return memory_used

    num_gpus = cp.cuda.runtime.getDeviceCount()
    if num_gpus == 0:
        pytest.skip("No gpu available")

    # Check for Unified Memory
    # (as in the case of devices using system-on-a-chip, SoC)
    # On these systems, CPU and GPU memory are the same, so this test, which
    # is designed to measure GPU-specific memory leaks, is not applicable.
    # ref: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
    dev = cp.cuda.Device()
    if dev.attributes.get("UnifiedAddressing", 0):
        pytest.skip(
            "Skipping memory leak test on unified memory systems (e.g., SoC)"
        )

    img = open_image_cucim(testimg_tiff_stripe_4096x4096_256_jpeg)

    mem_usage_history = [
        get_used_gpu_memory_mib()
    ]  # Memory before loop (image loaded)

    num_iterations = 30
    warmup_iterations = (
        20  # Number of iterations to run before establishing a baseline
    )

    for i in range(num_iterations):
        region_data = img.read_region(device="cuda")
        # Explicitly delete the CuPy array
        del region_data
        # Force CuPy to free unused blocks from its memory pool
        cp.get_default_memory_pool().free_all_blocks()
        mem_usage_history.append(get_used_gpu_memory_mib())

    print(f"Full memory usage history (MiB): {mem_usage_history}")

    # mem_usage_history[0] is before any read_region calls
    # mem_usage_history[k] is after the k-th iteration (read_region, del,
    # free_all_blocks)

    # Baseline memory after warmup_iterations (e.g., after 3rd iteration)
    # Ensure warmup_iterations is less than num_iterations
    if warmup_iterations >= num_iterations:
        pytest.fail(
            "warmup_iterations must be less than num_iterations for this test "
            "logic"
        )

    # Memory after the warmup period (e.g., after 3rd call, so index 3)
    mem_after_warmup = mem_usage_history[warmup_iterations]
    # Memory after all iterations (e.g., after 10th call, so index 10)
    mem_at_end = mem_usage_history[num_iterations]

    # Calculate the increase in memory after the warmup period
    memory_increase_after_warmup = mem_at_end - mem_after_warmup

    print(
        f"Memory after warmup ({warmup_iterations} iterations): "
        f"{mem_after_warmup:.2f} MiB"
    )
    print(f"Memory at end ({num_iterations} iterations): {mem_at_end:.2f} MiB")
    print(
        f"Memory increase after warmup: {memory_increase_after_warmup:.2f} MiB"
    )

    # The increase in memory after the warm-up phase and explicit freeing
    # should be minimal, ideally close to zero for a perfectly clean operation.
    # This threshold (leak_threshold_mib, e.g., 100.0 MiB) defines an acceptable
    # upper bound for the *cumulative* memory increase observed over the
    # (num_iterations - warmup_iterations) test iterations.
    # It accounts for potential minor non-reclaimable memory that might
    # accumulate due to factors like fragmentation, persistent driver/runtime
    # overheads, or small, consistent allocation patterns within the tested
    # function, even with explicit attempts to free memory.
    #
    # For instance, a 100.0 MiB threshold over 10 active test iterations
    # (30 total iterations - 20 warmup iterations) allows for an average of
    # roughly 10.0 MiB of such net memory growth per iteration during the
    # measurement phase. (Note that a 4096x4096 image consumes approximately 50
    # MiB of GPU memory)
    # This approach is significantly different from a previous version of this
    # test, which used a 180MB threshold for a non-cumulative comparison
    # (i.e., `memory_at_iteration_5 - memory_at_iteration_9`), which could
    # be affected by transient spikes rather than sustained growth.
    # If the `read_region` operation has a consistent memory leak (i.e., memory
    # that is allocated and not freed properly on an ongoing basis), the
    # `memory_increase_after_warmup` is expected to exceed this threshold.
    leak_threshold_mib = 100.0
    assert memory_increase_after_warmup < leak_threshold_mib, (
        f"Memory increase ({memory_increase_after_warmup:.2f} MiB) "
        f"exceeded threshold ({leak_threshold_mib} MiB) "
        f"over {num_iterations - warmup_iterations} iterations after warmup."
    )


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


@pytest.mark.skip(
    reason="Memory usage regression with nvImageCodec v0.7.0 decoder - investigating (gh-998)"
)
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
