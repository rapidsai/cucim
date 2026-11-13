#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import sys

import pytest

pytest.importorskip("imagecodecs")


def _loaded_libcucim_files():
    """Return the distinct libcucim shared objects mapped into this process."""
    files = set()
    with open("/proc/self/maps") as f:
        for line in f:
            path = (
                line.split(maxsplit=5)[-1].strip()
                if len(line.split()) > 5
                else ""
            )
            name = os.path.basename(path)
            if name.startswith("libcucim") and ".so" in name:
                files.add(os.path.realpath(path))
    return files


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="needs /proc/self/maps"
)
def test_plugins_share_libcucim_with_python_extension(
    testimg_tiff_stripe_4096x4096_256_jpeg, restore_cache
):
    """The plugins must use the same libcucim as the Python extension.

    If the wheel vendors a second, renamed copy of libcucim for the plugins,
    the cache configured from Python does not reach them, so no cache
    statistics are recorded for reads done by the plugin.
    """
    from cucim import CuImage

    CuImage.cache("per_process", memory_capacity=64, record_stat=True)
    img = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg)
    img.read_region((0, 0), (512, 512), level=0)

    assert len(_loaded_libcucim_files()) == 1, _loaded_libcucim_files()
    cache = CuImage.cache()
    assert cache.hit_count + cache.miss_count > 0
