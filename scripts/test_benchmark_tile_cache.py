# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise benchmark workloads without TIFF decoding or a CUDA runtime."""

from itertools import combinations

import numpy as np
import pytest
from benchmark_tile_cache import measure


def covered_tiles(origin, size, tile):
    x, y = origin
    width, height = size
    tile_w, tile_h = tile
    return {
        (tx, ty)
        for ty in range(y // tile_h, (y + height - 1) // tile_h + 1)
        for tx in range(x // tile_w, (x + width - 1) // tile_w + 1)
    }


class TileCache:
    def __init__(self, cache_type):
        self.type = cache_type
        self.tiles = set()
        self.hit_count = 0
        self.miss_count = 0
        self.recording = False

    def record(self, enabled):
        self.recording = enabled


@pytest.fixture
def fake_cuimage():
    class FakeCuImage:
        shape = (4096, 4096, 3)
        resolutions = {"level_tile_sizes": ((256, 256),)}
        instances = []
        active_cache = None

        @classmethod
        def cache(cls, cache_type=None, **kwargs):
            if cache_type is not None:
                cls.active_cache = TileCache(cache_type)
            return cls.active_cache

        def __init__(self, path):
            self.path = path
            self.origins = []
            self.closed = False
            self.instances.append(self)

        def read_region(self, origin, size, *, level, device):
            assert not self.closed
            assert level == 0
            self.origins.append(origin)
            tile_sizes = self.resolutions.get("level_tile_sizes", ())
            cache = self.cache()
            if (
                cache.type != "nocache"
                and tile_sizes
                and all(t > 0 for t in tile_sizes[0])
            ):
                for tile in covered_tiles(origin, size, tile_sizes[0]):
                    key = (self.path, tile)
                    if cache.recording:
                        if key in cache.tiles:
                            cache.hit_count += 1
                        else:
                            cache.miss_count += 1
                    cache.tiles.add(key)
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    return FakeCuImage


def test_default_cold_grid_and_cache_counts(fake_cuimage):
    # Repeating a cell must also start with an empty cache.
    for _ in range(2):
        result = measure(
            "image.tif", fake_cuimage, "per_process", "cpu", (1024, 1024), 20, 8
        )
        cold = fake_cuimage.instances[-1]
        assert cold.origins == [
            (0, 0),
            (1024, 0),
            (2048, 0),
            (3072, 0),
            (0, 1024),
            (1024, 1024),
            (2048, 1024),
            (3072, 1024),
        ]
        assert result["cache_active"]
        assert result["warm_hits"] == 320
        assert result["warm_misses"] == 0
        assert result["cold_hits"] == 0
        assert result["cold_misses"] == 128


@pytest.mark.parametrize(
    "image_size, size, tile, num_regions, tiles_per_region",
    [
        ((4096, 4096), (1024, 1024), (256, 256), 16, 16),
        ((3072, 2048), (700, 300), (256, 128), 20, 9),
        ((1000, 800), (300, 200), (256, 128), 6, 4),
        ((1024, 768), (1024, 768), (256, 128), 1, 24),
        ((512, 512), (100, 80), (256, 128), 8, 1),
    ],
)
def test_cold_regions_fit_and_share_no_tiles(
    fake_cuimage, image_size, size, tile, num_regions, tiles_per_region
):
    width, height = image_size
    fake_cuimage.shape = (height, width, 3)
    fake_cuimage.resolutions = {"level_tile_sizes": (tile,)}
    result = measure(
        "image.tif", fake_cuimage, "per_process", "cpu", size, 2, num_regions
    )
    origins = fake_cuimage.instances[-1].origins
    assert len(origins) == len(set(origins)) == num_regions
    for x, y in origins:
        assert 0 <= x <= width - size[0]
        assert 0 <= y <= height - size[1]
    tile_sets = [covered_tiles(origin, size, tile) for origin in origins]
    assert all(a.isdisjoint(b) for a, b in combinations(tile_sets, 2))
    assert result["warm_hits"] == 2 * tiles_per_region
    assert result["warm_misses"] == 0
    assert result["cold_hits"] == 0
    assert result["cold_misses"] == num_regions * tiles_per_region


@pytest.mark.parametrize(
    "size, repeats, num_regions, message",
    [
        ((), 1, 1, "two positive dimensions"),
        ((1024,), 1, 1, "two positive dimensions"),
        ((1024, 1024, 3), 1, 1, "two positive dimensions"),
        ((0, 1024), 1, 1, "two positive dimensions"),
        ((1024, -1), 1, 1, "two positive dimensions"),
        ((4097, 1024), 1, 1, "fit inside the image"),
        ((1024, 4097), 1, 1, "fit inside the image"),
        ((1024, 1024), 0, 1, "repeats and regions must be positive"),
        ((1024, 1024), -1, 1, "repeats and regions must be positive"),
        ((1024, 1024), 1, 0, "repeats and regions must be positive"),
        ((1024, 1024), 1, -1, "repeats and regions must be positive"),
        ((1024, 1024), 1, 17, "not enough tile-disjoint regions"),
        # Pixel rectangles fit, but rounding strides to tiles reduces capacity.
        ((1000, 1000), 1, 10, "not enough tile-disjoint regions"),
    ],
)
def test_invalid_workload_is_rejected(
    fake_cuimage, size, repeats, num_regions, message
):
    fake_cuimage.resolutions = {"level_tile_sizes": ((300, 300),)}
    with pytest.raises(ValueError, match=message):
        measure(
            "image.tif", fake_cuimage, "per_process", "cpu", size, repeats, num_regions
        )
    assert all(not img.origins for img in fake_cuimage.instances)


@pytest.mark.parametrize("resolutions", [{}, {"level_tile_sizes": ((0, 0),)}])
def test_unknown_tile_dimensions_use_pixel_alignment(fake_cuimage, resolutions):
    fake_cuimage.shape = (600, 1000, 3)
    fake_cuimage.resolutions = resolutions
    result = measure("image.tif", fake_cuimage, "per_process", "cpu", (300, 200), 2, 9)
    assert fake_cuimage.instances[-1].origins == [
        (x, y) for y in (0, 200, 400) for x in (0, 300, 600)
    ]
    assert not result["cache_active"]
    assert result["tile"] == [0, 0]
    assert result["warm_hits"] == result["warm_misses"] == 0
    assert result["cold_hits"] == result["cold_misses"] == 0


def test_nocache_does_not_record_hits_or_misses(fake_cuimage):
    result = measure("image.tif", fake_cuimage, "nocache", "cpu", (1024, 1024), 2, 8)
    assert len(set(fake_cuimage.instances[-1].origins)) == 8
    assert not result["cache_active"]
    assert result["warm_hits"] == result["warm_misses"] == 0
    assert result["cold_hits"] == result["cold_misses"] == 0
