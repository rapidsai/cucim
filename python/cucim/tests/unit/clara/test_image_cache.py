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

# skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")


def test_get_nocache():
    from cucim import CuImage

    cache = CuImage.cache()

    assert int(cache.type) == 0
    assert cache.memory_size == 0
    assert cache.memory_capacity == 0
    assert cache.free_memory == 0
    assert cache.size == 0
    assert cache.capacity == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    config = cache.config
    # Check essential properties
    #  {'type': 'nocache', 'memory_capacity': 1024, 'capacity': 5461,
    #   'mutex_pool_capacity': 11117, 'list_padding': 10000,
    #   'extra_shared_memory_size': 100, 'record_stat': False}
    assert config["type"] == "nocache"
    assert not config["record_stat"]


def test_get_per_process_cache():
    from cucim import CuImage

    cache = CuImage.cache("per_process", memory_capacity=2048)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 2048
    assert cache.free_memory == 2**20 * 2048
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    config = cache.config
    # Check essential properties
    #  {'type': 'per_process', 'memory_capacity': 2048, 'capacity': 10922,
    #   'mutex_pool_capacity': 11117, 'list_padding': 10000,
    #   'extra_shared_memory_size': 100, 'record_stat': False}
    assert config["type"] == "per_process"
    assert config["memory_capacity"] == 2048
    assert not config["record_stat"]


def test_get_shared_memory_cache():
    from cucim import CuImage

    cache = CuImage.cache("shared_memory", memory_capacity=128)
    assert int(cache.type) == 2
    assert cache.memory_size == 0
    # It allocates additional memory
    assert cache.memory_capacity > 2**20 * 128
    assert cache.free_memory > 2**20 * 128
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    config = cache.config
    # Check essential properties
    #  {'type': 'shared_memory', 'memory_capacity': 2048, 'capacity': 682,
    #   'mutex_pool_capacity': 100003, 'list_padding': 10000,
    #   'extra_shared_memory_size': 100, 'record_stat': False}
    assert config["type"] == "shared_memory"
    assert config["memory_capacity"] == 128
    assert not config["record_stat"]


def test_preferred_memory_capacity(testimg_tiff_stripe_32x24_16_jpeg):
    from cucim import CuImage
    from cucim.clara.cache import preferred_memory_capacity

    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)

    # same with `img.resolutions["level_dimensions"][0]`
    image_size = img.size("XY")  # 32x24
    tile_size = img.resolutions["level_tile_sizes"][0]  # 16x16
    patch_size = (tile_size[0] * 2, tile_size[0] * 2)
    bytes_per_pixel = 3  # default: 3

    # Below three statements are the same.
    memory_capacity = preferred_memory_capacity(img, patch_size=patch_size)
    memory_capacity2 = preferred_memory_capacity(
        None, image_size, tile_size, patch_size, bytes_per_pixel
    )
    memory_capacity3 = preferred_memory_capacity(
        None, image_size, patch_size=patch_size
    )

    assert memory_capacity == memory_capacity2  # 1 == 1
    assert memory_capacity2 == memory_capacity3  # 1 == 1

    # You can also manually set capacity` (e.g., `capacity=500`)
    cache = CuImage.cache("per_process", memory_capacity=memory_capacity)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 1
    assert cache.free_memory == 2**20 * 1
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    basic_memory_capacity = preferred_memory_capacity(
        None,
        image_size=(1024 * 1024, 1024 * 1024),
        tile_size=(256, 256),
        patch_size=(256, 256),
        bytes_per_pixel=3,
    )
    assert basic_memory_capacity == 1536  # https://godbolt.org/z/jY7G84xzT


def test_reserve_more_cache_memory():
    from cucim import CuImage
    from cucim.clara.cache import preferred_memory_capacity

    memory_capacity = preferred_memory_capacity(
        None,
        image_size=(1024 * 1024, 1024 * 1024),
        tile_size=(256, 256),
        patch_size=(256, 256),
        bytes_per_pixel=3,
    )
    new_memory_capacity = preferred_memory_capacity(
        None,
        image_size=(1024 * 1024, 1024 * 1024),
        tile_size=(256, 256),
        patch_size=(512, 512),
        bytes_per_pixel=3,
    )

    cache = CuImage.cache("per_process", memory_capacity=memory_capacity)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 1536
    assert cache.free_memory == 2**20 * 1536
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    cache.reserve(new_memory_capacity)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 2304
    assert cache.free_memory == 2**20 * 2304
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    cache.reserve(memory_capacity, capacity=500)
    # Smaller `memory_capacity` value does not change this')
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 2304
    assert cache.free_memory == 2**20 * 2304
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    cache = CuImage.cache("no_cache")
    # Set new cache will reset memory size
    assert int(cache.type) == 0
    assert cache.memory_size == 0
    assert cache.memory_capacity == 0
    assert cache.free_memory == 0
    assert cache.size == 0
    assert cache.capacity == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0


@pytest.mark.skip(reason="currently fails (gh-626)")
def test_cache_hit_miss(testimg_tiff_stripe_32x24_16_jpeg):
    from cucim import CuImage
    from cucim.clara.cache import preferred_memory_capacity

    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    memory_capacity = preferred_memory_capacity(img, patch_size=(16, 16))
    cache = CuImage.cache(
        "per_process", memory_capacity=memory_capacity, record_stat=True
    )

    img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (0, 1)

    _ = img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (1, 1)

    _ = img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (2, 1)
    assert cache.record()

    cache.record(False)
    assert not cache.record()

    _ = img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (0, 0)

    assert int(cache.type) == 1
    assert cache.memory_size == 768
    assert cache.memory_capacity == 2**20 * 1
    assert cache.free_memory == 2**20 * 1 - 768
    assert cache.size == 1
    assert cache.capacity == 5

    cache = CuImage.cache("no_cache")

    assert int(cache.type) == 0
    assert cache.memory_size == 0
    assert cache.memory_capacity == 0
    assert cache.free_memory == 0
    assert cache.size == 0
    assert cache.capacity == 0
