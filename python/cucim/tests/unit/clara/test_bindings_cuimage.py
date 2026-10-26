#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import gc
import os
import threading
import time
import weakref

import numpy as np
import pytest

# skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")

from cucim import CuImage  # noqa: E402
from cucim.clara import DLDataType, DLDataTypeCode, cache, io  # noqa: E402
from cucim.clara._cucim import CuImageIterator  # noqa: E402


def _gpu_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def test_open_image_properties(testimg_tiff_stripe_32x24_16_jpeg):
    path = testimg_tiff_stripe_32x24_16_jpeg
    img = CuImage(path=path)

    assert img.path == path
    assert img.is_loaded is True
    assert bool(img) is False
    assert bool(img.read_region((0, 0), (8, 8))) is False
    assert img.device.type == io.DeviceType.CPU
    assert img.ndim == 3
    assert img.dims == "YXC"
    assert img.shape == [24, 32, 3] and type(img.shape) is list
    assert img.size("XY") == [32, 24]
    assert img.size() == [24, 32, 3]
    assert img.dtype == DLDataType(DLDataTypeCode.DLUInt, 8, 1)
    assert img.typestr == "|u1"
    assert img.channel_names == ["R", "G", "B"]
    assert len(img.spacing()) == 3
    assert all(type(v) is float for v in img.spacing("XY"))
    assert all(type(v) is str for v in img.spacing_units())
    assert type(img.origin) is list and len(img.origin) == 3
    assert type(img.direction) is list
    assert [len(row) for row in img.direction] == [3, 3, 3]
    assert isinstance(img.coord_sys, str)
    assert isinstance(img.raw_metadata, str)
    assert type(img.associated_images) is set
    assert repr(img) == f"<cucim.CuImage path:{path}>"


def test_resolutions_are_tuples(testimg_tiff_stripe_4096x4096_256_jpeg):
    img = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg)
    res = img.resolutions
    assert list(res) == [
        "level_count",
        "level_dimensions",
        "level_downsamples",
        "level_tile_sizes",
    ]
    assert res["level_count"] == 6
    assert type(res["level_dimensions"]) is tuple
    assert res["level_dimensions"][0] == (4096, 4096)
    assert res["level_dimensions"][1] == (2048, 2048)
    assert res["level_downsamples"][:2] == (1.0, 2.0)
    assert res["level_tile_sizes"][0] == (256, 256)


def test_metadata_is_a_json_dict(testimg_tiff_stripe_4096x4096_256_jpeg):
    img = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg)
    md = img.metadata
    assert type(md) is dict
    cucim_md = md["cucim"]
    assert list(cucim_md) == sorted(cucim_md)
    assert cucim_md["path"] == img.path
    assert cucim_md["shape"] == img.shape
    assert cucim_md["dims"] == "YXC"
    assert cucim_md["dtype"] == {"bits": 8, "code": 1, "lanes": 1}
    assert cucim_md["spacing"] == pytest.approx(img.spacing())
    assert type(cucim_md["associated_images"]) is list
    resolutions = cucim_md["resolutions"]
    assert resolutions["level_count"] == img.resolutions["level_count"]
    assert resolutions["level_dimensions"][0] == [4096, 4096]
    assert type(resolutions["level_downsamples"][0]) is float
    assert "tiff" in md


def test_context_manager_returns_same_object(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    with img as entered:
        assert entered is img
    with pytest.raises(KeyError):
        with CuImage(testimg_tiff_stripe_32x24_16_jpeg):
            raise KeyError("not suppressed")


def test_dynamic_attributes(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    img.user_tag = "kept"
    assert img.user_tag == "kept"


def test_cpu_array_interface(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    assert not hasattr(img, "__array_interface__")
    assert not hasattr(img, "__cuda_array_interface__")

    region = img.read_region((4, 2), (16, 8))
    assert type(region) is CuImage
    assert not hasattr(region, "__cuda_array_interface__")
    interface = region.__array_interface__
    assert set(interface) == {
        "data",
        "strides",
        "descr",
        "typestr",
        "shape",
        "version",
    }
    pointer, readonly = interface["data"]
    assert type(pointer) is int and pointer != 0
    assert readonly is False
    assert interface["strides"] is None
    assert interface["descr"] == [("", "|u1")]
    assert interface["typestr"] == "|u1"
    assert interface["shape"] == (8, 16, 3)
    assert interface["version"] == 3

    whole = np.asarray(img.read_region((0, 0), (32, 24)))
    assert whole.shape == (24, 32, 3)
    np.testing.assert_array_equal(np.asarray(region), whole[2:10, 4:20])


def test_read_region_argument_forms(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    expected = np.asarray(img.read_region((4, 2), (16, 8)))
    for region in (
        img.read_region([4, 2], [16, 8]),
        img.read_region(location=(4, 2), size=(16, 8), level=0),
        img.read_region((4, 2), (16, 8), 0, device="cpu"),
        img.read_region((4, 2), (16, 8), 0, device=io.Device("cpu")),
        img.read_region((4, 2), (16, 8), 0, device=b"cpu"),
        img.read_region((4, 2), (16, 8), C=0),
    ):
        np.testing.assert_array_equal(np.asarray(region), expected)


def test_read_region_errors(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    with pytest.raises(RuntimeError, match="should be 2"):
        img.read_region((0, 0), (1, 2, 3))
    with pytest.raises(ValueError, match="single character"):
        img.read_region((0, 0), (8, 8), ab=1)
    with pytest.raises(ValueError, match="alphabet"):
        img.read_region((0, 0), (8, 8), **{"1": 1})
    with pytest.raises(TypeError):
        img.read_region((0, 0), (8, 8), level="0")
    with pytest.raises(RuntimeError):
        img.read_region([("a", "b")], (8, 8))
    with pytest.raises(RuntimeError):
        img.read_region((0, 0), (8, 8), C="x")


def test_read_region_numpy_locations(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    expected = np.asarray(img.read_region((4, 2), (16, 8)))
    for dtype in (np.int64, np.longlong):
        region = img.read_region(np.array([4, 2], dtype=dtype), (16, 8))
        np.testing.assert_array_equal(np.asarray(region), expected)
    with pytest.raises(ValueError, match="int64"):
        img.read_region(np.array([4, 2], dtype=np.int32), (16, 8))
    with pytest.raises(ValueError, match="C-contiguous"):
        img.read_region(np.array([4, 0, 2, 0], dtype=np.int64)[::2], (16, 8))


def test_region_data_outlives_source(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    region = img.read_region((0, 0), (32, 24))
    expected = np.array(region, copy=True)
    del img
    view = np.asarray(region)
    del region
    gc.collect()
    np.testing.assert_array_equal(view, expected)


def test_objects_are_released(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    region = img.read_region((0, 0), (8, 8))
    refs = [weakref.ref(img), weakref.ref(region)]
    del img, region
    gc.collect()
    assert [ref() for ref in refs] == [None, None]


def test_region_iteration(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    with pytest.raises(IndexError):
        iter(img)

    region = img.read_region((0, 0), (8, 8))
    it = iter(region)
    assert type(it) is CuImageIterator
    assert iter(it) is it
    assert len(it) == 1
    assert [item for item in it] == [region]
    with pytest.raises(StopIteration):
        next(it)
    assert repr(it).startswith("<cucim.CuImageIterator index:")
    assert len(CuImageIterator(region)) == 1


def test_batch_read_region(testimg_tiff_stripe_4096x4096_256_jpeg):
    img = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg)
    locations = [(0, 0), (256, 256), (512, 512), (768, 768), (1024, 1024)]
    expected = [
        np.asarray(img.read_region(loc, (64, 64)), copy=True)
        for loc in locations
    ]

    batches = img.read_region(
        locations, (64, 64), 0, num_workers=2, batch_size=2
    )
    assert type(batches) is CuImageIterator
    assert len(batches) == 3
    del img
    gc.collect()

    seen = []
    for batch in batches:
        assert type(batch) is CuImage
        arr = np.asarray(batch, copy=True)
        assert arr.ndim == 4 and arr.shape[1:] == (64, 64, 3)
        seen.extend(arr)
    assert len(seen) == len(locations)
    for got, want in zip(seen, expected):
        np.testing.assert_array_equal(got, want)

    flat = np.asarray(locations, dtype=np.int64).ravel().tolist()
    regions = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg).read_region(
        flat, (64, 64), 0, num_workers=1
    )
    assert len(regions) == len(locations)


def test_multiple_locations_need_workers(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    with pytest.raises(RuntimeError):
        img.read_region([(0, 0), (8, 8)], (8, 8))


def test_read_region_releases_gil(
    testimg_tiff_stripe_4096x4096_256_jpeg, restore_cache
):
    CuImage.cache("nocache")
    img = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg)
    img.read_region((0, 0), (256, 256))
    stamps = []
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.0005)

    worker = threading.Thread(target=spin)
    worker.start()
    try:
        time.sleep(0.01)
        start = time.perf_counter()
        img.read_region((0, 0), (4096, 4096))
        end = time.perf_counter()
    finally:
        stop.set()
        worker.join()
    if end - start < 0.02:
        pytest.skip("read was too short to observe GIL release")
    quarter = (end - start) / 4
    during = [s for s in stamps if start + quarter < s < end - quarter]
    assert during, "no Python thread ran while read_region decoded"


@pytest.mark.skipif(not _gpu_available(), reason="needs a CUDA device")
def test_cuda_array_interface(testimg_tiff_stripe_32x24_16_jpeg):
    import cupy

    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    region = img.read_region((0, 0), (16, 8), device="cuda")
    assert region.device.type == io.DeviceType.CUDA
    assert not hasattr(region, "__array_interface__")
    interface = region.__cuda_array_interface__
    assert set(interface) == {
        "data",
        "strides",
        "descr",
        "typestr",
        "shape",
        "version",
        "mask",
        "stream",
    }
    assert interface["shape"] == (8, 16, 3)
    assert interface["typestr"] == "|u1"
    assert interface["mask"] is None
    assert interface["stream"] == 1
    assert interface["version"] == 3
    arr = cupy.asarray(region)
    assert arr.shape == (8, 16, 3) and arr.dtype == cupy.uint8


def test_preferred_memory_capacity(testimg_tiff_stripe_4096x4096_256_jpeg):
    img = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg)
    from_image = cache.preferred_memory_capacity(img, patch_size=(256, 256))
    from_sizes = cache.preferred_memory_capacity(
        None,
        image_size=(4096, 4096),
        tile_size=(256, 256),
        patch_size=(256, 256),
        bytes_per_pixel=3,
    )
    assert type(from_image) is int and from_image > 0
    assert from_image == from_sizes
    with pytest.raises(ValueError, match="image_size"):
        cache.preferred_memory_capacity(None)


def test_pybind11_argument_conversions(testimg_tiff_stripe_32x24_16_jpeg):
    path = testimg_tiff_stripe_32x24_16_jpeg
    for path_arg in (os.fsencode(path), bytearray(os.fsencode(path))):
        assert CuImage(path_arg).shape == [24, 32, 3]
    img = CuImage(path)
    assert img.size(b"XY") == [32, 24]
    locations = [(0, 0), (8, 8)]
    for flag in (np.True_, 1, 0, None, 2.0):
        batches = img.read_region(
            locations, (8, 8), 0, num_workers=1, shuffle=flag, drop_last=flag
        )
        assert type(batches) is CuImageIterator
    with pytest.raises(TypeError):
        img.read_region(locations, (8, 8), 0, num_workers=1, shuffle="yes")
    region = img.read_region((0, 0), (8, 8))
    assert len(CuImageIterator(region, ending=0)) == 1


def test_python_subclass(testimg_tiff_stripe_32x24_16_jpeg):
    class MyImage(CuImage):
        def __init__(self, path, tag):
            super().__init__(path)
            self.tag = tag

    img = MyImage(testimg_tiff_stripe_32x24_16_jpeg, "slide-1")
    assert type(img) is MyImage
    assert img.tag == "slide-1"
    assert img.shape == [24, 32, 3]
    region = img.read_region((0, 0), (8, 8))
    assert type(region) is CuImage
    assert np.asarray(region).shape == (8, 8, 3)


def test_associated_image_without_data(testimg_tiff_stripe_32x24_16_jpeg):
    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    assert img.associated_images == set()
    missing = img.associated_image("thumbnail")
    assert type(missing) is CuImage
    assert not hasattr(missing, "__array_interface__")
    assert not hasattr(missing, "__cuda_array_interface__")
