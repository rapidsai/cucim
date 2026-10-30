# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import functools
import math
import warnings
from copy import deepcopy

import cupy as cp
import numpy as np
import pytest
from cupy.testing import (
    assert_allclose,
    assert_array_equal,
)
from scipy.ndimage import find_objects as cpu_find_objects
from skimage import measure as measure_cpu

from cucim.skimage import data, measure
from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.measure._regionprops import PROPS
from cucim.skimage.measure._regionprops_gpu import (
    _find_close_labels,
    equivalent_diameter_area,
    ndim_2_only,
    need_intensity_image,
    regionprops_area,
    regionprops_area_bbox,
    regionprops_area_convex,
    regionprops_bbox_coords,
    regionprops_centroid,
    regionprops_centroid_local,
    regionprops_centroid_weighted,
    regionprops_coords,
    regionprops_dict,
    regionprops_euler,
    regionprops_extent,
    regionprops_feret_diameter_max,
    regionprops_image,
    regionprops_inertia_tensor,
    regionprops_inertia_tensor_eigvals,
    regionprops_intensity_mean,
    regionprops_intensity_min_max,
    regionprops_intensity_std,
    regionprops_moments,
    regionprops_moments_central,
    regionprops_moments_hu,
    regionprops_moments_normalized,
    regionprops_num_pixels,
    regionprops_perimeter,
    regionprops_perimeter_crofton,
)
from cucim.skimage.measure._regionprops_gpu_basic_kernels import (
    _fill_holes_by_label,
    basic_deps,
)
from cucim.skimage.measure._regionprops_gpu_convex import convex_deps
from cucim.skimage.measure._regionprops_gpu_intensity_kernels import (
    _regionprops_intensity,
    intensity_deps,
)
from cucim.skimage.measure._regionprops_gpu_misc_kernels import misc_deps
from cucim.skimage.measure._regionprops_gpu_moments_kernels import (
    moment_deps,
)


def get_labels_nd(
    shape,
    blob_size_fraction=0.05,
    volume_fraction=0.25,
    rng=5,
    insert_holes=False,
    dilate_blobs=False,
):
    ndim = len(shape)
    blobs_kwargs = dict(
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        rng=rng,
    )
    blobs = data.binary_blobs(max(shape), n_dim=ndim, **blobs_kwargs)
    # crop to rectangular
    blobs = blobs[tuple(slice(s) for s in shape)]

    if dilate_blobs:
        blobs = ndi.binary_dilation(blobs, 3)

    if insert_holes:
        blobs2_kwargs = dict(
            blob_size_fraction=blob_size_fraction / 5,
            volume_fraction=0.1,
            rng=rng,
        )
        # create smaller blobs and invert them to create a holes mask to apply
        # to the original blobs
        temp = data.binary_blobs(max(shape), n_dim=ndim, **blobs2_kwargs)
        temp = temp[tuple(slice(s) for s in shape)]
        mask = cp.logical_and(blobs > 0, temp == 0)
        blobs = blobs * mask

    # binary blobs only creates square outputs
    labels = measure.label(blobs)
    # print(f"# labels generated = {labels.max()}")
    return labels


def get_intensity_image(shape, dtype=cp.float32, seed=5, num_channels=None):
    npixels = math.prod(shape)
    rng = cp.random.default_rng(seed)
    dtype = cp.dtype(dtype)
    if dtype.kind == "f":
        img = cp.arange(npixels, dtype=cp.float32) - npixels // 2
        img = img.reshape(shape)
        if dtype == cp.float16:
            temp = 100 * rng.standard_normal(img.shape, dtype=cp.float32)
            img += temp.astype(cp.float16)
        else:
            img += 100 * rng.standard_normal(img.shape, dtype=dtype)
    else:
        iinfo = cp.iinfo(dtype)
        imax = min(16384, iinfo.max)
        imin = max(0, iinfo.min)
        img = rng.integers(imin, imax, shape)

    if num_channels and num_channels > 1:
        # generate slightly shifted versions for the additional channels
        img = cp.stack((img,) * num_channels, axis=-1)
        for c in range(1, num_channels):
            img[..., c] = cp.roll(img[..., c], shift=c, axis=0)
    return img


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
def test_num_pixels(precompute_max, ndim):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels)) if precompute_max else None
    num_pixels = regionprops_num_pixels(labels, max_label=max_label)
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["num_pixels"]
    )
    assert_allclose(num_pixels, expected["num_pixels"])


@pytest.mark.parametrize("pixels_per_thread", [1, 127, 128, 200, 255, 256, 300])
def test_num_pixels_large_pixels_per_thread(pixels_per_thread):
    # the per-thread counters must hold up to pixels_per_thread pixels of one
    # label (they used to be int8_t, which overflowed above 127)
    labels = cp.ones((64, 64), dtype=cp.uint8)
    labels[40:, :] = 2
    expected = cp.asarray([40 * 64, 24 * 64])
    num_pixels = regionprops_num_pixels(
        labels, pixels_per_thread=pixels_per_thread
    )
    assert_array_equal(num_pixels, expected)
    # the intensity kernels count pixels with the same kind of counters
    intensity_image = cp.ones(labels.shape, dtype=cp.float32)
    props = regionprops_intensity_mean(
        labels, intensity_image, pixels_per_thread=pixels_per_thread
    )
    assert_array_equal(props["num_pixels"], expected)
    assert_allclose(props["intensity_mean"], cp.ones(2))


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("area_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("spacing", [None, (0.5, 0.35, 0.75)])
def test_area(precompute_max, ndim, area_dtype, spacing):
    shape = (256, 512) if ndim == 2 else (45, 63, 37)
    labels = get_labels_nd(shape)
    # discard any extra dimensions from spacing
    if spacing is not None:
        spacing = spacing[:ndim]

    max_label = int(cp.max(labels)) if precompute_max else None
    area = regionprops_area(
        labels, spacing=spacing, max_label=max_label, dtype=area_dtype
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        spacing=spacing,
        properties=["area", "equivalent_diameter_area"],
    )
    assert_allclose(area, expected["area"])

    ed = equivalent_diameter_area(area, ndim)
    assert_allclose(
        ed, expected["equivalent_diameter_area"], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (1, 1, 1), (0.5, 0.35, 0.75)])
@pytest.mark.parametrize(
    "blob_kwargs", [{}, dict(blob_size_fraction=0.12, volume_fraction=0.3)]
)
def test_area_convex_and_solidity(ndim, spacing, blob_kwargs):
    shape = (256, 512) if ndim == 2 else (64, 64, 80)
    labels = get_labels_nd(shape, **blob_kwargs)
    # discard any extra dimensions from spacing
    if spacing is not None:
        spacing = spacing[:ndim]

    max_label = int(cp.max(labels))
    area = regionprops_area(
        labels,
        spacing=spacing,
        max_label=max_label,
    )
    _, _, images_convex = regionprops_image(
        labels,
        max_label=max_label,
        compute_convex=True,
    )
    area_convex = regionprops_area_convex(
        images_convex,
        max_label=max_label,
        spacing=spacing,
    )
    solidity = area / area_convex

    # suppress any QHull warnings coming from the scikit-image implementation
    warnings.filterwarnings(
        "ignore",
        message="Failed to get convex hull image",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="divide by zero",
        category=RuntimeWarning,
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        spacing=spacing,
        properties=["area", "area_convex", "solidity"],
    )
    warnings.resetwarnings()

    assert_allclose(area, expected["area"])

    # Note if 3d blobs are size 1 on one of the axes, it can cause QHull to
    # fail and return a zeros convex image for that label. This has been
    # resolved for cuCIM, but not yet for scikit-image.
    # The test case with blob_kwargs != {} was chosen as a known good
    # setting where such an edge case does NOT occur.
    if blob_kwargs:
        assert_allclose(area_convex, expected["area_convex"])
        assert_allclose(solidity, expected["solidity"])
    else:
        # Can't compare to scikit-image in this case
        # Just make sure the convex area is not smaller than the original
        rtol = 1e-4
        assert cp.all(area_convex >= (area - rtol))
        assert not cp.any(cp.isnan(solidity))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("area_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("spacing", [None, (0.5, 0.35, 0.75)])
def test_extent(ndim, area_dtype, spacing):
    shape = (512, 512) if ndim == 2 else (64, 64, 64)
    labels = get_labels_nd(shape)
    # discard any extra dimensions from spacing
    if spacing is not None:
        spacing = spacing[:ndim]

    # compute area
    max_label = int(cp.max(labels))
    area = regionprops_area(
        labels, spacing=spacing, max_label=max_label, dtype=area_dtype
    )

    # compute bounding-box area
    bbox, slices = regionprops_bbox_coords(
        labels,
        max_label=max_label,
        return_slices=True,
    )
    area_bbox = regionprops_area_bbox(
        bbox, area_dtype=cp.float32, spacing=spacing
    )

    # compute extents from these
    extent = regionprops_extent(area=area, area_bbox=area_bbox)

    # compare to expected result
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), spacing=spacing, properties=["extent"]
    )
    assert_allclose(extent, expected["extent"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("image_dtype", [cp.uint16, cp.uint8, cp.float32])
@pytest.mark.parametrize("mean_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("num_channels", [1, 4])
def test_mean_intensity(
    precompute_max, ndim, image_dtype, mean_dtype, num_channels
):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(
        shape, dtype=image_dtype, num_channels=num_channels
    )

    max_label = int(cp.max(labels)) if precompute_max else None
    props_dict = regionprops_intensity_mean(
        labels, intensity_image, max_label=max_label, mean_dtype=mean_dtype
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        intensity_image=cp.asnumpy(intensity_image),
        properties=["num_pixels", "intensity_mean"],
    )
    assert_array_equal(props_dict["num_pixels"], expected["num_pixels"])
    if num_channels == 1:
        assert_allclose(
            props_dict["intensity_mean"], expected["intensity_mean"], rtol=1e-3
        )
    else:
        for c in range(num_channels):
            assert_allclose(
                props_dict["intensity_mean"][..., c],
                expected[f"intensity_mean-{c}"],
                rtol=1e-3,
            )


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "image_dtype", [cp.uint16, cp.uint8, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize("op_name", ["intensity_min", "intensity_max"])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_intensity_min_and_max(
    precompute_max, ndim, image_dtype, op_name, num_channels
):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(
        shape, dtype=image_dtype, num_channels=num_channels
    )

    max_label = int(cp.max(labels)) if precompute_max else None

    compute_min = op_name == "intensity_min"
    compute_max = not compute_min

    values = regionprops_intensity_min_max(
        labels,
        intensity_image,
        max_label=max_label,
        compute_min=compute_min,
        compute_max=compute_max,
    )[op_name]

    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        intensity_image=cp.asnumpy(intensity_image),
        properties=[op_name],
    )
    if num_channels == 1:
        assert_allclose(values, expected[op_name])
    else:
        for c in range(num_channels):
            assert_allclose(values[..., c], expected[f"{op_name}-{c}"])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_intensity_min_and_max_float64_exact(ndim, num_channels):
    # float64 minima and maxima are pixel values, not rounded to float32
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)
    rng = cp.random.default_rng(5)
    image_shape = shape + ((num_channels,) if num_channels > 1 else ())
    intensity_image = rng.uniform(0.1, 1.0, size=image_shape)
    out = regionprops_intensity_min_max(
        labels, intensity_image, compute_min=True, compute_max=True
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        intensity_image=cp.asnumpy(intensity_image),
        properties=["intensity_min", "intensity_max"],
    )
    for op_name in ["intensity_min", "intensity_max"]:
        values = cp.asnumpy(out[op_name])
        if num_channels == 1:
            assert_array_equal(values, expected[op_name])
        else:
            for c in range(num_channels):
                assert_array_equal(values[..., c], expected[f"{op_name}-{c}"])


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("image_dtype", [cp.uint16, cp.uint8, cp.float32])
@pytest.mark.parametrize("std_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("num_channels", [1, 5])
def test_intensity_std(
    precompute_max, ndim, image_dtype, std_dtype, num_channels
):
    shape = (1024, 2048) if ndim == 2 else (40, 64, 80)
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(
        shape, dtype=image_dtype, num_channels=num_channels
    )

    max_label = int(cp.max(labels)) if precompute_max else None

    # add some specifically sized regions
    if ndim == 2 and precompute_max:
        # clear small region
        labels[50:54, 50:56] = 0
        # add a single pixel labeled region
        labels[51, 51] = max_label + 1
        # add a two pixel labeled region
        labels[53, 53:55] = max_label + 2
        max_label += 2

    props_dict = regionprops_intensity_std(
        labels, intensity_image, max_label=max_label, std_dtype=std_dtype
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        intensity_image=cp.asnumpy(intensity_image),
        properties=["num_pixels", "intensity_mean", "intensity_std"],
    )
    assert_array_equal(props_dict["num_pixels"], expected["num_pixels"])
    if num_channels == 1:
        assert_allclose(
            props_dict["intensity_mean"], expected["intensity_mean"], rtol=1e-3
        )
        assert_allclose(
            props_dict["intensity_std"], expected["intensity_std"], rtol=1e-3
        )
    else:
        for c in range(num_channels):
            assert_allclose(
                props_dict["intensity_mean"][..., c],
                expected[f"intensity_mean-{c}"],
                rtol=1e-3,
            )
            assert_allclose(
                props_dict["intensity_std"][..., c],
                expected[f"intensity_std-{c}"],
                rtol=1e-3,
            )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("image_dtype", [cp.uint8, cp.int16, cp.float32])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("compute_std", [False, True])
def test_intensity_stats_one_pass(ndim, image_dtype, num_channels, compute_std):
    # the mean (or std), min and max computed in one pass match the separate
    # functions
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(
        shape, dtype=image_dtype, num_channels=num_channels
    )
    props = _regionprops_intensity(
        labels,
        intensity_image,
        compute_mean=not compute_std,
        compute_std=compute_std,
        compute_min=True,
        compute_max=True,
    )
    if compute_std:
        expected = regionprops_intensity_std(labels, intensity_image)
    else:
        expected = regionprops_intensity_mean(labels, intensity_image)
    regionprops_intensity_min_max(
        labels,
        intensity_image,
        compute_min=True,
        compute_max=True,
        props_dict=expected,
    )
    assert set(props) == set(expected)
    for name, values in expected.items():
        assert props[name].dtype == values.dtype
        assert_allclose(props[name], values, rtol=1e-12)


@pytest.mark.parametrize("image_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("sample_std", [False, True])
def test_intensity_std_constant_regions(image_dtype, sample_std):
    # sumsq - mean**2 * count rounds to a tiny negative number for some
    # constant values, which used to give NaN instead of 0. (The one-pass
    # formula can also leave a small positive value, up to about 1e-8 of the
    # intensity magnitude, so the tolerance is relative to the values.)
    labels = cp.zeros((20, 50), dtype=cp.uint8)
    intensity_image = cp.zeros(labels.shape, dtype=image_dtype)
    values = (0.1, 1.7, 1000.3, 1e6 + 0.1)
    for i, value in enumerate(values):
        labels[2:18, 12 * i + 1 : 12 * i + 11] = i + 1
        intensity_image[2:18, 12 * i + 1 : 12 * i + 11] = value
    props = regionprops_intensity_std(
        labels, intensity_image, sample_std=sample_std
    )
    std = props["intensity_std"]
    assert not cp.any(cp.isnan(std))
    assert cp.all(std <= 1e-7 * cp.asarray(values))


@pytest.mark.parametrize("image_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("sample_std", [False, True])
def test_intensity_std_nonfinite(image_dtype, sample_std):
    # regions containing NaN or +/-inf have a NaN standard deviation, as with
    # numpy.std (the clip at 0 for constant regions must not change that)
    labels = cp.zeros((4, 8), dtype=cp.uint8)
    intensity_image = cp.ones(labels.shape, dtype=image_dtype)
    for i, value in enumerate((cp.nan, cp.inf, -cp.inf)):
        labels[:, 2 * i : 2 * i + 2] = i + 1
        intensity_image[0, 2 * i] = value
    labels[:, 6:] = 4  # finite values only
    props = regionprops_intensity_std(
        labels, intensity_image, sample_std=sample_std
    )
    std = props["intensity_std"]
    assert cp.all(cp.isnan(std[:3]))
    assert std[3] == 0


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("dtype", [cp.uint32, cp.int64])
@pytest.mark.parametrize("return_slices", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
def test_bbox_coords_and_area(precompute_max, ndim, dtype, return_slices):
    shape = (1024, 512) if ndim == 2 else (80, 64, 48)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels)) if precompute_max else None
    bbox, slices = regionprops_bbox_coords(
        labels,
        max_label=max_label,
        return_slices=return_slices,
    )
    assert bbox.dtype == cp.uint32
    if not return_slices:
        assert slices is None
    else:
        expected_slices = cpu_find_objects(cp.asnumpy(labels))
        assert slices == expected_slices

    spacing = (0.35, 0.75, 0.5)[:ndim]
    expected_bbox = measure_cpu.regionprops_table(
        cp.asnumpy(labels), spacing=spacing, properties=["bbox", "area_bbox"]
    )
    if ndim == 2:
        # TODO make ordering of bbox consistent with regionprops bbox?
        assert_allclose(bbox[:, 0], expected_bbox["bbox-0"])
        assert_allclose(bbox[:, 1], expected_bbox["bbox-1"])
        assert_allclose(bbox[:, 2], expected_bbox["bbox-2"])
        assert_allclose(bbox[:, 3], expected_bbox["bbox-3"])
    elif ndim == 3:
        assert_allclose(bbox[:, 0], expected_bbox["bbox-0"])
        assert_allclose(bbox[:, 1], expected_bbox["bbox-1"])
        assert_allclose(bbox[:, 2], expected_bbox["bbox-2"])
        assert_allclose(bbox[:, 3], expected_bbox["bbox-3"])
        assert_allclose(bbox[:, 4], expected_bbox["bbox-4"])
        assert_allclose(bbox[:, 5], expected_bbox["bbox-5"])

    # compute area_bbox from bbox array
    area_bbox = regionprops_area_bbox(
        bbox, area_dtype=cp.float32, spacing=spacing
    )
    assert_allclose(area_bbox, expected_bbox["area_bbox"], rtol=1e-5)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("via_moments", [False, True])
def test_centroid(via_moments, local, ndim):
    shape = (1024, 512) if ndim == 2 else (80, 64, 48)
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels))
    if via_moments:
        props = {}
        moments = regionprops_moments(
            labels, max_label=max_label, order=1, props_dict=props
        )
        assert "bbox" in props
        assert "moments" in props
    if local:
        name = "centroid_local"
        if via_moments:
            centroid = regionprops_centroid_weighted(
                moments_raw=moments,
                ndim=labels.ndim,
                bbox=props["bbox"],
                compute_local=True,
                compute_global=False,
                weighted=False,
                props_dict=props,
            )[name]
            assert name in props
        else:
            centroid = regionprops_centroid_local(labels, max_label=max_label)
    else:
        name = "centroid"
        if via_moments:
            centroid = regionprops_centroid_weighted(
                moments_raw=moments,
                ndim=labels.ndim,
                bbox=props["bbox"],
                compute_local=False,
                compute_global=True,
                weighted=False,
                props_dict=props,
            )[name]
            assert name in props
        else:
            centroid = regionprops_centroid(labels, max_label=max_label)
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=[name]
    )
    assert_allclose(centroid[:, 0], expected[name + "-0"])
    if ndim > 1:
        assert_allclose(centroid[:, 1], expected[name + "-1"])
    if ndim > 2:
        assert_allclose(centroid[:, 2], expected[name + "-2"])


@pytest.mark.parametrize("ndim", [2, 3])
def test_centroid_reuses_num_pixels(ndim):
    shape = (256, 128) if ndim == 2 else (40, 32, 24)
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels))
    expected = regionprops_centroid(labels, max_label=max_label)
    props = {}
    regionprops_num_pixels(labels, max_label=max_label, props_dict=props)
    centroid = regionprops_centroid(
        labels, max_label=max_label, props_dict=props
    )
    assert_allclose(centroid, expected)


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5), (0.2, 1.3)])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "weighted, intensity_dtype, num_channels",
    [
        (False, None, 1),
        (True, cp.float32, 1),
        (True, cp.uint8, 3),
    ],
)
@pytest.mark.parametrize("norm_type", ["raw", "central", "normalized", "hu"])
@pytest.mark.parametrize("blob_size_fraction", [0.03, 0.1, 0.3])
def test_moments_2d(
    spacing,
    order,
    weighted,
    intensity_dtype,
    num_channels,
    norm_type,
    blob_size_fraction,
):
    shape = (800, 600)
    labels = get_labels_nd(shape, blob_size_fraction=blob_size_fraction)
    max_label = int(cp.max(labels))
    kwargs = {"spacing": spacing}
    prop = "moments"
    if norm_type == "hu":
        if order != 3:
            pytest.skip("Hu moments require order = 3")
        elif spacing and spacing != (1.0, 1.0):
            pytest.skip("Hu moments only support spacing = (1.0, 1.0)")
    if norm_type == "normalized" and order < 2:
        pytest.skip("normalized case only supports order >=2")
    if weighted:
        intensity_image = get_intensity_image(
            shape, dtype=intensity_dtype, num_channels=num_channels
        )
        kwargs["intensity_image"] = intensity_image
        prop += "_weighted"
    if norm_type == "central":
        prop += "_central"
    elif norm_type == "normalized":
        prop += "_normalized"
    elif norm_type == "hu":
        prop += "_hu"
    kwargs_cpu = deepcopy(kwargs)
    if "intensity_image" in kwargs_cpu:
        kwargs_cpu["intensity_image"] = cp.asnumpy(
            kwargs_cpu["intensity_image"]
        )

    # ignore possible warning from skimage implementation
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar power",
        category=RuntimeWarning,
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=[prop], **kwargs_cpu
    )
    warnings.resetwarnings()

    moments = regionprops_moments(
        labels, max_label=max_label, order=order, **kwargs
    )
    if norm_type in ["central", "normalized", "hu"]:
        ndim = len(shape)
        moments = regionprops_moments_central(moments, ndim=ndim)
        if norm_type in ["normalized", "hu"]:
            moments = regionprops_moments_normalized(
                moments, ndim=ndim, spacing=spacing
            )
            if norm_type == "normalized":
                # assert that np.nan values were set for non-computed orders
                orders = cp.arange(order + 1)[:, cp.newaxis]
                orders = orders + orders.T
                mask = cp.logical_and(orders < 1, orders > order)
                # prepend labels (and channels) axes
                if num_channels > 1:
                    mask = mask[cp.newaxis, cp.newaxis, ...]
                    mask = cp.tile(mask, moments.shape[:2] + (1, 1))
                else:
                    mask = mask[cp.newaxis, ...]
                    mask = cp.tile(mask, moments.shape[:1] + (1, 1))
                assert cp.all(cp.isnan(moments[mask]))

            if norm_type == "hu":
                moments = regionprops_moments_hu(moments)
                assert moments.shape[-1] == 7

    # regionprops does not use the more accurate analytical expressions for the
    # central moments, so need to relax tolerance in the "central" moments case
    rtol = 1e-4 if norm_type != "raw" else 1e-5
    atol = 1e-4 if norm_type != "raw" else 1e-7
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if norm_type == "hu":
        # hu moments are stored as a 7-element vector
        if num_channels == 1:
            for d in range(7):
                allclose(moments[:, d], expected[prop + f"-{d}"])
        else:
            for c in range(num_channels):
                for d in range(7):
                    allclose(moments[:, c, d], expected[prop + f"-{d}-{c}"])
    else:
        # All other moment types produce a (order + 1, order + 1) matrix
        if num_channels == 1:
            # zeroth moment
            allclose(moments[:, 0, 0], expected[prop + "-0-0"])

            if order > 0 and norm_type != "normalized":
                # first-order moments
                if norm_type == "central":
                    assert_array_equal(moments[:, 0, 1], 0.0)
                    assert_array_equal(moments[:, 1, 0], 0.0)
                else:
                    allclose(moments[:, 0, 1], expected[prop + "-0-1"])
                    allclose(moments[:, 1, 0], expected[prop + "-1-0"])
            if order > 1:
                # second-order moments
                allclose(moments[:, 0, 2], expected[prop + "-0-2"])
                allclose(moments[:, 1, 1], expected[prop + "-1-1"])
                allclose(moments[:, 2, 0], expected[prop + "-2-0"])
            if order > 3:
                # third-order moments
                allclose(moments[:, 0, 3], expected[prop + "-0-3"])
                allclose(moments[:, 1, 2], expected[prop + "-1-2"])
                allclose(moments[:, 2, 1], expected[prop + "-2-1"])
                allclose(moments[:, 3, 0], expected[prop + "-3-0"])
        else:
            for c in range(num_channels):
                # zeroth moment
                allclose(moments[:, c, 0, 0], expected[prop + f"-0-0-{c}"])

            if order > 0 and norm_type != "normalized":
                # first-order moments
                if norm_type == "central":
                    assert_array_equal(moments[:, c, 0, 1], 0.0)
                    assert_array_equal(moments[:, c, 1, 0], 0.0)
                else:
                    allclose(moments[:, c, 0, 1], expected[prop + f"-0-1-{c}"])
                    allclose(moments[:, c, 1, 0], expected[prop + f"-1-0-{c}"])
            if order > 1:
                # second-order moments
                allclose(moments[:, c, 0, 2], expected[prop + f"-0-2-{c}"])
                allclose(moments[:, c, 1, 1], expected[prop + f"-1-1-{c}"])
                allclose(moments[:, c, 2, 0], expected[prop + f"-2-0-{c}"])
            if order > 3:
                # third-order moments
                allclose(moments[:, c, 0, 3], expected[prop + f"-0-3-{c}"])
                allclose(moments[:, c, 1, 2], expected[prop + f"-1-2-{c}"])
                allclose(moments[:, c, 2, 1], expected[prop + f"-2-1-{c}"])
                allclose(moments[:, c, 3, 0], expected[prop + f"-3-0-{c}"])


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5, 0.75)])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "weighted, intensity_dtype, num_channels",
    [
        (False, None, 1),
        (True, cp.float32, 1),
        (True, cp.uint8, 3),
    ],
)
@pytest.mark.parametrize("norm_type", ["raw", "central", "normalized"])
def test_moments_3d(
    spacing, order, weighted, intensity_dtype, num_channels, norm_type
):
    shape = (80, 64, 48)
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels))
    kwargs = {"spacing": spacing}
    prop = "moments"
    if norm_type == "normalized" and order < 2:
        pytest.skip("normalized case only supports order >=2")
    if weighted:
        intensity_image = get_intensity_image(
            shape, dtype=intensity_dtype, num_channels=num_channels
        )
        kwargs["intensity_image"] = intensity_image
        prop += "_weighted"
    if norm_type == "central":
        prop += "_central"
    elif norm_type == "normalized":
        prop += "_normalized"
    kwargs_cpu = deepcopy(kwargs)
    if "intensity_image" in kwargs_cpu:
        kwargs_cpu["intensity_image"] = cp.asnumpy(
            kwargs_cpu["intensity_image"]
        )

    # ignore possible warning from skimage implementation
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar power",
        category=RuntimeWarning,
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=[prop], **kwargs_cpu
    )
    warnings.resetwarnings()

    moments = regionprops_moments(
        labels, max_label=max_label, order=order, **kwargs
    )
    if norm_type in ["central", "normalized"]:
        ndim = len(shape)
        moments = regionprops_moments_central(moments, ndim=ndim)
        if norm_type == "normalized":
            moments = regionprops_moments_normalized(
                moments, ndim=ndim, spacing=spacing
            )

            # assert that np.nan values were set for non-computed orders
            orders = cp.arange(order + 1)
            orders = (
                orders[:, cp.newaxis, cp.newaxis]
                + orders[cp.newaxis, :, cp.newaxis]
                + orders[cp.newaxis, cp.newaxis, :]
            )
            mask = cp.logical_and(orders < 1, orders > order)
            # prepend labels (and channels) axes and replicate mask to match
            # the moments shape
            if num_channels > 1:
                mask = mask[cp.newaxis, cp.newaxis, ...]
                mask = cp.tile(mask, moments.shape[:2] + (1, 1, 1))
            else:
                mask = mask[cp.newaxis, ...]
                mask = cp.tile(mask, moments.shape[:1] + (1, 1, 1))
            assert cp.all(cp.isnan(moments[mask]))

    # regionprops does not use the more accurate analytical expressions for the
    # central moments, so need to relax tolerance in the "central" moments case
    rtol = 1e-3 if norm_type != "raw" else 1e-5
    atol = 1e-4 if norm_type != "raw" else 1e-7
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if num_channels == 1:
        # zeroth moment
        allclose(moments[:, 0, 0, 0], expected[prop + "-0-0-0"])
        if order > 0 and norm_type != "normalized":
            # first-order moments
            if norm_type == "central":
                assert_array_equal(moments[:, 0, 0, 1], 0.0)
                assert_array_equal(moments[:, 0, 1, 0], 0.0)
                assert_array_equal(moments[:, 1, 0, 0], 0.0)
            else:
                allclose(moments[:, 0, 0, 1], expected[prop + "-0-0-1"])
                allclose(moments[:, 0, 1, 0], expected[prop + "-0-1-0"])
                allclose(moments[:, 1, 0, 0], expected[prop + "-1-0-0"])
        if order > 1:
            # second-order moments
            allclose(moments[:, 0, 0, 2], expected[prop + "-0-0-2"])
            allclose(moments[:, 0, 2, 0], expected[prop + "-0-2-0"])
            allclose(moments[:, 2, 0, 0], expected[prop + "-2-0-0"])
            allclose(moments[:, 1, 1, 0], expected[prop + "-1-1-0"])
            allclose(moments[:, 1, 0, 1], expected[prop + "-1-0-1"])
            allclose(moments[:, 0, 1, 1], expected[prop + "-0-1-1"])
        if order > 2:
            # third-order moments
            allclose(moments[:, 0, 0, 3], expected[prop + "-0-0-3"])
            allclose(moments[:, 0, 3, 0], expected[prop + "-0-3-0"])
            allclose(moments[:, 3, 0, 0], expected[prop + "-3-0-0"])
            allclose(moments[:, 1, 2, 0], expected[prop + "-1-2-0"])
            allclose(moments[:, 2, 1, 0], expected[prop + "-2-1-0"])
            allclose(moments[:, 1, 0, 2], expected[prop + "-1-0-2"])
            allclose(moments[:, 2, 0, 1], expected[prop + "-2-0-1"])
            allclose(moments[:, 0, 1, 2], expected[prop + "-0-1-2"])
            allclose(moments[:, 0, 2, 1], expected[prop + "-0-2-1"])
            allclose(moments[:, 1, 1, 1], expected[prop + "-1-1-1"])
    else:
        for c in range(num_channels):
            # zeroth moment
            allclose(moments[:, c, 0, 0, 0], expected[prop + f"-0-0-0-{c}"])
            if order > 0 and norm_type != "normalized":
                # first-order moments
                if norm_type == "central":
                    assert_array_equal(moments[:, c, 0, 0, 1], 0.0)
                    assert_array_equal(moments[:, c, 0, 1, 0], 0.0)
                    assert_array_equal(moments[:, c, 1, 0, 0], 0.0)
                else:
                    allclose(
                        moments[:, c, 0, 0, 1], expected[prop + f"-0-0-1-{c}"]
                    )
                    allclose(
                        moments[:, c, 0, 1, 0], expected[prop + f"-0-1-0-{c}"]
                    )
                    allclose(
                        moments[:, c, 1, 0, 0], expected[prop + f"-1-0-0-{c}"]
                    )
            if order > 1:
                # second-order moments
                allclose(moments[:, c, 0, 0, 2], expected[prop + f"-0-0-2-{c}"])
                allclose(moments[:, c, 0, 2, 0], expected[prop + f"-0-2-0-{c}"])
                allclose(moments[:, c, 2, 0, 0], expected[prop + f"-2-0-0-{c}"])
                allclose(moments[:, c, 1, 1, 0], expected[prop + f"-1-1-0-{c}"])
                allclose(moments[:, c, 1, 0, 1], expected[prop + f"-1-0-1-{c}"])
                allclose(moments[:, c, 0, 1, 1], expected[prop + f"-0-1-1-{c}"])
            if order > 2:
                # third-order moments
                allclose(moments[:, c, 0, 0, 3], expected[prop + f"-0-0-3-{c}"])
                allclose(moments[:, c, 0, 3, 0], expected[prop + f"-0-3-0-{c}"])
                allclose(moments[:, c, 3, 0, 0], expected[prop + f"-3-0-0-{c}"])
                allclose(moments[:, c, 1, 2, 0], expected[prop + f"-1-2-0-{c}"])
                allclose(moments[:, c, 2, 1, 0], expected[prop + f"-2-1-0-{c}"])
                allclose(moments[:, c, 1, 0, 2], expected[prop + f"-1-0-2-{c}"])
                allclose(moments[:, c, 2, 0, 1], expected[prop + f"-2-0-1-{c}"])
                allclose(moments[:, c, 0, 1, 2], expected[prop + f"-0-1-2-{c}"])
                allclose(moments[:, c, 0, 2, 1], expected[prop + f"-0-2-1-{c}"])
                allclose(moments[:, c, 1, 1, 1], expected[prop + f"-1-1-1-{c}"])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (0.5, 1.5, 0.8)])
def test_inertia_tensor_eigenvectors(ndim, spacing):
    # the columns of inertia_tensor_eigenvectors are the eigenvectors of
    # inertia_tensor, in the order of inertia_tensor_eigvals (descending)
    shape = (256, 512) if ndim == 2 else (48, 64, 40)
    labels = get_labels_nd(shape, blob_size_fraction=0.12)
    if spacing is not None:
        spacing = spacing[:ndim]
    out = regionprops_dict(
        labels,
        properties=[
            "inertia_tensor",
            "inertia_tensor_eigvals",
            "inertia_tensor_eigenvectors",
        ],
        spacing=spacing,
    )
    tensor = cp.asnumpy(out["inertia_tensor"])
    eigvals = cp.asnumpy(out["inertia_tensor_eigvals"])
    eigvecs = cp.asnumpy(out["inertia_tensor_eigenvectors"])
    assert eigvecs.shape == tensor.shape
    present = np.isfinite(eigvals).all(axis=1) & (eigvals[:, 0] > 0)
    assert present.sum() > 10
    for t, w, v in zip(tensor[present], eigvals[present], eigvecs[present]):
        scale = np.abs(t).max()
        assert_allclose(t @ v, v * w, atol=1e-9 * scale)
        assert_allclose(v.T @ v, np.eye(ndim), atol=1e-12)
        assert np.all(np.diff(w) <= 0)


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5, 1.2)])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("shape", [(500, 400), (64, 96, 32)])
@pytest.mark.parametrize("compute_orientation", [False, True])
@pytest.mark.parametrize("compute_axis_lengths", [False, True])
@pytest.mark.parametrize("blob_size_fraction", [0.03, 0.1, 0.3])
def test_inertia_tensor(
    shape,
    spacing,
    order,
    compute_orientation,
    compute_axis_lengths,
    blob_size_fraction,
):
    ndim = len(shape)
    labels = get_labels_nd(shape, blob_size_fraction=blob_size_fraction)
    max_label = int(cp.max(labels))
    if spacing is not None:
        # omit 3rd element for 2d images
        spacing = spacing[:ndim]
    props = ["inertia_tensor", "inertia_tensor_eigvals"]
    compute_eccentricity = True if ndim == 2 else False
    if compute_eccentricity:
        props += ["eccentricity"]
    if compute_orientation:
        props += ["orientation"]
    if compute_axis_lengths:
        props += ["axis_major_length", "axis_minor_length"]
    moments_raw = regionprops_moments(
        labels, max_label=max_label, order=order, spacing=spacing
    )
    moments_central = regionprops_moments_central(moments_raw, ndim=ndim)

    itensor_kwargs = dict(ndim=ndim, compute_orientation=compute_orientation)
    if order < 2:
        # can't compute inertia tensor without 2nd order moments
        with pytest.raises(ValueError):
            regionprops_inertia_tensor(moments_central, **itensor_kwargs)
        return

    if compute_orientation:
        if ndim != 2:
            with pytest.raises(ValueError):
                regionprops_inertia_tensor(moments_central, **itensor_kwargs)
            return
        itensor, orientation = regionprops_inertia_tensor(
            moments_central, **itensor_kwargs
        )
        assert orientation.shape == itensor.shape[:-2]
    else:
        itensor = regionprops_inertia_tensor(moments_central, **itensor_kwargs)

    assert itensor.shape[-2:] == (ndim, ndim)

    props_dict = regionprops_inertia_tensor_eigvals(
        itensor,
        compute_axis_lengths=compute_axis_lengths,
        compute_eccentricity=compute_eccentricity,
    )
    eigvals = props_dict["inertia_tensor_eigvals"]
    assert eigvals.shape == (max_label, ndim)
    if compute_eccentricity:
        eccentricity = props_dict["eccentricity"]
        assert eccentricity.shape == (max_label,)
    if compute_axis_lengths:
        axis_lengths = props_dict["axis_lengths"]
        assert axis_lengths.shape == (max_label, ndim)

    # Do not compare to scikit-image via measure_cpu due to unhandled
    # ValueError: math domain error in scikit-image. (Floating point numeric
    # error can cause square root of very slightly negative value)
    expected = measure.regionprops_table(
        labels,
        properties=props,
        spacing=spacing,
    )

    # regionprops does not use the more accurate analytical expressions for the
    # central moments, so need to relax tolerance in the "central" moments case
    rtol = 1e-4
    atol = 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if ndim == 2:
        # valida inertia tensor
        allclose(itensor[:, 0, 0], expected["inertia_tensor-0-0"])
        allclose(itensor[:, 0, 1], expected["inertia_tensor-0-1"])
        allclose(itensor[:, 1, 0], expected["inertia_tensor-1-0"])
        allclose(itensor[:, 1, 1], expected["inertia_tensor-1-1"])

        # validate eigenvalues
        allclose(eigvals[:, 0], expected["inertia_tensor_eigvals-0"])
        allclose(eigvals[:, 1], expected["inertia_tensor_eigvals-1"])

        if compute_orientation:
            pass
            # Disabled orientation comparison as it is currently not robust
            # (fails for the spacing != None case)
            #
            # # validate orientation
            # # use sin/cos to avoid PI and -PI from being considered different
            # tol_kw = dict(rtol=1e-3, atol=1e-3)
            # assert_allclose(
            #     cp.cos(orientation), cp.cos(expected["orientation"]), **tol_kw
            # )
            # assert_allclose(
            #     cp.sin(orientation), cp.sin(expected["orientation"]), **tol_kw
            # )
        if compute_eccentricity:
            allclose(eccentricity, expected["eccentricity"])

    elif ndim == 3:
        # valida inertia tensor
        allclose(itensor[:, 0, 0], expected["inertia_tensor-0-0"])
        allclose(itensor[:, 0, 1], expected["inertia_tensor-0-1"])
        allclose(itensor[:, 0, 2], expected["inertia_tensor-0-2"])
        allclose(itensor[:, 1, 0], expected["inertia_tensor-1-0"])
        allclose(itensor[:, 1, 1], expected["inertia_tensor-1-1"])
        allclose(itensor[:, 1, 2], expected["inertia_tensor-1-2"])
        allclose(itensor[:, 2, 0], expected["inertia_tensor-2-0"])
        allclose(itensor[:, 2, 1], expected["inertia_tensor-2-1"])
        allclose(itensor[:, 2, 2], expected["inertia_tensor-2-2"])

        # validate eigenvalues
        allclose(eigvals[:, 0], expected["inertia_tensor_eigvals-0"])
        allclose(eigvals[:, 1], expected["inertia_tensor_eigvals-1"])
        allclose(eigvals[:, 2], expected["inertia_tensor_eigvals-2"])
    rtol = 1e-5
    # seems to be a larger fractional pixel error in length in 3D case
    atol = 5e-3 if ndim == 3 else 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if compute_axis_lengths:
        allclose(axis_lengths[..., 0], expected["axis_major_length"])
        allclose(axis_lengths[..., -1], expected["axis_minor_length"])


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5, 1.2)])
@pytest.mark.parametrize(
    "intensity_dtype, num_channels",
    [(cp.float32, 1), (cp.uint8, 3)],
)
@pytest.mark.parametrize("shape", [(800, 600), (80, 60, 40)])
@pytest.mark.parametrize("local", [False, True])
def test_centroid_weighted(
    shape, spacing, intensity_dtype, num_channels, local
):
    ndim = len(shape)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels))
    if spacing is not None:
        # omit 3rd element for 2d images
        spacing = spacing[:ndim]
    intensity_image = get_intensity_image(
        shape, dtype=intensity_dtype, num_channels=num_channels
    )
    kwargs = {"spacing": spacing, "intensity_image": intensity_image}
    prop = "centroid_weighted"
    if local:
        prop += "_local"

    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        properties=[prop],
        spacing=spacing,
        intensity_image=cp.asnumpy(intensity_image),
    )
    moments_raw = regionprops_moments(
        labels, max_label=max_label, order=1, **kwargs
    )

    if local:
        bbox = None
    else:
        bbox, _ = regionprops_bbox_coords(
            labels, max_label=max_label, return_slices=False
        )

    centroids = regionprops_centroid_weighted(
        moments_raw,
        ndim=ndim,
        bbox=bbox,
        compute_local=local,
        compute_global=not local,
        spacing=spacing,
    )[prop]

    assert centroids.shape[-1] == ndim

    rtol = 1e-7
    atol = 0
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if num_channels == 1:
        for d in range(ndim):
            allclose(centroids[:, d], expected[prop + f"-{d}"])
    else:
        for c in range(num_channels):
            for d in range(ndim):
                allclose(centroids[:, c, d], expected[prop + f"-{d}-{c}"])


@pytest.mark.parametrize("shape", [(256, 512), (4096, 1024)])
@pytest.mark.parametrize("volume_fraction", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("blob_size_fraction", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("robust", [False, True])
def test_perimeter(shape, robust, volume_fraction, blob_size_fraction):
    labels = get_labels_nd(
        shape,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
    )

    max_label = int(cp.max(labels))
    if not robust:
        # remove any regions that are to close for non-robust algorithm
        labels_close = _find_close_labels(labels, labels > 0, max_label)
        for value in labels_close:
            labels[labels == value] = 0

        # relabel to ensure sequential
        labels = measure.label(labels > 0)
        max_label = int(cp.max(labels))

    max_label = int(cp.max(labels))
    values = regionprops_perimeter(labels, max_label=max_label, robust=robust)
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["perimeter"]
    )
    rtol = 1e-5
    atol = 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    allclose(values, expected["perimeter"])


@pytest.mark.parametrize("shape", [(256, 512), (4096, 1024)])
@pytest.mark.parametrize("volume_fraction", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("blob_size_fraction", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("robust", [False, True])
def test_perimeter_crofton(shape, robust, volume_fraction, blob_size_fraction):
    labels = get_labels_nd(
        shape,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
    )

    max_label = int(cp.max(labels))
    if not robust:
        # remove any regions that are to close for non-robust algorithm
        labels_close = _find_close_labels(labels, labels > 0, max_label)
        for value in labels_close:
            labels[labels == value] = 0

        # relabel to ensure sequential
        labels = measure.label(labels > 0)
        max_label = int(cp.max(labels))

    values = regionprops_perimeter_crofton(
        labels, max_label=max_label, robust=robust
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["perimeter_crofton"]
    )
    rtol = 1e-5
    atol = 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    allclose(values, expected["perimeter_crofton"])


@pytest.mark.parametrize(
    "shape, connectivity",
    [
        ((256, 512), None),
        ((256, 512), 1),
        ((256, 512), 2),
        ((96, 64, 48), None),
        ((96, 64, 48), 1),
        ((96, 64, 48), 3),
    ],
)
@pytest.mark.parametrize("volume_fraction", [0.25])
@pytest.mark.parametrize("blob_size_fraction", [0.04])
@pytest.mark.parametrize("insert_holes", [False, True])
@pytest.mark.parametrize("robust", [False, True])
def test_euler(
    shape,
    connectivity,
    robust,
    volume_fraction,
    blob_size_fraction,
    insert_holes,
):
    labels = get_labels_nd(
        shape,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        insert_holes=insert_holes,
    )

    max_label = int(cp.max(labels))
    if not robust:
        # remove any regions that are to close for non-robust algorithm
        labels_close = _find_close_labels(labels, labels > 0, max_label)
        for value in labels_close:
            labels[labels == value] = 0

        # relabel to ensure sequential
        labels = measure.label(labels > 0)
        max_label = int(cp.max(labels))

    values = regionprops_euler(
        labels, connectivity, max_label=max_label, robust=robust
    )
    if connectivity is None or connectivity == labels.ndim:
        # regionprops_table has hard-coded connectivity, can't use it to verify
        # other values
        expected = measure_cpu.regionprops_table(
            cp.asnumpy(labels), properties=["euler_number"]
        )
        assert_array_equal(values, expected["euler_number"])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize(
    "blob_kwargs", [{}, dict(blob_size_fraction=0.12, volume_fraction=0.3)]
)
def test_image(ndim, num_channels, blob_kwargs):
    shape = (256, 512) if ndim == 2 else (64, 64, 80)

    labels = get_labels_nd(shape, **blob_kwargs)
    intensity_image = get_intensity_image(
        shape, dtype=cp.uint16, num_channels=num_channels
    )
    max_label = int(cp.max(labels))
    images, intensity_images, images_convex = regionprops_image(
        labels,
        max_label=max_label,
        intensity_image=intensity_image,
        compute_convex=True,
    )
    assert len(images) == max_label
    assert len(intensity_images) == max_label
    assert len(images_convex) == max_label

    # suppress any QHull warnings coming from the scikit-image implementation
    warnings.filterwarnings(
        "ignore",
        message="Failed to get convex hull image",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="divide by zero",
        category=RuntimeWarning,
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        intensity_image=cp.asnumpy(intensity_image),
        properties=["image", "image_intensity", "image_convex"],
    )
    warnings.resetwarnings()

    for n in range(max_label):
        assert_array_equal(images[n], expected["image"][n])
        assert_array_equal(intensity_images[n], expected["image_intensity"][n])
        # Note if 3d blobs are size 1 on one of the axes, it can cause QHull to
        # fail and return a zeros convex image for that label. This has been
        # resolved for cuCIM, but not yet for scikit-image.
        # The test case with blob_kwargs != {} was chosen as a known good
        # setting where such an edge case does NOT occur.
        if blob_kwargs:
            assert_array_equal(images_convex[n], expected["image_convex"][n])
        else:
            # Can't compare to scikit-image in this case
            # Just make sure the convex size is not smaller than the original
            assert (images_convex[n].sum()) >= (images[n].sum())


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (1, 1, 1), (1.5, 0.5, 0.76)])
def test_coords(ndim, spacing):
    shape = (256, 512) if ndim == 2 else (35, 63, 37)
    if spacing is not None:
        spacing = spacing[:ndim]
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels))
    coords, coords_scaled = regionprops_coords(
        labels,
        max_label=max_label,
        spacing=spacing,
        compute_coords=True,
        compute_coords_scaled=True,
    )
    assert len(coords) == max_label
    assert len(coords_scaled) == max_label

    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        spacing=spacing,
        properties=["coords", "coords_scaled"],
    )
    for n in range(max_label):
        # cast to Python int to match dtype from CPU case
        assert_array_equal(coords[n].astype(int), expected["coords"][n])

        assert_allclose(
            coords_scaled[n], expected["coords_scaled"][n], rtol=1e-5
        )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (1, 1, 1), (0.5, 0.35, 0.75)])
@pytest.mark.parametrize(
    "blob_kwargs", [dict(blob_size_fraction=0.15, volume_fraction=0.1)]
)
def test_feret_diameter_max(ndim, spacing, blob_kwargs):
    shape = (1024, 2048) if ndim == 2 else (64, 80, 48)
    # use dilate blobs to avoid error from singleton dimension regions in
    # scikit-image
    labels = get_labels_nd(shape, dilate_blobs=ndim == 3, **blob_kwargs)
    # discard any extra dimensions from spacing
    if spacing is not None:
        spacing = spacing[:ndim]

    max_label = int(cp.max(labels))
    _, _, images_convex = regionprops_image(
        labels,
        max_label=max_label,
        compute_convex=True,
    )
    feret_diameters = regionprops_feret_diameter_max(
        images_convex,
        spacing=spacing,
    )

    # suppress any QHull warnings coming from the scikit-image implementation
    warnings.filterwarnings(
        "ignore",
        message="Failed to get convex hull image",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="divide by zero",
        category=RuntimeWarning,
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        spacing=spacing,
        properties=["num_pixels", "feret_diameter_max"],
    )
    warnings.resetwarnings()

    # print(f"{ndim=}, {spacing=}, {max_label=}")
    # print(f"num_pixels={expected['num_pixels']}")
    # print(f"diameters={expected['feret_diameter_max']}")
    max_diff = np.max(
        np.abs(feret_diameters.get() - expected["feret_diameter_max"])
    )
    # print(f"max_diff = {max_diff}")
    assert max_diff < math.sqrt(ndim)


@pytest.mark.parametrize("ndim", [2, 3])
def test_feret_diameter_max_absent_label(ndim):
    # label 2 does not occur: its row gets 0, like its area_convex
    labels = cp.zeros((12,) * ndim, dtype=cp.uint8)
    labels[(slice(1, 4),) * ndim] = 1
    labels[(slice(6, 11),) * ndim] = 3
    props = regionprops_dict(
        labels, properties=["area_convex", "feret_diameter_max"]
    )
    labels[labels == 3] = 2
    expected = regionprops_dict(labels, properties=["feret_diameter_max"])
    expected = expected["feret_diameter_max"]
    assert_array_equal(props["area_convex"][1], 0)
    assert_array_equal(
        props["feret_diameter_max"],
        cp.stack([expected[0], cp.zeros_like(expected[0]), expected[1]]),
    )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (1.5, 0.5, 0.76)])
@pytest.mark.parametrize(
    "property_name",
    (
        list(basic_deps.keys())
        + list(convex_deps.keys())
        + list(intensity_deps.keys())
        + list(misc_deps.keys())
        + list(moment_deps.keys())
    ),
)
def test_regionprops_dict_single_property(ndim, spacing, property_name):
    """Test to verify that any dependencies for a given property are
    automatically handled.
    """
    if ndim != 2 and property_name in ndim_2_only:
        pytest.skip(f"{property_name} is for 2d images only.")
        return
    shape = (768, 512) if ndim == 2 else (64, 64, 64)
    if spacing is not None:
        spacing = spacing[:ndim]
    labels = get_labels_nd(shape)
    if property_name in need_intensity_image:
        intensity_image = get_intensity_image(
            shape, dtype=cp.uint16, num_channels=1
        )
    else:
        intensity_image = None
    if spacing is not None and property_name in [
        "moments_hu",
        "moments_weighted_hu",
        "perimeter",
        "perimeter_crofton",
    ]:
        # as in scikit-image, Hu moments require unit spacing and the
        # perimeters isotropic spacing (this test's spacing is anisotropic)
        with pytest.raises(NotImplementedError):
            regionprops_dict(
                labels,
                intensity_image,
                properties=[property_name],
                spacing=spacing,
            )
        return
    props = regionprops_dict(
        labels, intensity_image, properties=[property_name], spacing=spacing
    )
    assert property_name in props
    # any unrequested dependent properties are not retained in the output dict
    assert len(props) == 1


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [(2, 2, 2), (1.5, 0.5, 0.76)])
@pytest.mark.parametrize(
    "property_name",
    [
        "area_bbox",
        "area_convex",
        "centroid",
        "centroid_local",
        "centroid_weighted",
        "centroid_weighted_local",
        "extent",
        "moments_normalized",
        "moments_weighted_normalized",
        "solidity",
    ],
)
def test_regionprops_dict_spacing(ndim, spacing, property_name):
    """Batch properties with non-unit spacing must match scikit-image."""
    if ndim != 2 and property_name in ndim_2_only:
        pytest.skip(f"{property_name} is for 2d images only.")
    shape = (256, 512) if ndim == 2 else (48, 64, 64)
    spacing = spacing[:ndim]
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(shape, dtype=cp.float32)
    intensity_image -= intensity_image.min() - 1  # positive weights
    # scikit-image computes moment matrices up to order 3
    is_moments = property_name.startswith("moments")
    result = regionprops_dict(
        labels,
        intensity_image,
        properties=[property_name],
        spacing=spacing,
        moment_order=3 if is_moments else None,
        to_table=True,
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        cp.asnumpy(intensity_image),
        properties=[property_name],
        spacing=spacing,
    )
    assert set(result) == set(expected)
    for key in expected:
        if is_moments and sum(int(i) for i in key.split("-")[1:]) > 3:
            # entries above the computed order are not computed (NaN or 0)
            continue
        valid = slice(None)
        if property_name in ["area_convex", "solidity"]:
            # scikit-image's QHull-based convex hull returns an empty image
            # for 3D regions that are flat along one axis (see the note in
            # test_area_convex_and_solidity); skip those regions
            valid = np.isfinite(expected[key]) & (expected[key] != 0)
        assert_allclose(
            result[key][valid], expected[key][valid], rtol=1e-5, atol=1e-8
        )


def _filled_case(name):
    """Small label images for checking the filled-region properties."""
    if name == "non_raster_order":
        labels = cp.zeros((20, 20), dtype=cp.uint8)
        labels[2:8, 2:8] = 2
        labels[4:6, 4:6] = 0  # hole in label 2
        labels[12:18, 12:18] = 1
        labels[14:16, 14:16] = 0  # hole in label 1
        labels[10, 2:6] = 3
    elif name == "touching":
        labels = cp.zeros((12, 20), dtype=cp.uint8)
        labels[2:10, 2:10] = 1
        labels[5:7, 5:7] = 0
        labels[2:10, 10:18] = 2
    elif name == "nested_with_gap":
        labels = cp.zeros((16, 16), dtype=cp.uint8)
        labels[2:14, 2:14] = 1
        labels[4:12, 4:12] = 0
        labels[6:10, 6:10] = 2  # inside the hole of label 1
    elif name == "nested_touching":
        labels = cp.zeros((16, 16), dtype=cp.uint8)
        labels[2:14, 2:14] = 1
        labels[5:11, 5:11] = 2  # touches label 1 on all sides
        labels[7:9, 7:9] = 0  # hole in label 2
    elif name == "nested_disconnected":
        labels = cp.zeros((12, 12), dtype=cp.uint8)
        labels[2:10, 2:10] = 1
        labels[4:8, 4:8] = 2  # fills the hole of label 1
        labels[0, 0] = 2  # so the bounding box of label 2 is not inside 1's
    elif name == "nested_diagonal_gap":
        labels = cp.zeros((12, 12), dtype=cp.uint8)
        labels[2:10, 2:10] = 1
        labels[3:9, 3:9] = 0
        labels[2, 2] = 0  # the inside of label 1 is open diagonally
        labels[6:9, 6:9] = 2  # touches label 1 (which is then recomputed)
    elif name == "hole_between_two_labels":
        labels = cp.zeros((12, 16), dtype=cp.uint8)
        labels[2:10, 2:8] = 1
        labels[2:10, 8:14] = 2
        labels[4:8, 5:11] = 0  # hole bordered by labels 1 and 2
    elif name == "3d":
        labels = cp.zeros((12, 14, 16), dtype=cp.uint8)
        labels[1:11, 1:11, 1:11] = 1
        labels[3:9, 3:9, 3:9] = 0  # cavity of label 1
        labels[5:7, 5:7, 5:7] = 2  # nested in the cavity
        labels[2:6, 12:14, 2:14] = 3
        labels[3:5, 12:14, 4:6] = 0  # not a hole: open along axis 1
    else:
        raise ValueError(name)
    return labels


FILLED_CASES = [
    "non_raster_order",
    "touching",
    "nested_with_gap",
    "nested_touching",
    "nested_disconnected",
    "nested_diagonal_gap",
    "hole_between_two_labels",
    "3d",
]


@pytest.mark.parametrize("case", FILLED_CASES)
def test_regionprops_dict_filled(case):
    labels = _filled_case(case)
    result = regionprops_dict(
        labels, properties=["area_filled", "num_pixels_filled", "image_filled"]
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["area_filled", "image_filled"]
    )
    assert_allclose(result["area_filled"], expected["area_filled"])
    assert_allclose(result["num_pixels_filled"], expected["area_filled"])
    assert len(result["image_filled"]) == len(expected["image_filled"])
    for image, image_expected in zip(
        result["image_filled"], expected["image_filled"]
    ):
        assert_array_equal(image, image_expected)
    # num_pixels_filled requested alone (used to count the unfilled labels)
    num_pixels_filled = regionprops_dict(
        labels, properties=["num_pixels_filled"]
    )["num_pixels_filled"]
    assert_allclose(num_pixels_filled, expected["area_filled"])


@pytest.mark.parametrize("ndim", [2, 3])
def test_regionprops_dict_filled_random(ndim):
    shape = (256, 512) if ndim == 2 else (48, 64, 64)
    labels = get_labels_nd(shape, insert_holes=True)
    result = regionprops_dict(labels, properties=["area_filled"])
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["area_filled"]
    )
    assert_allclose(result["area_filled"], expected["area_filled"])


def test_label_filled():
    labels = _filled_case("non_raster_order")
    label_filled = regionprops_dict(labels, properties=["label_filled"])[
        "label_filled"
    ]
    # same shape and label values as the input (no padding, no relabeling)
    assert label_filled.shape == labels.shape
    expected = labels.copy()
    expected[4:6, 4:6] = 2
    expected[14:16, 14:16] = 1
    assert_array_equal(label_filled, expected)


def test_fill_holes_by_label_tiling_is_exact():
    # touching labels without nesting need no per-label recomputation
    labels = cp.arange(1, 65, dtype=cp.int32).reshape(8, 8)
    labels = cp.kron(labels, cp.ones((5, 4), dtype=cp.int32))
    label_filled, labels_inexact = _fill_holes_by_label(labels, 64)
    assert_array_equal(label_filled, labels)
    assert labels_inexact.size == 0
    # a label nested in another one is reported
    labels[1:4, 1:3] = 100
    _, labels_inexact = _fill_holes_by_label(labels, 100)
    assert list(labels_inexact) == [1]


@pytest.mark.parametrize("spacing", [None, (2, 3)])
def test_regionprops_dict_area_filled_spacing(spacing):
    labels = cp.zeros((20, 30), dtype=cp.uint8)
    labels[2:8, 2:9] = 1
    labels[4:6, 4:6] = 0  # a hole in label 1
    labels[11:18, 12:27] = 2
    labels[13:16, 15:17] = 0  # a hole in label 2
    result = regionprops_dict(
        labels, properties=["area_filled"], spacing=spacing
    )["area_filled"]
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["area_filled"], spacing=spacing
    )["area_filled"]
    assert_allclose(result, expected)


@pytest.mark.parametrize("property_name", ["perimeter", "perimeter_crofton"])
def test_regionprops_dict_perimeter_spacing(property_name):
    labels = get_labels_nd((256, 512))
    for spacing in [(2, 2), (0.5, 0.5)]:
        result = regionprops_dict(
            labels, properties=[property_name], spacing=spacing
        )[property_name]
        expected = measure_cpu.regionprops_table(
            cp.asnumpy(labels), properties=[property_name], spacing=spacing
        )[property_name]
        assert_allclose(result, expected, rtol=1e-5)
    # scikit-image raises NotImplementedError for anisotropic spacing
    with pytest.raises(NotImplementedError):
        regionprops_dict(labels, properties=[property_name], spacing=(1, 2))


@pytest.mark.parametrize("property_name", ["moments_hu", "moments_weighted_hu"])
def test_regionprops_dict_hu_spacing_not_implemented(property_name):
    labels = get_labels_nd((64, 64))
    intensity_image = get_intensity_image((64, 64), dtype=cp.float32)
    for spacing in [None, (1, 1)]:
        regionprops_dict(
            labels, intensity_image, properties=[property_name], spacing=spacing
        )
    # scikit-image raises NotImplementedError for non-unit spacing
    with pytest.raises(NotImplementedError):
        regionprops_dict(
            labels, intensity_image, properties=[property_name], spacing=(2, 2)
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_regionprops_dict_axis_lengths(ndim):
    shape = (256, 512) if ndim == 2 else (64, 64, 64)
    labels = get_labels_nd(shape)
    # axis_lengths alone used to raise KeyError
    lengths = regionprops_dict(labels, properties=["axis_lengths"])[
        "axis_lengths"
    ]
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        properties=["axis_major_length", "axis_minor_length"],
    )
    assert lengths.shape == (int(labels.max()), ndim)
    assert_allclose(lengths[:, 0], expected["axis_major_length"], rtol=1e-5)
    assert_allclose(lengths[:, -1], expected["axis_minor_length"], rtol=1e-5)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "property_name",
    [
        "label",
        "image",
        "image_convex",
        "image_intensity",
        "image_filled",
        "coords",
        "coords_scaled",
    ],
)
def test_regionprops_image_and_coords_sequence(ndim, property_name):
    shape = (768, 512) if ndim == 2 else (64, 64, 64)
    spacing = (1.5, 0.5, 0.76)
    if spacing is not None:
        spacing = spacing[:ndim]
    labels = get_labels_nd(shape)
    max_label = int(labels.max())
    if property_name in need_intensity_image:
        intensity_image = get_intensity_image(
            shape, dtype=cp.uint16, num_channels=1
        )
    else:
        intensity_image = None
    props = regionprops_dict(
        labels,
        intensity_image,
        properties=[property_name],
        spacing=spacing,
        max_label=max_label,
    )
    assert property_name in props
    result = props[property_name]
    assert len(result) == max_label

    # compute expected result on CPU
    labels_cpu = cp.asnumpy(labels)
    if intensity_image is not None:
        intensity_image_cpu = cp.asnumpy(intensity_image)
    else:
        intensity_image_cpu = None
    expected = measure_cpu.regionprops_table(
        labels_cpu,
        intensity_image_cpu,
        properties=[property_name],
        spacing=spacing,
    )[property_name]
    assert len(expected) == max_label

    # verify
    if property_name == "label":
        assert_array_equal(expected, result)
    else:
        if property_name == "coords_scaled":
            comparison_func = functools.partial(
                assert_allclose, atol=1e-6, rtol=1e-6
            )
        else:
            comparison_func = assert_array_equal
        for i, (expected_val, val) in enumerate(zip(expected, result)):
            comparison_func(expected_val, val)


@pytest.mark.parametrize(
    "property_name", ["Area", "BoundingBoxArea", "Image", "Slice"]
)
def test_regionprops_dict_deprecated_property_names(property_name):
    shape = (1024, 1024)
    labels = get_labels_nd(shape)
    props = regionprops_dict(labels, properties=[property_name])
    # deprecated name is used in the returned results dict
    assert property_name in props
    # non-deprecated version of the name is not also present
    assert PROPS[property_name] not in props


def test_regionprops_dict_coordinates_beyond_2_32_pixels():
    # The coordinates of pixels whose raveled index is 2**32 or more must not
    # wrap, even though every axis is shorter than 2**32.
    shape = (2**16 + 1, 2**16)  # the last row starts at raveled index 2**32
    if cp.cuda.Device().mem_info[0] < 2 * math.prod(shape):
        pytest.skip("not enough free GPU memory for a 4 GB label image")
    labels = cp.zeros(shape, dtype=cp.uint8)
    rows, cols = shape
    labels[rows - 3 :, cols - 6 :] = 1  # straddles raveled index 2**32
    labels[rows - 1, 100:300] = 2  # entirely beyond raveled index 2**32
    properties = ["bbox", "centroid", "centroid_local", "moments_central"]
    props = regionprops_dict(labels, properties=properties)
    del labels

    assert_array_equal(
        props["bbox"],
        [[rows - 3, cols - 6, rows, cols], [rows - 1, 100, rows, 300]],
    )
    assert_allclose(
        props["centroid"], [[rows - 2, cols - 3.5], [rows - 1, 199.5]]
    )
    # translation-invariant properties match those of the same regions in a
    # small image
    small = cp.zeros((8, 400), dtype=cp.uint8)
    small[1:4, 2:8] = 1
    small[5, 100:300] = 2
    expected = regionprops_dict(small, properties=properties)
    for name in ["centroid_local", "moments_central"]:
        assert_allclose(props[name], expected[name])


def test_centroid_region_of_2_32_pixels():
    # pixel counts of 2**32 or more must not wrap
    shape = (2**16 + 1, 2**16)
    if cp.cuda.Device().mem_info[0] < 2 * math.prod(shape):
        pytest.skip("not enough free GPU memory for a 4 GB label image")
    labels = cp.ones(shape, dtype=cp.uint8)
    rows, cols = shape
    labels[rows - 4 : rows - 1, 7:12] = 2
    centroid = regionprops_centroid(labels, max_label=2)
    centroid_local = regionprops_centroid_local(labels, max_label=2)
    del labels

    # centroid of label 1 from those of the whole image and of label 2
    n1 = rows * cols - 15
    row1 = ((rows - 1) * rows / 2 * cols - 15 * (rows - 3)) / n1
    col1 = ((cols - 1) * cols / 2 * rows - 3 * (7 + 8 + 9 + 10 + 11)) / n1
    assert_allclose(centroid, [[row1, col1], [rows - 3, 9]], rtol=1e-12)
    assert_allclose(centroid_local, [[row1, col1], [1, 2]], rtol=1e-12)
