# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import functools
import itertools
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
from skimage.morphology import convex_hull_image

from cucim.skimage import data, measure
from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.measure import _regionprops_gpu_convex
from cucim.skimage.measure._regionprops import PROPS
from cucim.skimage.measure._regionprops_gpu import (
    equivalent_diameter_area,
    ndim_2_only,
    ndim_3_only,
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
    _regionprops_label_pass,
    basic_deps,
)
from cucim.skimage.measure._regionprops_gpu_convex import (
    _regionprops_row_extents,
    convex_deps,
)
from cucim.skimage.measure._regionprops_gpu_intensity_kernels import (
    _regionprops_intensity,
    intensity_deps,
)
from cucim.skimage.measure._regionprops_gpu_misc_kernels import (
    _CROFTON_3D_WEIGHTS,
    _get_boundary_kernel,
    _get_euler_coefs,
    _get_perimeter_coefs,
    _get_perimeter_crofton_coefs,
    _get_voronoi_crofton_weights,
    _surface_area_groups,
    misc_deps,
)
from cucim.skimage.measure._regionprops_gpu_moments_kernels import (
    get_raw_moments_kernel,
    moment_deps,
)
from cucim.skimage.measure._regionprops_gpu_utils import _find_close_labels


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


def _row_extents_reference(labels, max_label):
    """First and last column of each label in each row of its bbox."""
    labels = cp.asnumpy(labels)
    first, last = [], []
    for lab in range(1, max_label + 1):
        coords = np.nonzero(labels == lab)
        if coords[0].size == 0:
            continue
        crop = labels[tuple(slice(c.min(), c.max() + 1) for c in coords)]
        rows = (crop == lab).reshape(-1, crop.shape[-1])
        nonempty = rows.any(axis=1)
        n_cols = rows.shape[1]
        first.append(np.where(nonempty, rows.argmax(axis=1), n_cols))
        last.append(
            np.where(nonempty, n_cols - 1 - rows[:, ::-1].argmax(axis=1), -1)
        )
    return np.concatenate(first), np.concatenate(last)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "label_dtype", [cp.uint8, cp.int16, cp.int32, cp.int64]
)
@pytest.mark.parametrize("pixels_per_thread", [1, 3, 16])
@pytest.mark.parametrize("kind", ["blobs", "random"])
def test_row_extents(ndim, label_dtype, pixels_per_thread, kind):
    shape = (61, 87) if ndim == 2 else (17, 13, 19)
    if kind == "blobs":
        labels = get_labels_nd(shape, blob_size_fraction=0.2)
    else:
        # labels with several components, rows without pixels in their
        # bounding box, runs continuing from one row to the next, and an
        # absent label value (3)
        rng = np.random.default_rng(0)
        labels = cp.asarray(rng.choice([0, 1, 2, 4, 5], size=shape))
    labels = labels.astype(label_dtype)
    max_label = int(labels.max())
    props = _regionprops_label_pass(labels, max_label, compute_bbox=True)
    row_first, row_last, offsets, num_rows = _regionprops_row_extents(
        labels, props["bbox"], max_label, pixels_per_thread=pixels_per_thread
    )
    expected_first, expected_last = _row_extents_reference(labels, max_label)
    assert num_rows == expected_first.size == int(offsets[-1])
    assert offsets.size == max_label + 1
    # rows without pixels of the label only need row_first > row_last
    nonempty = expected_last >= 0
    assert_array_equal(row_first <= row_last, nonempty)
    assert_array_equal(row_first[nonempty], expected_first[nonempty])
    assert_array_equal(row_last[nonempty], expected_last[nonempty])


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


def _convex_test_labels(kind, ndim):
    """Label images for comparing the convex-hull properties with skimage."""
    if kind == "blobs":
        shape = (256, 512) if ndim == 2 else (48, 64, 40)
        return get_labels_nd(shape)
    if kind == "large_blobs":
        shape = (256, 512) if ndim == 2 else (48, 64, 40)
        return get_labels_nd(
            shape, blob_size_fraction=0.12, volume_fraction=0.3
        )
    rng = np.random.default_rng(0)
    if kind == "lines":
        # thin digital lines in random directions (nearly collinear hull
        # edges), each with a separate pixel of the same label
        shape = (200, 300) if ndim == 2 else (40, 50, 60)
        labels = np.zeros(shape, dtype=np.int32)
        t = np.linspace(0, 1, 2 * max(shape))
        for lab in range(1, 31):
            start, end = (rng.integers(0, shape) for _ in range(2))
            line = start + t[:, np.newaxis] * (end - start)
            labels[tuple(np.round(line).astype(int).T)] = lab
            labels[tuple(rng.integers(0, shape))] = lab
        return cp.asarray(labels)
    if kind == "random":
        # labels with several components and rows without any of their pixels
        # in their bounding box; label 3 is absent
        shape = (37, 29) if ndim == 2 else (11, 9, 13)
        return cp.asarray(rng.choice([0, 0, 1, 2, 4, 5], size=shape))
    raise ValueError(kind)


@pytest.mark.parametrize("spacing", [None, (0.5, 0.35)])
@pytest.mark.parametrize("kind", ["blobs", "large_blobs", "lines", "random"])
def test_area_convex_and_solidity_batch_2d(spacing, kind):
    labels = _convex_test_labels(kind, 2)
    max_label = int(labels.max())
    result = regionprops_dict(
        labels, properties=["area_convex", "solidity"], spacing=spacing
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        properties=["label", "area_convex", "solidity"],
        spacing=spacing,
    )
    present = expected["label"] - 1
    absent = np.setdiff1d(np.arange(max_label), present)
    assert_allclose(
        result["area_convex"][present], expected["area_convex"], rtol=1e-12
    )
    assert_allclose(
        result["solidity"][present], expected["solidity"], rtol=1e-6
    )
    assert_array_equal(result["area_convex"][absent], 0)


@pytest.mark.parametrize("kind", ["blobs", "large_blobs", "lines", "random"])
def test_image_convex_batch_2d(kind):
    labels = _convex_test_labels(kind, 2)
    max_label = int(labels.max())
    result = regionprops_dict(labels, properties=["image_convex"])
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["label", "image_convex"]
    )
    images = result["image_convex"]
    assert len(images) == max_label
    for lab, image in zip(expected["label"], expected["image_convex"]):
        assert_array_equal(images[lab - 1], image)
    for lab in set(range(1, max_label + 1)) - set(expected["label"]):
        assert images[lab - 1].size == 0


@pytest.mark.parametrize("spacing", [None, (0.5, 0.35, 0.75)])
@pytest.mark.parametrize("kind", ["blobs", "large_blobs", "lines", "random"])
def test_convex_batch_3d(spacing, kind):
    labels = _convex_test_labels(kind, 3)
    max_label = int(labels.max())
    properties = ["image_convex", "area_convex", "solidity"]
    result = regionprops_dict(labels, properties=properties, spacing=spacing)
    with warnings.catch_warnings():
        # scikit-image's qhull fails for flat regions (see below)
        warnings.simplefilter("ignore")
        expected = measure_cpu.regionprops_table(
            cp.asnumpy(labels),
            properties=["label"] + properties,
            spacing=spacing,
        )
    images = result["image_convex"]
    assert len(images) == max_label
    n_compared = 0
    for n, lab in enumerate(expected["label"]):
        if not expected["image_convex"][n].any():
            # scikit-image returns an empty image for regions whose voxel
            # centers are coplanar (test_convex_flat_3d)
            continue
        assert_array_equal(images[lab - 1], expected["image_convex"][n])
        assert_allclose(
            result["area_convex"][lab - 1],
            expected["area_convex"][n],
            rtol=1e-12,
        )
        assert_allclose(
            result["solidity"][lab - 1], expected["solidity"][n], rtol=1e-6
        )
        n_compared += 1
    assert n_compared > 0
    for lab in set(range(1, max_label + 1)) - set(expected["label"]):
        assert images[lab - 1].size == 0
        assert result["area_convex"][lab - 1] == 0


def test_convex_flat_3d():
    # Regions that are flat along an axis get the convex image and Feret
    # diameter of the squeezed 2D region (as cuCIM's convex_hull_image did
    # for the image; scikit-image's qhull fails, it returns an empty image
    # and its feret_diameter_max raises), and a single voxel itself.
    labels = cp.zeros((5, 30, 40), dtype=cp.int32)
    labels[2, 3:20, 5:9] = 1
    labels[2, 10:25, 20:35] = 1  # one plane, two components
    labels[0:4, 7, 1:30:3] = 2  # one row index, gaps along x
    labels[4, 0, 0] = 3
    properties = ["image_convex", "area_convex", "feret_diameter_max"]
    result = regionprops_dict(labels, properties=properties)
    images = result["image_convex"]
    for lab, axis in [(1, 0), (2, 1)]:
        mask = labels == lab
        coords = cp.nonzero(mask)
        crop = mask[
            tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
        ]
        expected = measure_cpu.regionprops_table(
            cp.asnumpy(crop.squeeze(axis)).astype(np.uint8),
            properties=properties,
        )
        image = expected["image_convex"][0]
        assert_array_equal(images[lab - 1].squeeze(axis), image)
        assert result["area_convex"][lab - 1] == image.sum()
        assert_allclose(
            result["feret_diameter_max"][lab - 1],
            expected["feret_diameter_max"][0],
            rtol=1e-12,
        )
    assert_array_equal(images[2], cp.ones((1, 1, 1), dtype=bool))
    assert result["feret_diameter_max"][2] == 1


@pytest.mark.parametrize("kind", ["blobs", "lines", "random"])
def test_convex_hull_candidates_3d(kind):
    # each label's candidates are distinct diamond points of its voxels and
    # include every vertex of their hull
    from scipy.spatial import ConvexHull

    labels = _convex_test_labels(kind, 3)
    max_label = int(labels.max())
    bbox = _regionprops_label_pass(labels, max_label, compute_bbox=True)["bbox"]
    row_first, row_last, offsets, num_rows = _regionprops_row_extents(
        labels, bbox, max_label
    )
    points, point_offsets = _regionprops_gpu_convex._convex_hull_candidates_3d(
        row_first,
        row_last,
        offsets,
        num_rows,
        bbox,
        cp.asnumpy(bbox).astype(np.int64),
        max_label,
    )
    points, point_offsets = cp.asnumpy(points), cp.asnumpy(point_offsets)
    labels, bbox = cp.asnumpy(labels), cp.asnumpy(bbox)
    diamond = np.concatenate((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    for lab in range(1, max_label + 1):
        candidates = points[point_offsets[lab - 1] : point_offsets[lab]]
        coords = np.argwhere(labels == lab)
        if coords.size == 0:
            assert candidates.size == 0
            continue
        candidates = {tuple(p) for p in candidates}
        assert len(candidates) == point_offsets[lab] - point_offsets[lab - 1]
        # all diamond points, in doubled coordinates relative to the bbox
        centers = 2 * (coords - bbox[lab - 1, :3])
        diamond_points = np.unique(
            (centers[:, np.newaxis, :] + diamond).reshape(-1, 3), axis=0
        )
        assert candidates <= {tuple(p) for p in diamond_points}
        hull = ConvexHull(diamond_points)
        assert {tuple(p) for p in diamond_points[hull.vertices]} <= candidates


@pytest.mark.parametrize("max_threads", [None, 256])
@pytest.mark.parametrize("group", ["warp", "block"])
@pytest.mark.parametrize("kind", ["blobs", "large_blobs", "lines", "wide"])
def test_convex_hulls_3d(group, kind, max_threads, monkeypatch):
    # the facets of the device hulls are those of qhull's hulls of the same
    # points (merging qhull's coplanar triangles)
    from scipy.spatial import ConvexHull

    if kind == "wide":
        # a label over 2**13 voxels wide (64-bit products in the scans)
        rng = np.random.default_rng(0)
        labels = cp.zeros((4, 5, 9000), dtype=cp.int32)
        for lab in (1, 2):
            coords = rng.integers(0, labels.shape, size=(12, 3))
            labels[tuple(cp.asarray(coords.T))] = lab
    else:
        labels = _convex_test_labels(kind, 3)
    monkeypatch.setattr(
        _regionprops_gpu_convex,
        "_HULL_3D_BLOCK_POINTS",
        0 if group == "block" else 2**31,
    )
    max_label = int(labels.max())
    bbox = _regionprops_label_pass(labels, max_label, compute_bbox=True)["bbox"]
    row_first, row_last, offsets, num_rows = _regionprops_row_extents(
        labels, bbox, max_label
    )
    points, point_offsets = _regionprops_gpu_convex._convex_hull_candidates_3d(
        row_first,
        row_last,
        offsets,
        num_rows,
        bbox,
        cp.asnumpy(bbox).astype(np.int64),
        max_label,
    )
    if max_threads is not None:
        # as on a device where a block of more than max_threads threads does
        # not fit (as for sm_100 and sm_120 with the ~250 registers per thread
        # of an earlier version of the kernel): fewer warps per block
        get_kernel = _regionprops_gpu_convex._get_convex_hulls_3d_kernel

        class LimitedKernel:
            def __init__(self, *args):
                self.kernel = get_kernel(*args)
                self.max_threads_per_block = min(
                    max_threads, self.kernel.max_threads_per_block
                )

            def __call__(self, *args):
                return self.kernel(*args)

        monkeypatch.setattr(
            _regionprops_gpu_convex,
            "_get_convex_hulls_3d_kernel",
            LimitedKernel,
        )
        _, warps = _regionprops_gpu_convex._convex_hulls_3d_kernel(
            points.dtype, bbox.dtype
        )
        assert warps == max_threads // 32
    equations, num_faces = _regionprops_gpu_convex._convex_hulls_3d(
        points, point_offsets, bbox, max_label
    )
    points = cp.asnumpy(points) * 0.5
    point_offsets = cp.asnumpy(point_offsets)
    equations, num_faces = cp.asnumpy(equations), cp.asnumpy(num_faces)

    def planes(eq):
        return {tuple(row) for row in np.round(eq, 9) + 0.0}

    for lab in range(max_label):
        start, end = point_offsets[lab], point_offsets[lab + 1]
        if start == end:
            assert num_faces[lab] == 0
            continue
        hull = ConvexHull(points[start:end])
        result = equations[2 * start : 2 * start + num_faces[lab]]
        assert len(result) == len(planes(result))
        assert planes(result) == planes(hull.equations)


def test_convex_batch_3d_host_hulls(monkeypatch):
    # labels the device hulls cannot handle (here: wider than 5 voxels) get
    # qhull's hulls on the host, with the same convex images
    labels = _convex_test_labels("large_blobs", 3)
    properties = ["image_convex", "area_convex"]
    expected = regionprops_dict(labels, properties=properties)
    monkeypatch.setattr(_regionprops_gpu_convex, "_HULL_3D_MAX_EXTENT", 5)
    result = regionprops_dict(labels, properties=properties)
    for image, expected_image in zip(
        result["image_convex"], expected["image_convex"]
    ):
        assert_array_equal(image, expected_image)
    assert_array_equal(result["area_convex"], expected["area_convex"])


@pytest.mark.parametrize("kind", ["large_blobs", "lines", "random"])
def test_convex_batch_3d_qhull(kind, monkeypatch):
    # convex_hull_3d="qhull" computes every hull with qhull on the host, with
    # the same results as the device hulls
    labels = _convex_test_labels(kind, 3)
    properties = ["image_convex", "area_convex", "feret_diameter_max"]
    expected = regionprops_dict(labels, properties=properties)

    def no_device_hulls(*args):
        raise AssertionError("device hulls computed")

    monkeypatch.setattr(
        _regionprops_gpu_convex, "_convex_hulls_3d", no_device_hulls
    )
    result = regionprops_dict(
        labels, properties=properties, convex_hull_3d="qhull"
    )
    for image, expected_image in zip(
        result["image_convex"], expected["image_convex"]
    ):
        assert_array_equal(image, expected_image)
    assert_array_equal(result["area_convex"], expected["area_convex"])
    assert_allclose(
        result["feret_diameter_max"], expected["feret_diameter_max"], rtol=1e-14
    )
    table = measure.regionprops_table(
        labels, properties=["area_convex"], convex_hull_3d="qhull"
    )
    assert_array_equal(table["area_convex"], expected["area_convex"])
    with pytest.raises(ValueError, match="convex_hull_3d"):
        regionprops_dict(labels, properties=properties, convex_hull_3d="cpu")


@pytest.mark.parametrize("ndim", [2, 3])
def test_convex_batch_64bit_coordinates(ndim, monkeypatch):
    # images with an axis of 2**30 pixels or more use 64-bit row extents and
    # hull vertices; force that path on a small image
    labels = _convex_test_labels("large_blobs", ndim)
    properties = ["image_convex", "area_convex", "feret_diameter_max"]
    spacing = (0.7,) * ndim
    expected = regionprops_dict(labels, properties=properties, spacing=spacing)
    monkeypatch.setattr(
        _regionprops_gpu_convex,
        "_get_row_extent_dtype",
        lambda shape: cp.dtype(cp.int64),
    )
    result = regionprops_dict(labels, properties=properties, spacing=spacing)
    for image, expected_image in zip(
        result["image_convex"], expected["image_convex"]
    ):
        assert_array_equal(image, expected_image)
    assert_array_equal(result["area_convex"], expected["area_convex"])
    assert_allclose(
        result["feret_diameter_max"], expected["feret_diameter_max"], rtol=1e-14
    )


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


@pytest.mark.parametrize(
    "label_dtype", [cp.uint8, cp.int16, cp.uint16, cp.int32, cp.uint64]
)
@pytest.mark.parametrize("pixels_per_thread", [1, 3, 16])
@pytest.mark.parametrize("pattern", ["single", "stripes", "random", "blocks"])
def test_label_run_warp_aggregation(label_dtype, pixels_per_thread, pattern):
    # The last runs of neighboring lanes with the same label are combined
    # with warp shuffles before the atomics. Check the counts, bounding boxes,
    # coordinate sums and intensity statistics against NumPy for groups of
    # lanes of various lengths, a partial last warp, and all label dtypes.
    shape = (37, 1001)  # 37037 pixels: not a multiple of 32 * pixels_per_thread
    rng = np.random.default_rng(0)
    if pattern == "single":
        labels = np.ones(shape, dtype=np.int64)
    elif pattern == "stripes":
        # runs of 1 to 100 pixels with labels cycling through 1..5, so that
        # groups of equal labels have different lengths
        lengths = rng.integers(1, 100, size=2000)
        values = np.arange(lengths.size) % 5 + 1
        labels = np.repeat(values, lengths)[: math.prod(shape)].reshape(shape)
    elif pattern == "blocks":
        # a large region with background blocks of 16 to 48 pixels, so that
        # whole lanes have no labeled pixel, and a second region
        cols = np.arange(shape[1])
        labels = np.broadcast_to((cols // 16) % 3 != 1, shape).astype(int)
        labels[:5, :500] *= 2
    else:
        labels = rng.integers(0, 4, size=shape)
    labels_gpu = cp.asarray(labels.astype(label_dtype))
    # positive intensities in odd and negative ones in even labels, so that
    # neither 0 nor any other wrong identity of min/max would go unnoticed
    image = rng.uniform(1, 10, size=shape).astype(np.float32)
    image[labels % 2 == 0] *= -1
    image_gpu = cp.asarray(image)
    max_label = int(labels.max())

    props = _regionprops_label_pass(
        labels_gpu,
        max_label,
        compute_num_pixels=True,
        compute_bbox=True,
        compute_coordinate_sums=True,
        pixels_per_thread=pixels_per_thread,
    )
    _regionprops_intensity(
        labels_gpu,
        image_gpu,
        max_label,
        compute_mean=True,
        compute_min=True,
        compute_max=True,
        pixels_per_thread=pixels_per_thread,
        props_dict=props,
    )
    rr, cc = np.indices(shape)
    for lab in range(1, max_label + 1):
        mask = labels == lab
        assert int(props["num_pixels"][lab - 1]) == mask.sum()
        expected_bbox = [
            rr[mask].min(),
            cc[mask].min(),
            rr[mask].max() + 1,
            cc[mask].max() + 1,
        ]
        assert_array_equal(props["bbox"][lab - 1], expected_bbox)
        assert_array_equal(
            props["coordinate_sums"][lab - 1], [rr[mask].sum(), cc[mask].sum()]
        )
        assert float(props["intensity_min"][lab - 1]) == image[mask].min()
        assert float(props["intensity_max"][lab - 1]) == image[mask].max()
        assert_allclose(
            props["intensity_mean"][lab - 1], image[mask].mean(), rtol=1e-5
        )


@pytest.mark.parametrize("label_dtype", [cp.uint16, cp.int32, cp.int64])
@pytest.mark.parametrize("image_dtype", [cp.float32, cp.float64])
def test_label_run_unaligned_inputs(label_dtype, image_dtype):
    # 4 and 8-byte labels and images are read with 16-byte loads when they
    # are 16-byte aligned; views starting one (odd-length) row in are not,
    # and must give the same results as aligned copies
    shape = (65, 999)
    labels = cp.asarray(
        get_labels_nd(shape, blob_size_fraction=0.1), dtype=label_dtype
    )
    image = get_intensity_image(shape, dtype=image_dtype)
    labels_view, image_view = labels[1:], image[1:]
    assert labels_view.flags.c_contiguous and image_view.flags.c_contiguous
    assert labels_view.data.ptr % 16 != 0 or image_view.data.ptr % 16 != 0
    labels_copy, image_copy = labels_view.copy(), image_view.copy()
    max_label = int(labels.max())
    results = []
    for lab, img in [(labels_view, image_view), (labels_copy, image_copy)]:
        props = _regionprops_label_pass(
            lab,
            max_label,
            compute_num_pixels=True,
            compute_bbox=True,
            compute_coordinate_sums=True,
        )
        _regionprops_intensity(
            lab,
            img,
            max_label,
            compute_std=True,
            compute_min=True,
            compute_max=True,
            props_dict=props,
        )
        regionprops_moments(
            lab, img, max_label=max_label, order=2, props_dict=props
        )
        results.append(props)
    for name, values in results[1].items():
        assert_allclose(results[0][name], values, rtol=1e-12)


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


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("spacing", [None, (0.8, 0.5, 1.3)])
def test_raw_moments_accumulator_types(ndim, order, spacing):
    # the unweighted moments are summed per run in a 32-bit or 64-bit integer
    # type depending on the image size, or in double for huge images; all
    # must give the same result
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    if spacing is not None:
        spacing = spacing[:ndim]
    labels = get_labels_nd(shape)
    max_label = int(labels.max())
    bbox, _ = regionprops_bbox_coords(labels, max_label=max_label)
    results = []
    for accum_c_type in ["unsigned int", "unsigned long long", "double"]:
        kernel = get_raw_moments_kernel(
            ndim,
            order,
            spacing=spacing,
            pixels_per_thread=16,
            accum_c_type=accum_c_type,
        )
        moments = cp.zeros((max_label,) + (order + 1,) * ndim, cp.float64)
        args = (labels, labels.size, bbox)
        if spacing is not None:
            args += (cp.asarray(spacing, dtype=cp.float64),)
        kernel(*args, moments, size=math.ceil(labels.size / 16))
        results.append(moments)
    if spacing is None:
        # integer sums of integers are exact
        assert_array_equal(results[0], results[1])
    # (with spacing, the scaled per-run sums are added by floating-point
    # atomics in a nondeterministic order)
    assert_allclose(results[0], results[1], rtol=1e-14)
    assert_allclose(results[0], results[2], rtol=1e-14)


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


def _crofton_weights_reference(spacing):
    # the weight of each of the 26 neighbor offsets: the fraction of the
    # unit sphere in the Voronoi cells of its spacing-scaled direction and of
    # the opposite direction, whose cells have the same area
    from scipy.spatial import SphericalVoronoi

    spacing = np.ones(3) if spacing is None else np.asarray(spacing, float)
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
    dirs = np.array(offsets, float) * spacing
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    areas = SphericalVoronoi(dirs).calculate_areas() / (4 * math.pi)
    return {o: 2 * a for o, a in zip(offsets, areas)}


def _surface_area_reference(labels, spacing):
    # each region alone, padded with background: every voxel adds the weight
    # of each of its 26 neighbors that is outside the region
    weights = _crofton_weights_reference(spacing)
    spacing = np.ones(3) if spacing is None else np.asarray(spacing)
    labels = cp.asnumpy(labels)
    out = np.zeros(int(labels.max()))
    for lab in range(1, out.size + 1):
        mask = np.pad(labels == lab, 1)
        for k, off in enumerate(itertools.product(range(-1, 2), repeat=3)):
            if any(off):
                weight = weights[off]
                length = np.linalg.norm(np.asarray(off) * spacing)
                coef = 2 * weight * np.prod(spacing) / length
                shifted = np.roll(mask, [-o for o in off], axis=(0, 1, 2))
                out[lab - 1] += coef * np.count_nonzero(mask & ~shifted)
    return out


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kind", ["blobs", "random"])
@pytest.mark.parametrize("spacing", [None, (0.7, 1.3, 1.1)])
def test_oriented_bbox(ndim, kind, spacing):
    labels = _convex_test_labels(kind, ndim)
    if spacing is not None:
        spacing = spacing[:ndim]
    out = regionprops_dict(
        labels,
        properties=[
            "oriented_bbox_size",
            "oriented_bbox_center",
            "inertia_tensor_eigenvectors",
        ],
        spacing=spacing,
    )
    size = cp.asnumpy(out["oriented_bbox_size"])
    center = cp.asnumpy(out["oriented_bbox_center"])
    eigvecs = cp.asnumpy(out["inertia_tensor_eigenvectors"])
    scale = np.ones(ndim) if spacing is None else np.asarray(spacing)
    corners = np.array(list(itertools.product((-0.5, 0.5), repeat=ndim)))
    labels_cpu = cp.asnumpy(labels)
    for region in measure_cpu.regionprops(labels_cpu):
        lab = region.label - 1
        v = eigvecs[lab]
        # the extremes of the pixel centers along each axis, widened by the
        # extent of one pixel
        proj = (region.coords * scale) @ v
        expected_size = (
            proj.max(axis=0) - proj.min(axis=0) + np.abs(v).T @ scale
        )
        expected_center = v @ (0.5 * (proj.max(axis=0) + proj.min(axis=0)))
        assert_allclose(size[lab], expected_size, rtol=1e-12)
        assert_allclose(center[lab], expected_center, atol=1e-10)
        # the box from center, size and eigenvectors encloses every pixel
        # corner, and touches the extremes along each axis
        pixel_corners = (
            region.coords[:, np.newaxis] + corners[np.newaxis]
        ).reshape(-1, ndim) * scale
        local = (pixel_corners - center[lab]) @ v
        assert_allclose(np.abs(local).max(axis=0), 0.5 * size[lab], rtol=1e-12)
    # absent labels
    absent = np.setdiff1d(
        np.arange(1, int(labels_cpu.max()) + 1), np.unique(labels_cpu)
    )
    for lab in absent - 1:
        assert np.all(np.isnan(size[lab])) and np.all(np.isnan(center[lab]))


def test_oriented_bbox_rotated_rectangle():
    # a 51 x 17 rectangle rotated by 20 degrees
    rr, cc = np.mgrid[:80, :80]
    theta = math.radians(20)
    u = (rr - 40) * math.cos(theta) + (cc - 40) * math.sin(theta)
    v = -(rr - 40) * math.sin(theta) + (cc - 40) * math.cos(theta)
    labels = cp.asarray(((np.abs(u) <= 25) & (np.abs(v) <= 8)).astype(np.uint8))
    out = regionprops_dict(
        labels, properties=["oriented_bbox_size", "oriented_bbox_center"]
    )
    # shortest axis first; one pixel wider than the rectangle's pixel
    # centers along each axis, up to the digitization
    assert_allclose(out["oriented_bbox_size"][0], (17, 51), atol=0.5)
    assert_allclose(out["oriented_bbox_center"][0], (40, 40), atol=0.1)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (0.7, 1.3, 1.1)])
def test_oriented_bbox_corners(ndim, spacing):
    labels = _convex_test_labels("random", ndim)
    if spacing is not None:
        spacing = spacing[:ndim]
    names = [
        "oriented_bbox_size",
        "oriented_bbox_center",
        "inertia_tensor_eigenvectors",
    ]
    out = regionprops_dict(labels, properties=names, spacing=spacing)
    corners = measure.oriented_bbox_corners(out)
    assert isinstance(corners, cp.ndarray)
    num_labels = int(labels.max())
    assert corners.shape == (num_labels, 2**ndim, ndim)
    size, center, eigvecs = (cp.asnumpy(out[name]) for name in names)
    present = np.unique(cp.asnumpy(labels))
    present = present[present > 0] - 1
    for lab in present:
        for k, signs in enumerate(itertools.product((-1, 1), repeat=ndim)):
            expected = center[lab] + sum(
                sg * size[lab, a] / 2 * eigvecs[lab][:, a]
                for a, sg in enumerate(signs)
            )
            assert_allclose(corners[lab, k], expected, rtol=1e-12, atol=1e-12)
        # the corners are those of a box: the box frame's extremes
        local = (cp.asnumpy(corners[lab]) - center[lab]) @ eigvecs[lab]
        assert_allclose(
            np.abs(local),
            np.broadcast_to(0.5 * size[lab], local.shape),
            rtol=1e-12,
        )
    # absent labels
    absent = np.setdiff1d(np.arange(num_labels), present)
    assert np.all(np.isnan(cp.asnumpy(corners)[absent]))
    if ndim == 2:
        # the order 0, 1, 3, 2 goes around the rectangle
        loop = cp.asnumpy(corners)[present][:, [0, 1, 3, 2, 0]]
        sides = np.diff(loop, axis=1)
        assert_allclose(
            np.einsum("lsd,lsd->ls", sides[:, :-1], sides[:, 1:]), 0, atol=1e-9
        )


@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("host", [False, True])
def test_oriented_bbox_corners_table(separator, host):
    # the columns of regionprops_table give the same corners
    labels = _convex_test_labels("random", 3)
    names = [
        "label",
        "oriented_bbox_size",
        "oriented_bbox_center",
        "inertia_tensor_eigenvectors",
    ]
    table = measure.regionprops_table(
        labels,
        properties=names,
        separator=separator,
        copy_output_to_host=host,
    )
    assert f"inertia_tensor_eigenvectors{separator}2{separator}1" in table
    corners = measure.oriented_bbox_corners(table, separator=separator)
    assert isinstance(corners, np.ndarray if host else cp.ndarray)
    expected = measure.oriented_bbox_corners(
        regionprops_dict(labels, properties=names[1:])
    )
    assert_allclose(
        cp.asnumpy(corners),
        cp.asnumpy(expected)[cp.asnumpy(table["label"]) - 1],
    )


def test_oriented_bbox_corners_missing_properties():
    labels = _convex_test_labels("blobs", 2)
    out = regionprops_dict(
        labels, properties=["oriented_bbox_size", "oriented_bbox_center"]
    )
    with pytest.raises(
        ValueError, match="missing: inertia_tensor_eigenvectors"
    ):
        measure.oriented_bbox_corners(out)
    with pytest.raises(ValueError, match="regionprops_table"):
        measure.oriented_bbox_corners({})


def test_surface_area_weights():
    # the Crofton weights are the areas of the Voronoi cells of the 26
    # neighbor directions on the unit sphere (a direction and its opposite)
    from scipy.spatial import SphericalVoronoi

    dirs = np.array(
        [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)], float
    )
    sv = SphericalVoronoi(dirs / np.linalg.norm(dirs, axis=1)[:, np.newaxis])
    areas = sv.calculate_areas() / (4 * math.pi)
    n_nonzero = np.count_nonzero(dirs, axis=1)
    for k in (1, 2, 3):
        assert_allclose(
            2 * areas[n_nonzero == k], _CROFTON_3D_WEIGHTS[k - 1], rtol=1e-12
        )


@pytest.mark.parametrize(
    "spacing",
    [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (0.5, 1.5, 0.8), (1.0, 1.0, 5.0)],
)
def test_surface_area_weights_spacing(spacing):
    # with anisotropic spacing, the weights are the Voronoi cells of the
    # spacing-scaled directions: the fraction of 2**20 nearly uniformly
    # spread points on the sphere (a Fibonacci lattice) nearest to a
    # direction or its opposite
    weights = _get_voronoi_crofton_weights(spacing)
    assert_allclose(sum(weights), 1, rtol=1e-12)
    if len(set(spacing)) == 1:
        expected = [_CROFTON_3D_WEIGHTS[0]] * 3 + [_CROFTON_3D_WEIGHTS[1]] * 6
        expected += [_CROFTON_3D_WEIGHTS[2]] * 4
        assert_allclose(sorted(weights), sorted(expected), rtol=1e-12)
    n = 2**20
    i = cp.arange(n, dtype=cp.float64)
    z = 1 - (2 * i + 1) / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = cp.sqrt(1 - z * z)
    points = cp.stack([r * cp.cos(phi), r * cp.sin(phi), z], axis=1)
    # one offset of each pair, in the order of the weights
    offsets = list(itertools.product(range(-1, 2), repeat=3))
    pairs = [offsets[g[0]] for g in _surface_area_groups(3, False)]
    dirs = np.array(pairs, float) * np.asarray(spacing)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    nearest = cp.argmax(cp.abs(points @ cp.asarray(dirs.T)), axis=1)
    counts = cp.bincount(nearest, minlength=len(pairs))
    assert_allclose(weights, cp.asnumpy(counts) / n, atol=1e-4)


@pytest.mark.parametrize("kind", ["random", "large_blobs"])
@pytest.mark.parametrize("spacing", [None, (0.5, 1.5, 0.8)])
@pytest.mark.parametrize("label_dtype", [cp.uint8, cp.int32])
def test_surface_area(kind, spacing, label_dtype):
    labels = _convex_test_labels(kind, 3).astype(label_dtype)
    result = regionprops_dict(
        labels, properties=["surface_area"], spacing=spacing
    )["surface_area"]
    assert_allclose(
        result, _surface_area_reference(labels, spacing), rtol=1e-12
    )
    # an absent label value has no surface
    if kind == "random":
        assert result[2] == 0


def test_surface_area_spacing_types():
    # the coefficients are cached per spacing: any sequence type gives the
    # same values, and so does None for unit spacing (up to the order of the
    # atomic additions)
    labels = _convex_test_labels("random", 3)
    results = [
        regionprops_dict(labels, properties=["surface_area"], spacing=s)[
            "surface_area"
        ]
        for s in [(0.5, 1.5, 0.8), [0.5, 1.5, 0.8], np.array([0.5, 1.5, 0.8])]
    ]
    for result in results[1:]:
        assert_allclose(result, results[0], rtol=1e-14)
    unit = regionprops_dict(labels, properties=["surface_area"], spacing=None)
    assert_allclose(
        unit["surface_area"],
        regionprops_dict(
            labels, properties=["surface_area"], spacing=(1, 1, 1)
        )["surface_area"],
        rtol=1e-14,
    )
    assert not np.allclose(
        cp.asnumpy(results[0]), cp.asnumpy(unit["surface_area"])
    )


def test_surface_area_ball():
    # close to the area of the sphere for a digitized ball
    from skimage.morphology import ball

    radius = 20
    labels = cp.asarray(ball(radius).astype(np.uint8))
    area = regionprops_dict(labels, properties=["surface_area"])["surface_area"]
    assert_allclose(area, 4 * math.pi * radius**2, rtol=0.01)


@pytest.mark.parametrize("spacing", [None, (2.0, 1.0, 1.0), (1.0, 2.0, 2.0)])
def test_surface_area_box(spacing):
    # Axis-aligned boxes: thin plates normal to each axis, and a box with
    # unequal sides (physical sizes).
    scale = np.ones(3) if spacing is None else np.asarray(spacing)
    weights = _crofton_weights_reference(spacing)
    # faces normal to axis i are measured with the factor
    # f_i = 2 * sum_k c_k |u_k[i]| over the 13 unit directions u_k and their
    # weights c_k, or the sum of c_k |u_k[i]| over the 26 offsets
    factors = np.zeros(3)
    for off, c in weights.items():
        d = np.asarray(off) * scale
        factors += c * np.abs(d) / np.linalg.norm(d)
    if spacing is None:
        # 7.3% too small for faces normal to an axis
        assert_allclose(factors, 0.92662, atol=1e-5)
    else:
        # within 16% (ITK's weights, which do not depend on the spacing,
        # give 1.17 and 0.75 for spacing (2, 1, 1), and 0.64 and 1.03 for
        # (1, 2, 2))
        assert np.all(np.abs(factors - 1) < 0.16)
    for size in [(8, 120, 120), (120, 8, 120), (120, 120, 8), (24, 40, 64)]:
        n = (np.asarray(size) / scale).astype(int)
        labels = np.zeros(tuple(n + 4), dtype=np.uint8)
        labels[tuple(slice(2, 2 + k) for k in n)] = 1
        area = regionprops_dict(
            cp.asarray(labels), properties=["surface_area"], spacing=spacing
        )["surface_area"][0]
        # Along each of the 26 directions d to the neighbors, every voxel
        # but (n0 - |d0|) * (n1 - |d1|) * (n2 - |d2|) has its neighbor
        # outside the box.
        expected = 0.0
        for off, c in weights.items():
            d = np.asarray(off)
            coef = 2 * c * np.prod(scale) / np.linalg.norm(d * scale)
            expected += coef * (np.prod(n) - np.prod(n - np.abs(d)))
        assert_allclose(area, expected, rtol=1e-12)
        # the faces, measured with their factors, less up to 2.5% for the
        # edges
        lengths = np.asarray(size, float)
        faces = np.prod(lengths) / lengths
        from_faces = 2 * np.sum(faces * factors)
        assert from_faces * 0.975 < float(area) < from_faces


def test_surface_area_2d_raises():
    with pytest.raises(ValueError, match="3D label images only"):
        regionprops_dict(cp.ones((4, 4), cp.uint8), properties=["surface_area"])


@pytest.mark.parametrize("granularity", [1, 2, 3, 5])
@pytest.mark.parametrize("label_dtype", [cp.uint8, cp.int32])
def test_boundary_properties_touching_labels(granularity, label_dtype):
    # every region is measured as in isolation, also when regions touch
    rng = np.random.default_rng(granularity)
    g = granularity
    small = rng.integers(0, 7, size=(61 // g + 1, 87 // g + 1))
    labels = np.kron(small, np.ones((g, g), dtype=int))[:61, :87]
    labels_gpu = cp.asarray(labels.astype(label_dtype))
    max_label = int(labels.max())
    masks = [labels == lab for lab in range(1, max_label + 1)]
    for neighborhood in [4, 8]:
        values = regionprops_perimeter(
            labels_gpu, neighborhood, max_label=max_label
        )
        expected = [measure_cpu.perimeter(m, neighborhood) for m in masks]
        assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
    for directions in [2, 4]:
        values = regionprops_perimeter_crofton(
            labels_gpu, directions, max_label=max_label
        )
        expected = [measure_cpu.perimeter_crofton(m, directions) for m in masks]
        assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
    for connectivity in [1, 2]:
        values = regionprops_euler(
            labels_gpu, connectivity, max_label=max_label
        )
        expected = [measure_cpu.euler_number(m, connectivity) for m in masks]
        assert_array_equal(values, expected)

    small = rng.integers(0, 5, size=(9, 8, 10))
    labels = np.kron(small, np.ones((g, g, g), dtype=int))[:17, :13, :19]
    labels_gpu = cp.asarray(labels.astype(label_dtype))
    max_label = int(labels.max())
    for connectivity in [1, 3]:
        values = regionprops_euler(
            labels_gpu, connectivity, max_label=max_label
        )
        expected = [
            measure_cpu.euler_number(labels == lab, connectivity)
            for lab in range(1, max_label + 1)
        ]
        assert_array_equal(values, expected)


@pytest.mark.parametrize("int32_coords", [True, False])
@pytest.mark.parametrize(
    "kind, ndim", [("perimeter", 2), ("crofton", 2), ("euler", 2), ("euler", 3)]
)
def test_boundary_kernel_coordinate_types(kind, ndim, int32_coords):
    # images of fewer than 2**31 pixels use 32-bit coordinates, larger ones
    # 64-bit coordinates; both must give scikit-image's values
    rng = np.random.default_rng(ndim)
    shape = (61, 87) if ndim == 2 else (17, 13, 19)
    small = rng.integers(0, 5, size=tuple(s // 2 + 1 for s in shape))
    labels = np.kron(small, np.ones((2,) * ndim, dtype=int))
    labels = labels[tuple(slice(0, s) for s in shape)]
    max_label = int(labels.max())
    masks = [labels == lab for lab in range(1, max_label + 1)]
    if kind == "perimeter":
        coefs, option = _get_perimeter_coefs(), 4
        expected = [measure_cpu.perimeter(m, 4) for m in masks]
    elif kind == "crofton":
        coefs, option = _get_perimeter_crofton_coefs(4), False
        expected = [measure_cpu.perimeter_crofton(m, 4) for m in masks]
    else:
        coefs, option = _get_euler_coefs(ndim, 1), False
        expected = [measure_cpu.euler_number(m, 1) for m in masks]
    labels = cp.asarray(labels, dtype=cp.int32)
    kernel = _get_boundary_kernel(kind, ndim, option, 10, int32_coords)
    out = cp.zeros((max_label,), dtype=coefs.dtype)
    kernel(labels, labels.size, coefs, out, size=math.ceil(labels.size / 10))
    if kind == "euler" and ndim == 3:
        out = (out * 0.125).astype(cp.int64)
    assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


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


_FERET_CALIPER_PROPS = [
    "feret_diameter_max",
    "feret_diameter_min",
    "feret_diameter_min_perpendicular",
    "feret_angle_max",
    "feret_angle_min",
    "perimeter_convex",
]


def _fold_angle(a):
    # an undirected angle in (-pi/2, pi/2]
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return a + math.pi if a <= -math.pi / 2 + 1e-12 else a


def _feret_calipers_reference(image, spacing):
    """Brute-force Feret values of the hull of the convex image's diamond
    points. Ties: the sets of the angles (and perpendicular widths) of all
    maximal pairs and minimal widths."""
    from scipy.spatial import ConvexHull
    from skimage.morphology import convex_hull_image

    spacing = np.ones(2) if spacing is None else np.asarray(spacing, float)
    centers = np.argwhere(convex_hull_image(image))
    diamond = np.array([[-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]])
    points = (centers[:, np.newaxis] + diamond).reshape(-1, 2)
    points = np.unique(points, axis=0) * spacing
    v = points[ConvexHull(points).vertices]
    dist = np.linalg.norm(v[:, np.newaxis] - v[np.newaxis], axis=-1)
    dmax = dist.max()
    angles_max = {
        round(_fold_angle(math.atan2(*(v[j] - v[i])[::-1])), 8)
        for i, j in zip(*np.nonzero(dist >= dmax * (1 - 1e-12)))
    }
    widths, angles, perps = [], [], []
    for k in range(len(v)):
        edge = v[(k + 1) % len(v)] - v[k]
        edge /= np.linalg.norm(edge)
        normal = np.array([-edge[1], edge[0]])
        widths.append(np.abs((v - v[k]) @ normal).max())
        angles.append(round(_fold_angle(math.atan2(normal[1], normal[0])), 8))
        t = v @ edge
        perps.append(t.max() - t.min())
    widths = np.array(widths)
    ties = np.nonzero(widths <= widths.min() * (1 + 1e-12))[0]
    # (the "area" of a 2D ConvexHull is its perimeter)
    return (
        dmax,
        widths.min(),
        {perps[k] for k in ties},
        angles_max,
        {angles[k] for k in ties},
        ConvexHull(points).area,
    )


@pytest.mark.parametrize("kind", ["blobs", "lines", "random"])
@pytest.mark.parametrize("spacing", [None, (0.7, 1.3)])
def test_feret_calipers_2d(kind, spacing):
    # minimum Feret diameter, the width perpendicular to it and the angles
    # of the maximum and minimum diameters, on the hull of the convex image
    # (the polygon of feret_diameter_max)
    labels = _convex_test_labels(kind, 2)
    result = regionprops_dict(
        labels, properties=_FERET_CALIPER_PROPS, spacing=spacing
    )
    result = {k: cp.asnumpy(v) for k, v in result.items()}
    labels_cpu = cp.asnumpy(labels)
    for region in measure_cpu.regionprops(labels_cpu):
        lab = region.label - 1
        dmax, wmin, perps, angles_max, angles_min, perimeter = (
            _feret_calipers_reference(region.image, spacing)
        )
        assert_allclose(result["feret_diameter_max"][lab], dmax, rtol=1e-10)
        assert_allclose(result["feret_diameter_min"][lab], wmin, rtol=1e-10)
        assert any(
            abs(result["feret_diameter_min_perpendicular"][lab] - p)
            <= 1e-10 * p
            for p in perps
        )
        assert round(result["feret_angle_max"][lab], 8) in angles_max
        assert round(result["feret_angle_min"][lab], 8) in angles_min
        assert_allclose(result["perimeter_convex"][lab], perimeter, rtol=1e-10)
    # absent labels: 0 diameters and NaN angles
    absent = np.setdiff1d(
        np.arange(1, int(labels_cpu.max()) + 1), np.unique(labels_cpu)
    )
    for lab in absent - 1:
        assert result["feret_diameter_min"][lab] == 0
        assert result["perimeter_convex"][lab] == 0
        assert np.isnan(result["feret_angle_max"][lab])


def _min_area_bbox_reference(region, spacing):
    """The minimum-area rectangles (size, center, angle) of the hull of the
    convex image's diamond points: one for each hull edge with the smallest
    area (the ties)."""
    from scipy.spatial import ConvexHull
    from skimage.morphology import convex_hull_image

    spacing = np.ones(2) if spacing is None else np.asarray(spacing, float)
    centers = np.argwhere(convex_hull_image(region.image))
    diamond = np.array([[-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]])
    points = (centers[:, np.newaxis] + diamond).reshape(-1, 2)
    points = np.unique(points, axis=0) * spacing
    v = points[ConvexHull(points).vertices]
    found = []
    for k in range(len(v)):
        edge = v[(k + 1) % len(v)] - v[k]
        edge /= np.linalg.norm(edge)
        normal = np.array([-edge[1], edge[0]])
        t, u = v @ edge, v @ normal
        extent, width = np.ptp(t), np.ptp(u)
        center = edge * (t.max() + t.min()) / 2
        center = center + normal * (u.max() + u.min()) / 2
        center = center + np.asarray(region.bbox[:2]) * spacing
        long_axis = edge if extent >= width else normal
        found.append(
            (
                extent * width,
                (max(extent, width), min(extent, width)),
                center,
                _fold_angle(math.atan2(long_axis[1], long_axis[0])),
            )
        )
    smallest = min(f[0] for f in found)
    return [f for f in found if f[0] <= smallest * (1 + 1e-12)]


@pytest.mark.parametrize("kind", ["blobs", "lines", "random"])
@pytest.mark.parametrize("spacing", [None, (0.7, 1.3)])
def test_min_area_bbox_2d(kind, spacing):
    labels = _convex_test_labels(kind, 2)
    props = [
        "min_area_bbox_size",
        "min_area_bbox_center",
        "min_area_bbox_angle",
    ]
    result = regionprops_dict(labels, properties=props, spacing=spacing)
    result = {k: cp.asnumpy(v) for k, v in result.items()}
    labels_cpu = cp.asnumpy(labels)
    for region in measure_cpu.regionprops(labels_cpu):
        lab = region.label - 1
        size = result["min_area_bbox_size"][lab]
        center = result["min_area_bbox_center"][lab]
        angle = result["min_area_bbox_angle"][lab]
        reference = _min_area_bbox_reference(region, spacing)
        assert any(
            np.allclose(size, f[1], rtol=1e-10)
            and np.allclose(center, f[2], rtol=1e-10, atol=1e-9)
            and abs(_fold_angle(angle - f[3])) < 1e-8
            or (  # a square: either side is the length
                np.isclose(size[0], size[1], rtol=1e-10)
                and np.allclose(size, f[1], rtol=1e-10)
                and np.allclose(center, f[2], rtol=1e-10, atol=1e-9)
            )
            for f in reference
        ), (size, center, angle, reference)
    # absent labels
    absent = np.setdiff1d(
        np.arange(1, int(labels_cpu.max()) + 1), np.unique(labels_cpu)
    )
    for lab in absent - 1:
        assert np.all(np.isnan(result["min_area_bbox_size"][lab]))
        assert np.all(np.isnan(result["min_area_bbox_center"][lab]))
        assert np.isnan(result["min_area_bbox_angle"][lab])


def test_min_area_bbox_rotated_square():
    # for a square rotated by 45 degrees the oriented box (from the equal
    # principal moments) is the axis-aligned one, with twice the area; the
    # minimum-area box has the sides of the square
    rr, cc = np.mgrid[:120, :120]
    labels = cp.asarray(
        ((np.abs(rr - 60) + np.abs(cc - 60)) <= 40).astype(np.uint8)
    )
    out = regionprops_dict(
        labels, properties=["min_area_bbox_size", "min_area_bbox_center"]
    )
    size = cp.asnumpy(out["min_area_bbox_size"][0])
    assert_allclose(size, [size[0], size[0]], rtol=1e-12)
    assert abs(size[0] - 40 * math.sqrt(2)) < 1.5
    assert_allclose(cp.asnumpy(out["min_area_bbox_center"][0]), (60, 60))


@pytest.mark.parametrize("kind", ["blobs", "lines", "random"])
@pytest.mark.parametrize("spacing", [None, (0.7, 1.3)])
def test_min_area_bbox_corners(kind, spacing):
    labels = _convex_test_labels(kind, 2)
    names = [
        "min_area_bbox_size",
        "min_area_bbox_center",
        "min_area_bbox_angle",
    ]
    out = regionprops_dict(labels, properties=names, spacing=spacing)
    corners = cp.asnumpy(measure.min_area_bbox_corners(out))
    size, center, angle = (cp.asnumpy(out[name]) for name in names)
    labels_cpu = cp.asnumpy(labels)
    assert corners.shape == (int(labels_cpu.max()), 4, 2)
    for region in measure_cpu.regionprops(labels_cpu):
        lab = region.label - 1
        c = corners[lab]
        # a rectangle (in the order 0, 1, 3, 2) of the size, around the center
        loop = c[[0, 1, 3, 2, 0]]
        sides = np.diff(loop, axis=0)
        assert_allclose(np.sum(sides[:-1] * sides[1:], axis=1), 0, atol=1e-9)
        assert_allclose(
            sorted(np.linalg.norm(sides, axis=1)[:2]),
            sorted(size[lab]),
            rtol=1e-10,
        )
        assert_allclose(c.mean(axis=0), center[lab], atol=1e-9)
        # the first axis is the direction of the length side
        axis = c[2] - c[0]
        assert_allclose(
            axis / np.linalg.norm(axis),
            (math.cos(angle[lab]), math.sin(angle[lab])),
            atol=1e-9,
        )
        # the corners enclose the polygon of the Feret diameters
        scale = np.ones(2) if spacing is None else np.asarray(spacing)
        centers = np.argwhere(convex_hull_image(region.image))
        diamond = np.array([[-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]])
        points = (centers[:, np.newaxis] + diamond).reshape(-1, 2)
        points = (points + region.bbox[:2]) * scale
        rel = points - c[0]
        u = rel @ (c[2] - c[0]) / np.sum((c[2] - c[0]) ** 2)
        v = rel @ (c[1] - c[0]) / np.sum((c[1] - c[0]) ** 2)
        assert min(u.min(), v.min()) > -1e-9
        assert max(u.max(), v.max()) < 1 + 1e-9
    absent = np.setdiff1d(
        np.arange(1, int(labels_cpu.max()) + 1), np.unique(labels_cpu)
    )
    assert np.all(np.isnan(corners[absent - 1]))


def test_min_area_bbox_corners_table_and_missing():
    labels = _convex_test_labels("blobs", 2)
    names = [
        "min_area_bbox_size",
        "min_area_bbox_center",
        "min_area_bbox_angle",
    ]
    table = measure.regionprops_table(
        labels,
        properties=["label", *names],
        separator="_",
        copy_output_to_host=True,
    )
    corners = measure.min_area_bbox_corners(table, separator="_")
    assert isinstance(corners, np.ndarray)
    expected = measure.min_area_bbox_corners(
        regionprops_dict(labels, properties=names)
    )
    assert_allclose(corners, cp.asnumpy(expected)[table["label"] - 1])
    out = regionprops_dict(labels, properties=names[:2])
    with pytest.raises(ValueError, match="missing: min_area_bbox_angle"):
        measure.min_area_bbox_corners(out)


def test_feret_calipers_3d_raises():
    with pytest.raises(ValueError, match="2D label images only"):
        regionprops_dict(
            cp.ones((4, 4, 4), cp.uint8), properties=["feret_diameter_min"]
        )


@pytest.mark.parametrize("spacing", [None, (0.5, 0.35)])
@pytest.mark.parametrize("kind", ["blobs", "large_blobs", "lines", "random"])
def test_feret_diameter_max_batch_2d(spacing, kind):
    # equal to scikit-image's (contour points of the convex image), also for
    # single pixels (diameter 1)
    labels = _convex_test_labels(kind, 2)
    result = regionprops_dict(
        labels, properties=["feret_diameter_max"], spacing=spacing
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        properties=["label", "feret_diameter_max"],
        spacing=spacing,
    )
    assert_allclose(
        result["feret_diameter_max"][expected["label"] - 1],
        expected["feret_diameter_max"],
        rtol=1e-12,
    )


@pytest.mark.parametrize("spacing", [None, (0.5, 0.35, 0.75)])
@pytest.mark.parametrize("kind", ["blobs", "large_blobs", "lines", "random"])
def test_feret_diameter_max_batch_3d(spacing, kind):
    labels = _convex_test_labels(kind, 3)
    result = regionprops_dict(
        labels, properties=["feret_diameter_max"], spacing=spacing
    )["feret_diameter_max"]
    n_compared = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for region in measure_cpu.regionprops(
            cp.asnumpy(labels), spacing=spacing
        ):
            if not region.image_convex.any():
                # scikit-image's qhull fails for flat regions, and its
                # feret_diameter_max then raises (test_convex_flat_3d)
                continue
            assert_allclose(
                result[region.label - 1],
                region.feret_diameter_max,
                rtol=1e-12,
            )
            n_compared += 1
    assert n_compared > 0


def test_feret_diameter_max_batch_3d_thread_cap(monkeypatch):
    # with many labels, the number of threads per label is capped (down to
    # one warp); the result does not depend on it
    labels = _convex_test_labels("large_blobs", 3)
    expected = regionprops_dict(labels, properties=["feret_diameter_max"])
    monkeypatch.setattr(_regionprops_gpu_convex, "_FERET_MAX_THREADS", 1)
    result = regionprops_dict(labels, properties=["feret_diameter_max"])
    assert_array_equal(
        result["feret_diameter_max"], expected["feret_diameter_max"]
    )


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
    if ndim != 3 and property_name in ndim_3_only:
        pytest.skip(f"{property_name} is for 3d images only.")
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
    if ndim != 3 and property_name in ndim_3_only:
        pytest.skip(f"{property_name} is for 3d images only.")
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


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("max_canvas_pixels", [None, 4000, 1])
def test_fill_holes_of_labels_batch(ndim, max_canvas_pixels):
    # the crops packed into one canvas are filled as one at a time, also when
    # the canvas is split in several batches (1: every crop on its own)
    from cucim.skimage.measure._regionprops_gpu_basic_kernels import (
        _fill_holes_of_labels,
    )

    rng = np.random.default_rng(3)
    shape = (120, 150) if ndim == 2 else (30, 40, 36)
    # nested and touching regions: labeled components of noise levels
    noise = ndi.gaussian_filter(cp.asarray(rng.random(shape)), 2)
    levels = cp.digitize(noise, cp.percentile(noise, [20, 50, 80]))
    labels = measure.label(levels + 1, connectivity=ndim)
    max_label = int(labels.max())
    bbox, _ = regionprops_bbox_coords(labels, max_label=max_label)
    bbox_host = cp.asnumpy(bbox)
    labels_list = np.arange(1, max_label + 1)
    counts, images = _fill_holes_of_labels(
        labels,
        labels_list,
        bbox_host,
        compute_image=True,
        max_canvas_pixels=max_canvas_pixels,
    )
    structure = cp.ones((3,) * ndim, dtype=bool)
    for lab, count, image in zip(labels_list, cp.asnumpy(counts), images):
        sl = tuple(
            slice(int(bbox_host[lab - 1, d]), int(bbox_host[lab - 1, d + ndim]))
            for d in range(ndim)
        )
        expected = ndi.binary_fill_holes(labels[sl] == lab, structure)
        assert_array_equal(image, expected)
        assert count == int(expected.sum())


def test_pack_crops_canvas_is_not_wider_than_the_boxes():
    from cucim.skimage.measure._regionprops_gpu_basic_kernels import (
        _pack_crops,
    )

    # one tall box: its own width, not that of a square canvas
    shape, starts = _pack_crops([[1000, 5]])
    assert shape == (1002, 7)
    assert tuple(starts[0]) == (1, 1)
    # several boxes: about square, and every box inside with a margin
    extents = np.array([[40, 30], [10, 70], [25, 25], [5, 5]])
    shape, starts = _pack_crops(extents)
    assert shape[1] < sum(extents[:, 1]) + len(extents) + 1
    canvas = np.zeros(shape, dtype=int)
    for i, (st, ext) in enumerate(zip(starts, extents), 1):
        box = canvas[
            st[0] - 1 : st[0] + ext[0] + 1, st[1] - 1 : st[1] + ext[1] + 1
        ]
        assert not box.any()
        canvas[st[0] : st[0] + ext[0], st[1] : st[1] + ext[1]] = i


@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16, cp.int64])
def test_fill_holes_by_label_dtypes(dtype):
    labels = cp.zeros((20, 20), dtype=dtype)
    labels[2:12, 2:12] = 1
    labels[4:10, 4:10] = 0
    labels[8:12, 14:18] = 2
    labels[9, 15:17] = 0
    label_filled, labels_inexact = _fill_holes_by_label(labels, 2)
    assert label_filled.dtype == dtype
    expected = labels.copy()
    expected[4:10, 4:10] = 1
    expected[9, 15:17] = 2
    assert_array_equal(label_filled, expected)
    assert labels_inexact.size == 0


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


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (1.5, 0.5, 0.75)])
def test_regionprops_dict_centroid_paths(ndim, spacing):
    # The centroids come from coordinate sums when no other property needs
    # the raw moments, and from the moments otherwise. Both must agree.
    shape = (256, 512) if ndim == 2 else (48, 64, 64)
    if spacing is not None:
        spacing = spacing[:ndim]
    labels = get_labels_nd(shape)
    labels[labels == 3] = 0  # an absent label gives NaN centroids
    props = ["centroid", "centroid_local"]
    from_sums = regionprops_dict(labels, properties=props, spacing=spacing)
    from_moments = regionprops_dict(
        labels, properties=props + ["moments_central"], spacing=spacing
    )
    for prop in props:
        assert from_sums[prop].dtype == from_moments[prop].dtype
        assert_allclose(from_sums[prop], from_moments[prop], rtol=1e-12)
        assert cp.all(cp.isnan(from_sums[prop][2]))


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
