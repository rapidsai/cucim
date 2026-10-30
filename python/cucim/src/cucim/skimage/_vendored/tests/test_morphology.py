# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing axes support that is not currently present in CuPy/SciPy."""

import cupy as cp
import numpy as np
import pytest
import scipy.ndimage as ndi_cpu

from cucim.skimage._vendored import ndimage as ndi


@pytest.mark.parametrize("border_value", [0, 1])
@pytest.mark.parametrize("origin", [(0, 0), (-1, 0)])
@pytest.mark.parametrize("expand_axis", [0, 1, 2])
@pytest.mark.parametrize(
    "func",
    [
        "binary_erosion",
        "binary_dilation",
        "binary_opening",
        "binary_closing",
        "binary_hit_or_miss",
        "binary_propagation",
        "binary_fill_holes",
    ],
)
def test_binary_axes(func, expand_axis, origin, border_value):
    struct = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    data = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        bool,
    )
    if func == "binary_hit_or_miss":
        kwargs = dict(origin1=origin, origin2=origin)
    else:
        kwargs = dict(origin=origin)
    border_supported = func not in ["binary_hit_or_miss", "binary_fill_holes"]
    if border_supported:
        kwargs["border_value"] = border_value
    elif border_value != 0:
        pytest.skip("border_value !=0 unsupported by this function")

    scipy_func = getattr(ndi_cpu, func)
    expected = scipy_func(data, struct, **kwargs)

    # copy data and expected results to GPU
    data = cp.asarray(data)
    expected = cp.asarray(expected)
    struct = cp.asarray(struct)

    # replicate data and expected result along a new axis
    n_reps = 5
    expected = cp.stack([expected] * n_reps, axis=expand_axis)
    data = cp.stack([data] * n_reps, axis=expand_axis)

    # filter all axes except expand_axis
    axes = [0, 1, 2]
    axes.remove(expand_axis)
    out = cp.zeros(data.shape, bool)
    cucim_func = getattr(ndi, func)
    cucim_func(data, struct, output=out, axes=axes, **kwargs)
    cp.testing.assert_array_almost_equal(out, expected)


@pytest.mark.parametrize("origin", [(0, 0), (-1, 0)])
@pytest.mark.parametrize("expand_axis", [0, 1, 2])
@pytest.mark.parametrize("mode", ["reflect", "constant"])
@pytest.mark.parametrize("footprint_mode", ["size", "footprint", "structure"])
@pytest.mark.parametrize(
    "func",
    [
        "grey_erosion",
        "grey_dilation",
        "grey_opening",
        "grey_closing",
        "morphological_laplace",
        "morphological_gradient",
        "white_tophat",
        "black_tophat",
    ],
)
def test_grey_axes(func, expand_axis, origin, footprint_mode, mode):
    data = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0],
            [0, 0, 2, 1, 0, 2, 0],
            [0, 3, 0, 6, 5, 0, 1],
            [0, 4, 5, 3, 3, 4, 0],
            [0, 0, 9, 3, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
        ]
    )
    kwargs = dict(origin=origin, mode=mode)
    if footprint_mode == "size":
        kwargs["size"] = (2, 3)
    else:
        kwargs["footprint"] = np.asarray([[1, 0, 1], [1, 1, 0]])
    if footprint_mode == "structure":
        kwargs["structure"] = np.ones_like(kwargs["footprint"])
    scipy_func = getattr(ndi_cpu, func)
    expected = scipy_func(data, **kwargs)

    # copy data and expected results to GPU
    data = cp.asarray(data)
    expected = cp.asarray(expected)
    if "footprint" in kwargs:
        kwargs["footprint"] = cp.asarray(kwargs["footprint"])
    if "structure" in kwargs:
        kwargs["structure"] = cp.asarray(kwargs["structure"])

    # replicate data and expected result along a new axis
    n_reps = 5
    expected = cp.stack([expected] * n_reps, axis=expand_axis)
    data = cp.stack([data] * n_reps, axis=expand_axis)

    # filter all axes except expand_axis
    axes = [0, 1, 2]
    axes.remove(expand_axis)
    out = cp.zeros(expected.shape, dtype=expected.dtype)
    cucim_func = getattr(ndi, func)
    cucim_func(data, output=out, axes=axes, **kwargs)
    cp.testing.assert_array_almost_equal(out, expected)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("connectivity", [1, 2, 3])
def test_binary_fill_holes_structure(ndim, connectivity):
    # the non-iterative algorithm (used with the default origin and axes)
    # must also use the structuring element
    if connectivity > ndim:
        pytest.skip("connectivity > ndim")
    rng = np.random.default_rng(5)
    shape = (64, 64) if ndim == 2 else (16, 16, 16)
    data = rng.random(shape) > 0.35
    struct = ndi_cpu.generate_binary_structure(ndim, connectivity)
    expected = ndi_cpu.binary_fill_holes(data, struct)
    # the structuring element changes which background pixels are holes
    default = ndi_cpu.binary_fill_holes(data)
    assert np.array_equal(expected, default) == (connectivity == 1)
    out = ndi.binary_fill_holes(cp.asarray(data), cp.asarray(struct))
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "footprint", ["horizontal", "large", "asymmetric", "diagonal"]
)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_binary_fill_holes_iterative_structure(ndim, footprint, empty, inplace):
    if footprint == "horizontal":
        struct = np.zeros((3,) * ndim, dtype=bool)
        struct[(1,) * (ndim - 1) + (slice(None),)] = True
    elif footprint == "large":
        struct = np.ones((5,) * ndim, dtype=bool)
    elif footprint == "asymmetric":
        struct = np.ones((3,) * ndim, dtype=bool)
        struct[(0,) * ndim] = False
    else:
        struct = np.zeros((3,) * ndim, dtype=bool)
        for index in range(3):
            struct[(index,) * ndim] = True

    data = np.zeros((9,) * ndim, dtype=bool)
    if not empty:
        data[(slice(1, -1),) * ndim] = True
        data[(slice(3, -3),) * ndim] = False
    expected = ndi_cpu.binary_fill_holes(data, struct)
    data = cp.asarray(data)
    struct = cp.asarray(struct)
    if inplace:
        ndi.binary_fill_holes(data, struct, output=data)
        out = data
    else:
        out = ndi.binary_fill_holes(data, struct)
    cp.testing.assert_array_equal(out, expected)


def test_binary_fill_holes_1d():
    # The exterior on each side of the image must remain unfilled even
    # though the padding cannot connect the two ends in one dimension.
    data = np.array([0, 1, 0, 1, 0], dtype=bool)
    expected = ndi_cpu.binary_fill_holes(data)
    out = ndi.binary_fill_holes(cp.asarray(data))
    cp.testing.assert_array_equal(out, expected)
