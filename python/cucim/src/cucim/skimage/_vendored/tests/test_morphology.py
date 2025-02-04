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
