# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that binary dispatch preserves grayscale morphology semantics."""

import cupy as cp
import numpy as np
import pytest
from skimage import morphology as morphology_cpu

from cucim.skimage import morphology
from cucim.skimage.morphology import gray

FUNCTIONS = [
    "erosion",
    "dilation",
    "opening",
    "closing",
    "white_tophat",
    "black_tophat",
]
MODES = [(mode, 0) for mode in sorted(gray._SUPPORTED_MODES)] + [
    ("constant", 1)
]


def _footprints(kind, ndim):
    if kind == "default":
        return None, None
    if kind == "tuple":
        shape = (3,) * ndim
        return shape, np.ones(shape, dtype=bool)
    if kind == "sequence":
        shape = (3,) + (1,) * (ndim - 1)
        fp = np.array([0, 1, 1], dtype=bool).reshape((1,) * (ndim - 1) + (3,))
        return [(shape, 2), (cp.asarray(fp), 1)], [
            (np.ones(shape, bool), 2),
            (fp, 1),
        ]
    if kind == "even":
        shape = (2,) + (3,) * (ndim - 1)
        fp = np.ones(shape, dtype=bool)
    else:
        fp = np.ones((3,) * ndim, dtype=bool)
        if kind == "asymmetric":
            fp[(0,) * ndim] = False
            fp[(1,) * ndim] = False
            fp[(2,) + (1,) * (ndim - 1)] = False
    return cp.asarray(fp), fp


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("mode,cval", MODES)
@pytest.mark.parametrize(
    "kind", ["default", "tuple", "odd", "even", "asymmetric", "sequence"]
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_boolean_dispatch_matches_cpu(function, mode, cval, kind, layout):
    image = np.random.default_rng(0).random((9, 11)) > 0.5
    if layout == "F":
        image = np.asfortranarray(image)
    image_gpu = cp.asarray(image)
    if layout == "strided":
        image = image[::-2, ::2]
        image_gpu = image_gpu[::-2, ::2]
    footprint_gpu, footprint_cpu = _footprints(kind, image.ndim)
    expected = getattr(morphology_cpu, function)(
        image, footprint_cpu, mode=mode, cval=cval
    )
    actual = getattr(morphology, function)(
        image_gpu, footprint_gpu, mode=mode, cval=cval
    )
    assert actual.dtype == bool
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("mode,cval", MODES)
@pytest.mark.parametrize("kind", ["even", "asymmetric", "sequence"])
def test_boolean_dispatch_3d(function, mode, cval, kind):
    image = np.random.default_rng(1).random((4, 5, 6)) > 0.5
    footprint_gpu, footprint_cpu = _footprints(kind, image.ndim)
    expected = getattr(morphology_cpu, function)(
        image, footprint_cpu, mode=mode, cval=cval
    )
    actual = getattr(morphology, function)(
        cp.asarray(image), footprint_gpu, mode=mode, cval=cval
    )
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "output_kind", ["bool", "int32", "float32", "in_place", "reversed"]
)
def test_boolean_dispatch_output(function, output_kind):
    image = np.random.default_rng(2).random((9, 11)) > 0.5
    image_gpu = cp.asarray(image)
    footprint_gpu, footprint_cpu = _footprints("asymmetric", 2)
    if output_kind == "in_place":
        output_cpu, output_gpu = image, image_gpu
    elif output_kind == "reversed":
        output_cpu = np.empty_like(image)[:, ::-1]
        output_gpu = cp.empty_like(image_gpu)[:, ::-1]
    else:
        output_cpu = np.empty(image.T.shape, dtype=output_kind).T
        output_gpu = cp.empty(image.T.shape, dtype=output_kind).T
    expected = getattr(morphology_cpu, function)(
        image, footprint_cpu, out=output_cpu, mode="constant"
    )
    actual = getattr(morphology, function)(
        image_gpu, footprint_gpu, out=output_gpu, mode="constant"
    )
    assert actual is output_gpu
    assert actual.dtype == output_cpu.dtype
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("function", ["erosion", "dilation"])
def test_boolean_dispatch_uses_binary_kernel(function, monkeypatch):
    image = cp.ones((5, 7), dtype=bool)

    def unexpected_grayscale_kernel(*args, **kwargs):
        pytest.fail("boolean input should use the private binary kernel")

    monkeypatch.setattr(
        gray.ndi, f"grey_{function}", unexpected_grayscale_kernel
    )
    getattr(morphology, function)(image, (3, 3))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("function", FUNCTIONS)
def test_non_boolean_input_uses_grayscale_kernel(function, dtype, monkeypatch):
    image = cp.ones((5, 7), dtype=dtype)

    def unexpected_binary_kernel(*args, **kwargs):
        pytest.fail("non-boolean input should use the grayscale kernel")

    monkeypatch.setattr(gray, "_binary_gray", unexpected_binary_kernel)
    getattr(morphology, function)(image, (3, 3))


@pytest.mark.parametrize("function", ["erosion", "dilation"])
@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_boolean_dispatch_rejects_empty_footprint(function, mode):
    with pytest.raises(ValueError, match="all-zero footprint"):
        getattr(morphology, function)(
            cp.ones((5, 7), dtype=bool), cp.zeros((3, 3), dtype=bool), mode=mode
        )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("cval", [-1, 0.5, 2])
def test_non_boolean_border_value_uses_grayscale_kernel(
    function, cval, monkeypatch
):
    def unexpected_binary_kernel(*args, **kwargs):
        pytest.fail("non-boolean border values should use the grayscale kernel")

    monkeypatch.setattr(gray, "_binary_gray", unexpected_binary_kernel)
    getattr(morphology, function)(
        cp.ones((5, 7), dtype=bool), (3, 3), mode="constant", cval=cval
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "mode,cval", [("constant", 0), ("constant", 1), ("reflect", 0)]
)
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 7)])
def test_boolean_dispatch_broadcast_input(function, mode, cval, shape):
    # Use a materialized reference; SciPy morphology does not reliably handle
    # zero-stride NumPy inputs.
    image = np.full(shape, True, dtype=bool)
    image_gpu = cp.broadcast_to(cp.asarray(True), shape)
    expected = getattr(morphology_cpu, function)(image, mode=mode, cval=cval)
    actual = getattr(morphology, function)(image_gpu, mode=mode, cval=cval)
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "function", ["erosion", "dilation", "opening", "closing"]
)
@pytest.mark.parametrize("as_tuple", [False, True])
@pytest.mark.parametrize("mode", ["reflect", "ignore"])
@pytest.mark.parametrize(
    "shape, fp_shape", [((16, 16, 16), (5, 5, 5)), ((40, 40), (17, 17))]
)
def test_large_dense_footprint_keeps_separable_path(
    function, as_tuple, mode, shape, fp_shape, monkeypatch
):
    def unexpected_binary_kernel(*args, **kwargs):
        pytest.fail("large dense footprints should retain separable filtering")

    monkeypatch.setattr(gray, "_binary_gray", unexpected_binary_kernel)
    image_cpu = np.random.default_rng(4).random(shape) > 0.5
    image = cp.asarray(image_cpu)
    footprint = fp_shape if as_tuple else cp.ones(fp_shape, dtype=bool)
    expected = getattr(morphology_cpu, function)(
        image_cpu, np.ones(fp_shape, dtype=bool), mode=mode
    )
    actual = getattr(morphology, function)(image, footprint, mode=mode)
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fp_shape", [(15, 15), (3, 5, 5), (3, 31)])
def test_moderate_dense_footprint_uses_binary_kernel(fp_shape, monkeypatch):
    def unexpected_grayscale_kernel(*args, **kwargs):
        pytest.fail("moderate dense footprints should use the binary kernel")

    monkeypatch.setattr(gray.ndi, "grey_erosion", unexpected_grayscale_kernel)
    image = cp.ones((8,) * len(fp_shape), dtype=bool)
    morphology.erosion(image, fp_shape)
    morphology.erosion(image, cp.ones(fp_shape, dtype=bool))
