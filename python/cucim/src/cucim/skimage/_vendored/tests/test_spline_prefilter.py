# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from scipy import ndimage as scipy_ndi

from cucim.skimage._vendored import _ndimage_interpolation as interpolation


@pytest.mark.parametrize("length", [2, 3, 7, 17])
@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64"])
def test_fused_gain_short_reflect_lines(length, order, dtype):
    rng = np.random.default_rng(6)
    image = rng.uniform(0, 1, (4, length))
    if dtype == "complex64":
        image = image + 1j * rng.uniform(0, 1, image.shape)
    image = image.astype(dtype)
    expected = scipy_ndi.spline_filter1d(
        image, order, axis=-1, output=dtype, mode="reflect"
    )
    actual = interpolation.spline_filter1d(
        cp.asarray(image),
        order,
        axis=-1,
        output=cp.dtype(dtype),
        mode="reflect",
    )
    tolerance = 1e-11 if dtype == "float64" else 5e-5
    cp.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["mirror", "reflect", "grid-wrap"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    "dtype,output_kind",
    [
        ("float32", "dtype"),
        ("float64", "array"),
        ("uint8", "dtype"),
        ("complex64", "inplace"),
        ("float32", "strided"),
    ],
)
def test_spline_filter_workspace(order, mode, transpose, dtype, output_kind):
    rng = np.random.default_rng(5)
    host = rng.uniform(0, 1, (7, 9))
    if dtype == "uint8":
        host *= 255
    elif dtype == "complex64":
        host = host + 1j * rng.uniform(0, 1, host.shape)
    image = cp.asarray(host.astype(dtype))
    if transpose:
        image = image.T
    original = image.get()
    working_dtype = np.result_type(dtype, np.float32)
    expected = scipy_ndi.spline_filter(
        original, order=order, output=working_dtype, mode=mode
    )
    if output_kind == "array":
        output = cp.empty_like(image, order="C")
    elif output_kind == "inplace":
        output = image
    elif output_kind == "strided":
        output = cp.empty(
            (image.shape[0], image.shape[1] * 2), dtype=working_dtype
        )[:, ::2]
    else:
        output = cp.dtype(working_dtype)
    actual = interpolation.spline_filter(image, order, output=output, mode=mode)
    tolerance = 1e-11 if working_dtype == np.dtype("float64") else 5e-5
    cp.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == working_dtype
    if isinstance(output, cp.ndarray):
        assert actual is output
    if output is not image:
        cp.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize(
    "mode", ["mirror", "reflect", "grid-wrap", "nearest", "grid-constant"]
)
def test_batch_prefilter_workspace(order, mode):
    image = (
        np.random.default_rng(4).uniform(0, 1, (2, 7, 9, 3)).astype(np.float32)
    )
    image_gpu = cp.asarray(image)
    padding = 12 if mode in ("nearest", "grid-constant") else 0
    if padding:
        kwargs = (
            dict(mode="edge")
            if mode == "nearest"
            else dict(mode="constant", constant_values=0.25)
        )
        padded = np.pad(
            image,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            **kwargs,
        )
    else:
        padded = image
    expected = np.empty_like(padded)
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            expected[batch, ..., channel] = scipy_ndi.spline_filter(
                padded[batch, ..., channel],
                order=order,
                output=np.float32,
                mode=mode,
            )
    actual, actual_padding = interpolation._filter_input(
        image_gpu, True, mode, 0.25, order, batch_axes=(0, 3)
    )
    assert actual_padding == padding
    cp.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)
    cp.testing.assert_array_equal(image_gpu, image)


@pytest.mark.parametrize("shape", [(), (1, 9), (7, 1)])
@pytest.mark.parametrize("inplace", [False, True])
def test_spline_filter_singleton_axes(shape, inplace):
    image = cp.asarray(
        np.random.default_rng(0).uniform(0, 1, shape).astype(np.float32)
    )
    expected = scipy_ndi.spline_filter(image.get(), output=np.float32)
    actual = interpolation.spline_filter(
        image, output=image if inplace else cp.float32
    )
    cp.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("mode", ["mirror", "nearest", "grid-constant"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "uint8"])
def test_filter_input_workspace(order, mode, dtype):
    rng = np.random.default_rng(8)
    host = rng.uniform(0, 1, (9, 11))
    if dtype == "uint8":
        host *= 255
    host = host.astype(dtype)
    image = cp.asarray(host)
    working_dtype = np.result_type(dtype, np.float32)
    padding = 12 if mode in ("nearest", "grid-constant") else 0
    if mode == "nearest":
        padded = np.pad(host, padding, mode="edge")
    elif mode == "grid-constant":
        padded = np.pad(host, padding, mode="constant", constant_values=0.5)
    else:
        padded = host
    expected = scipy_ndi.spline_filter(
        padded, order=order, output=working_dtype, mode=mode
    )
    actual, actual_padding = interpolation._filter_input(
        image, True, mode, 0.5, order
    )
    assert actual_padding == padding
    assert actual.dtype == working_dtype
    assert actual.flags.c_contiguous
    tolerance = 1e-11 if working_dtype == np.dtype("float64") else 5e-5
    cp.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    cp.testing.assert_array_equal(image, host)


@pytest.mark.parametrize(
    "shape,axis", [((17,), 0), ((9, 7), 0), ((9, 7), 1), ((2, 3, 5, 7), 2)]
)
@pytest.mark.parametrize("index_type", ["int", "ptrdiff_t"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "int8"])
@pytest.mark.parametrize("order", [2, 5])
def test_general_spline_scalar_metadata(
    monkeypatch, shape, axis, index_type, dtype, order
):
    rng = np.random.default_rng(9)
    host = rng.uniform(0, 1, shape)
    if dtype == "complex64":
        host = host + 1j * rng.uniform(0, 1, shape)
    elif dtype == "int8":
        host = host * 4 - 2
    image = cp.asarray(host.astype(dtype))
    if image.ndim > 1:
        image = cp.swapaxes(image, 0, -1)
    original = image.get()
    working_dtype = np.result_type(dtype, np.float32)
    expected = scipy_ndi.spline_filter1d(
        original, order, axis, output=working_dtype, mode="reflect"
    )
    original_array, original_asarray = cp.array, cp.asarray

    def require_device_input(original_func):
        def checked(value, *args, **kwargs):
            assert isinstance(value, cp.ndarray), (
                "prefilter copied host metadata"
            )
            return original_func(value, *args, **kwargs)

        return checked

    with monkeypatch.context() as patch:
        patch.setattr(
            interpolation._util, "_get_inttype", lambda image: index_type
        )
        patch.setattr(cp, "array", require_device_input(original_array))
        patch.setattr(cp, "asarray", require_device_input(original_asarray))
        actual = interpolation.spline_filter1d(
            image, order, axis, output=cp.dtype(working_dtype), mode="reflect"
        )
    tolerance = 1e-11 if working_dtype == np.dtype("float64") else 5e-5
    cp.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    cp.testing.assert_array_equal(image, original)
