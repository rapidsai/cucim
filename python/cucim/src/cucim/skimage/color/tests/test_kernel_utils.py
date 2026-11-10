# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

from cucim.skimage import color
from cucim.skimage.color._kernel_utils import (
    elementwise_kernel,
    typed_float_literals,
)


@pytest.mark.parametrize(
    "code, expected",
    [
        ("x > 0.04045", "x > ((X)0.04045)"),
        ("x /= 12.92;", "x /= ((X)12.92);"),
        ("(116. * y) - 16.0", "(((X)116.) * y) - ((X)16.0)"),
        ("a = .5 * b", "a = ((X).5) * b"),
        ("a = 1.5e-3 * b", "a = ((X)1.5e-3) * b"),
        ("a = 1e-5 * b", "a = ((X)1e-5) * b"),
        ("v = -0.0 * x", "v = -((X)0.0) * x"),
        # integers, indices, identifiers and suffixed literals are unchanged
        ("x[3*i + 1] = 2 * y", "x[3*i + 1] = 2 * y"),
        ("x * 0.5f", "x * 0.5f"),
        ("x2.5", "x2.5"),
        # already cast
        ("pow((v + (X)0.055) / (X)1.055, (X)2.4)", None),
    ],
)
def test_typed_float_literals(code, expected):
    if expected is None:
        expected = code
    assert typed_float_literals(code) == expected


_RGB_FUNCS = [
    "rgb2lab",
    "rgb2xyz",
    "rgb2luv",
    "rgb2hsv",
    "rgb2yuv",
    "rgb2yiq",
    "rgb2ypbpr",
    "rgb2ycbcr",
    "rgb2ydbdr",
    "rgb2hed",
    "rgb2rgbcie",
    "rgb2gray",
]
_INVERSE_FUNCS = [
    ("lab2rgb", "rgb2lab"),
    ("xyz2rgb", "rgb2xyz"),
    ("luv2rgb", "rgb2luv"),
    ("hsv2rgb", "rgb2hsv"),
    ("yuv2rgb", "rgb2yuv"),
    ("yiq2rgb", "rgb2yiq"),
    ("ypbpr2rgb", "rgb2ypbpr"),
    ("ycbcr2rgb", "rgb2ycbcr"),
    ("ydbdr2rgb", "rgb2ydbdr"),
    ("rgbcie2rgb", "rgb2rgbcie"),
]


def _rgb_image(dtype):
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    img[0, 0] = 0  # black
    img[0, 1] = 255  # white
    img[0, 2] = 128  # gray
    return cp.asarray(img).astype(dtype) / (255 if dtype != cp.uint8 else 1)


@pytest.mark.parametrize("name", _RGB_FUNCS)
def test_float32_result_matches_float64(name):
    # the kernels compute in the precision of the data: a float32 input gives a
    # float32 result that agrees with the float64 one up to float32 rounding
    func = getattr(color, name)
    img64 = _rgb_image(cp.float64)
    result64 = func(img64)
    result32 = func(img64.astype(cp.float32))
    assert result32.dtype == cp.float32
    assert result64.dtype == cp.float64
    scale = max(1.0, float(cp.abs(result64).max()))
    cp.testing.assert_allclose(result32, result64, rtol=0, atol=1e-5 * scale)


@pytest.mark.parametrize("dtype", [cp.uint8, cp.uint16, cp.float16])
def test_small_dtypes_use_float32_and_match(dtype):
    # 8 and 16-bit integer and float16 inputs are converted to float32
    img64 = _rgb_image(cp.float64)
    if dtype == cp.uint8:
        img = (img64 * 255).astype(cp.uint8)
    elif dtype == cp.uint16:
        img = (img64 * 255).astype(cp.uint16) * 257
    else:
        img = img64.astype(cp.float16)
    expected = color.rgb2lab(
        img.astype(cp.float64)
        / (1 if dtype == cp.float16 else (255 if dtype == cp.uint8 else 65535))
    )
    result = color.rgb2lab(img)
    assert result.dtype == cp.float32
    cp.testing.assert_allclose(result, expected, rtol=0, atol=2e-2)


@pytest.mark.parametrize("inverse, forward", _INVERSE_FUNCS)
def test_float32_round_trip(inverse, forward):
    img = _rgb_image(cp.float32)
    back = getattr(color, inverse)(getattr(color, forward)(img))
    assert back.dtype == cp.float32
    cp.testing.assert_allclose(back, img, rtol=0, atol=1e-4)


def test_elementwise_kernel_casts_operation_and_preamble():
    kern = elementwise_kernel(
        "raw X x",
        "raw X y",
        "y[i] = scale(x[i]) + 0.25;",
        name="cucim_skimage_color_test_kernel",
        preamble="static __device__ X scale(X v) { return v * 0.5; }",
    )
    assert "((X)0.25)" in kern.operation
    assert "((X)0.5)" in kern.preamble
    x = cp.arange(4, dtype=cp.float32)
    y = cp.empty_like(x)
    kern(x, y, size=x.size)
    cp.testing.assert_allclose(y, x * 0.5 + 0.25)
