# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

from cucim.skimage import color, img_as_float64
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


_ALL_CONVERSIONS = (
    [n for n in _RGB_FUNCS]
    + [inv for inv, _ in _INVERSE_FUNCS]
    + ["hed2rgb", "xyz2lab", "lab2xyz", "lab2lch", "lch2lab"]
)


@pytest.mark.parametrize("name", sorted(set(_ALL_CONVERSIONS)))
@pytest.mark.parametrize(
    "dtype", [cp.uint8, cp.uint16, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize("contiguous", [True, False])
def test_conversions_do_not_modify_their_input(name, dtype, contiguous):
    # the kernels work on local variables and write to a new array: the input
    # array is neither changed nor returned
    rng = np.random.default_rng(7)
    shape = (30, 20, 3) if contiguous else (20, 30, 3)
    values = rng.random(shape)
    if np.dtype(dtype).kind in "iu":
        values = values * np.iinfo(dtype).max
    img = cp.asarray(values.astype(dtype))
    if not contiguous:
        img = img.transpose(1, 0, 2)
        assert not img.flags.c_contiguous
    original = img.copy()
    result = getattr(color, name)(img)
    cp.testing.assert_array_equal(img, original)
    assert not cp.shares_memory(result, img)
    if name != "rgb2gray":
        assert result.shape == img.shape


@pytest.mark.parametrize("tospace", ["hsv", "xyz", "yuv", "ycbcr", "rgb cie"])
def test_convert_colorspace_does_not_modify_its_input(tospace):
    img = _rgb_image(cp.float32)
    original = img.copy()
    result = color.convert_colorspace(img, "rgb", tospace)
    cp.testing.assert_array_equal(img, original)
    assert not cp.shares_memory(result, img)


@pytest.mark.parametrize("nchannels", [3, 5])
def test_lab2lch_lch2lab_extra_channels_and_non_contiguous(nchannels):
    rng = np.random.default_rng(8)
    lab = cp.asarray(
        (rng.normal(size=(nchannels, 20, 30)) * 40).astype(np.float32)
    ).transpose(1, 2, 0)
    assert not lab.flags.c_contiguous
    lch = color.lab2lch(lab)
    a, b = lab[..., 1], lab[..., 2]
    expected_h = cp.arctan2(b, a)
    expected_h = cp.where(expected_h < 0, expected_h + 2 * np.pi, expected_h)
    cp.testing.assert_array_equal(lch[..., 0], lab[..., 0])
    cp.testing.assert_allclose(lch[..., 1], cp.hypot(a, b), rtol=1e-6)
    cp.testing.assert_allclose(lch[..., 2], expected_h, rtol=1e-6, atol=1e-6)
    cp.testing.assert_array_equal(lch[..., 3:], lab[..., 3:])
    back = color.lch2lab(lch)
    cp.testing.assert_allclose(back, lab, rtol=0, atol=1e-3)
    cp.testing.assert_array_equal(back[..., 3:], lab[..., 3:])


@pytest.mark.parametrize(
    "name",
    ["rgb2lab", "rgb2xyz", "rgb2luv", "rgb2hsv", "rgb2yuv", "rgb2ycbcr"]
    + [
        "rgb2yiq",
        "rgb2ypbpr",
        "rgb2ydbdr",
        "rgb2rgbcie",
        "rgb2gray",
        "rgb2hed",
    ],
)
@pytest.mark.parametrize(
    "dtype", [cp.uint8, cp.uint16, cp.float16, cp.int8, cp.int16, cp.int32]
)
@pytest.mark.parametrize("contiguous", [True, False])
def test_integer_and_float16_input_is_converted_in_the_kernel(
    name, dtype, contiguous
):
    # the kernels convert the 8 and 16-bit integer and float16 values
    # themselves (no separate img_as_float pass); the result is identical to
    # the one for the converted array
    from cucim.skimage import img_as_float

    rng = np.random.default_rng(9)
    info = np.iinfo(dtype) if np.dtype(dtype).kind in "iu" else None
    if info is None:
        img = rng.random((30, 20, 3)).astype(dtype)
    else:
        img = rng.integers(info.min, info.max, (30, 20, 3), dtype=dtype)
    img = cp.asarray(img)
    if not contiguous:
        img = cp.asarray(np.ascontiguousarray(img.get().transpose(1, 0, 2)))
        img = img.transpose(1, 0, 2)
        assert not img.flags.c_contiguous
    func = getattr(color, name)
    expected = func(img_as_float(img))
    result = func(img)
    assert result.dtype == expected.dtype
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "illuminant, observer", [("D65", "2"), ("D50", "10"), ("A", "R")]
)
@pytest.mark.parametrize("dtype", [cp.uint8, cp.float32, cp.float64])
def test_fused_rgb2lab_matches_the_two_step_conversion(
    illuminant, observer, dtype
):
    img = _rgb_image(dtype)
    expected = color.xyz2lab(color.rgb2xyz(img), illuminant, observer)
    result = color.rgb2lab(img, illuminant, observer)
    assert result.dtype == expected.dtype
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("illuminant, observer", [("D65", "2"), ("D50", "10")])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_fused_lab2rgb_matches_the_two_step_conversion(
    illuminant, observer, dtype
):
    rng = np.random.default_rng(10)
    lab = cp.asarray(
        np.stack(
            [
                rng.random((40, 50)) * 100,
                rng.normal(size=(40, 50)) * 40,
                rng.normal(size=(40, 50)) * 40,
            ],
            axis=-1,
        ).astype(dtype)
    )
    with pytest.warns(UserWarning, match="negative Z values"):
        xyz = color.lab2xyz(lab, illuminant, observer)
    expected = color.xyz2rgb(xyz)
    with pytest.warns(UserWarning, match="negative Z values"):
        result = color.lab2rgb(lab, illuminant, observer)
    cp.testing.assert_array_equal(result, expected)


def test_lab_to_xyz_warning_counts_the_invalid_pixels():
    # Z < 0 for L = 10, b = 100: 7 + 3 * 5 = 22 pixels; the others are valid
    lab = cp.zeros((4, 8, 3), dtype=cp.float32)
    lab[..., 0] = 50
    invalid = cp.zeros((4, 8), dtype=bool)
    invalid[0, :7] = True
    invalid[2, :5] = True
    invalid[3, 3:8] = True
    invalid[1, 1] = True
    lab[invalid, 0] = 10
    lab[invalid, 2] = 100
    n = int(invalid.sum())
    with pytest.warns(UserWarning, match=f"{n} negative Z values") as rec:
        xyz = color.lab2xyz(lab)
    assert len(rec) == 1
    assert xyz.shape == lab.shape
    with pytest.warns(UserWarning, match=f"{n} negative Z values"):
        color.lab2rgb(lab)


@pytest.mark.parametrize("dtype", [cp.uint8, cp.float32, cp.float64])
def test_separate_stains_is_not_negative_and_matches_reference(dtype):
    # the clipping of negative stain values is done in the kernel
    img = _rgb_image(dtype)
    stains = color.separate_stains(img, color.hed_from_rgb)
    assert float(stains.min()) >= 0
    rgb = img_as_float64(img)
    rgb = cp.maximum(rgb, 1e-6)
    log_adjust = np.log(1e-6)
    expected = cp.maximum(
        (cp.log(rgb) / log_adjust) @ cp.asarray(color.hed_from_rgb), 0
    )
    tol = 1e-5 if dtype == cp.float32 or dtype == cp.uint8 else 1e-12
    cp.testing.assert_allclose(stains, expected, rtol=0, atol=tol)
