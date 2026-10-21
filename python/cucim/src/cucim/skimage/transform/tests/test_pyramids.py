# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math
import warnings

import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from cupyx import allow_synchronize
from skimage import data

from cucim.skimage._shared.testing import assert_almost_equal
from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.transform import _warps, pyramids

image = cp.asarray(data.astronaut())
image_gray = image[..., 0]


@pytest.mark.parametrize("channel_axis", [0, 1, -1])
def test_pyramid_reduce_rgb(channel_axis):
    image = cp.array(data.astronaut())
    rows, cols, dim = image.shape
    image_ = cp.moveaxis(image, source=-1, destination=channel_axis)
    out_ = pyramids.pyramid_reduce(
        image_, downscale=2, channel_axis=channel_axis
    )
    out = cp.moveaxis(out_, channel_axis, -1)
    assert_array_equal(out.shape, (rows / 2, cols / 2, dim))


def test_pyramid_reduce_gray():
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_reduce(image_gray, downscale=2, channel_axis=None)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(float(out1.ptp()), 1.0, decimal=2)
    out2 = pyramids.pyramid_reduce(
        image_gray, downscale=2, channel_axis=None, preserve_range=True
    )
    assert_almost_equal(
        float(out2.ptp()) / float(image_gray.ptp()), 1.0, decimal=2
    )


def test_pyramid_reduce_gray_defaults():
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_reduce(image_gray)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(out1.ptp(), 1.0, decimal=2)
    out2 = pyramids.pyramid_reduce(image_gray, preserve_range=True)
    assert_almost_equal(out2.ptp() / image_gray.ptp(), 1.0, decimal=2)


def test_pyramid_reduce_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*((8,) * ndim))
        out = pyramids.pyramid_reduce(img, downscale=2, channel_axis=None)
        expected_shape = cp.asarray(img.shape) / 2
        assert_array_equal(out.shape, expected_shape)


@pytest.mark.parametrize("channel_axis", [0, 1, 2, -1, -2, -3])
def test_pyramid_expand_rgb(channel_axis):
    image = cp.array(data.astronaut())
    rows, cols, dim = image.shape
    image = cp.moveaxis(image, source=-1, destination=channel_axis)
    out = pyramids.pyramid_expand(image, upscale=2, channel_axis=channel_axis)
    expected_shape = [rows * 2, cols * 2]
    expected_shape.insert(channel_axis % image.ndim, dim)
    assert_array_equal(out.shape, expected_shape)


def test_pyramid_expand_gray():
    rows, cols = image_gray.shape
    out = pyramids.pyramid_expand(image_gray, upscale=2, channel_axis=None)
    assert_array_equal(out.shape, (rows * 2, cols * 2))


def test_pyramid_expand_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*((4,) * ndim))
        out = pyramids.pyramid_expand(img, upscale=2, channel_axis=None)
        expected_shape = cp.asarray(img.shape) * 2
        assert_array_equal(out.shape, expected_shape)


@pytest.mark.parametrize("channel_axis", [0, 1, 2, -1, -2, -3])
def test_build_gaussian_pyramid_rgb(channel_axis):
    image = cp.array(data.astronaut())
    rows, cols, dim = image.shape
    image = cp.moveaxis(image, source=-1, destination=channel_axis)
    pyramid = pyramids.pyramid_gaussian(
        image, downscale=2, channel_axis=channel_axis
    )
    for layer, out in enumerate(pyramid):
        layer_shape = [rows / 2**layer, cols / 2**layer]
        layer_shape.insert(channel_axis % image.ndim, dim)
        assert out.shape == tuple(layer_shape)


def test_build_gaussian_pyramid_gray():
    rows, cols = image_gray.shape
    pyramid = pyramids.pyramid_gaussian(
        image_gray, downscale=2, channel_axis=None
    )
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_gray_defaults():
    rows, cols = image_gray.shape
    pyramid = pyramids.pyramid_gaussian(image_gray)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*((8,) * ndim))
        original_shape = cp.asarray(img.shape)
        pyramid = pyramids.pyramid_gaussian(img, downscale=2, channel_axis=None)
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2**layer
            assert_array_equal(out.shape, layer_shape)


@pytest.mark.parametrize(
    "shape, downscale, channel_axis",
    [((1, 1), 2, None), ((2, 3), 1.1, None), ((1, 1, 4), 2, -1)],
)
@pytest.mark.parametrize("max_layer", [-1, 1, 10])
def test_gaussian_pyramid_skips_terminal_reduction(
    monkeypatch, shape, downscale, channel_axis, max_layer
):
    def fail_reduce(*args, **kwargs):
        pytest.fail("pyramid_reduce called for an unchanged output shape")

    monkeypatch.setattr(pyramids, "pyramid_reduce", fail_reduce)
    layers = list(
        pyramids.pyramid_gaussian(
            cp.ones(shape),
            max_layer=max_layer,
            downscale=downscale,
            channel_axis=channel_axis,
        )
    )
    assert len(layers) == 1
    assert layers[0].shape == shape


@pytest.mark.parametrize("channel_axis", [0, 1, 2, -1, -2, -3])
def test_build_laplacian_pyramid_rgb(channel_axis):
    image = cp.array(data.astronaut())
    rows, cols, dim = image.shape
    image = cp.moveaxis(image, source=-1, destination=channel_axis)
    pyramid = pyramids.pyramid_laplacian(
        image, downscale=2, channel_axis=channel_axis
    )
    for layer, out in enumerate(pyramid):
        layer_shape = [rows / 2**layer, cols / 2**layer]
        layer_shape.insert(channel_axis % image.ndim, dim)
        assert out.shape == tuple(layer_shape)


def test_build_laplacian_pyramid_defaults():
    rows, cols = image_gray.shape
    pyramid = pyramids.pyramid_laplacian(image_gray)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*(16,) * ndim)
        original_shape = cp.asarray(img.shape)
        pyramid = pyramids.pyramid_laplacian(
            img, downscale=2, channel_axis=None
        )
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2**layer
            assert_array_equal(out.shape, layer_shape)


@pytest.mark.parametrize("channel_axis", [0, 1, 2, -1, -2, -3])
def test_laplacian_pyramid_max_layers(channel_axis):
    for downscale in [2, 3, 5, 7]:
        if channel_axis is None:
            shape = (32, 8)
            shape_without_channels = shape
        else:
            shape_without_channels = (32, 8)
            ndim = len(shape_without_channels) + 1
            n_channels = 5
            shape = list(shape_without_channels)
            shape.insert(channel_axis % ndim, n_channels)
            shape = tuple(shape)
        img = cp.ones(shape)
        pyramid = pyramids.pyramid_laplacian(
            img, downscale=downscale, channel_axis=channel_axis
        )
        max_layer = math.ceil(math.log(max(shape_without_channels), downscale))
        for layer, out in enumerate(pyramid):
            if channel_axis is None:
                out_shape_without_channels = out.shape
            else:
                assert out.shape[channel_axis] == n_channels
                out_shape_without_channels = list(out.shape)
                out_shape_without_channels.pop(channel_axis)
                out_shape_without_channels = tuple(out_shape_without_channels)

            if layer < max_layer:
                # should not reach all axes as size 1 prior to final level
                assert max(out_shape_without_channels) > 1

        # total number of images is max_layer + 1
        assert max_layer == layer

        # final layer should be size 1 on all axes
        assert out_shape_without_channels == (1, 1)


def test_check_factor():
    with pytest.raises(ValueError):
        pyramids._check_factor(0.99)
    with pytest.raises(ValueError):
        pyramids._check_factor(-2)


@pytest.mark.parametrize("sigma", [-1.0, (-1.0, 1.0)])
@pytest.mark.parametrize(
    "pyramid_func", [pyramids.pyramid_reduce, pyramids.pyramid_expand]
)
def test_negative_sigma(pyramid_func, sigma):
    with pytest.raises(
        ValueError, match="Sigma values less than zero are not valid"
    ):
        pyramid_func(cp.ones((8, 8)), sigma=sigma)


@pytest.mark.parametrize("channel_axis", [0, -1])
@pytest.mark.parametrize(
    "func",
    [
        pyramids.pyramid_reduce,
        pyramids.pyramid_expand,
        pyramids.pyramid_gaussian,
        pyramids.pyramid_laplacian,
    ],
)
def test_pyramids_sigma_sequence_with_channel_axis(func, channel_axis):
    image = cp.linspace(0, 1, 12 * 14 * 3, dtype=cp.float32).reshape(12, 14, 3)
    kwargs = dict(sigma=(0.8, 1.3), preserve_range=True)
    is_generator = func in (
        pyramids.pyramid_gaussian,
        pyramids.pyramid_laplacian,
    )
    if is_generator:
        kwargs["max_layer"] = 2
    channels = [func(image[..., channel], **kwargs) for channel in range(3)]
    channels = [
        list(result) if is_generator else [result] for result in channels
    ]
    actual = func(
        cp.moveaxis(image, -1, channel_axis),
        channel_axis=channel_axis,
        **kwargs,
    )
    actual = list(actual) if is_generator else [actual]
    assert len(actual) == len(channels[0])
    for layer, result in enumerate(actual):
        expected = cp.stack(
            [channel[layer] for channel in channels], axis=channel_axis
        )
        cp.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "dtype", ["float16", "float32", "float64", "uint8", "int64"]
)
@pytest.mark.parametrize(
    "pyramid_func", [pyramids.pyramid_gaussian, pyramids.pyramid_laplacian]
)
def test_pyramid_dtype_support(pyramid_func, dtype):
    with warnings.catch_warnings():
        # Ignore arch specific warning on arm64, armhf, ppc64el, riscv64, s390x
        # https://github.com/scikit-image/scikit-image/issues/7391
        warnings.filterwarnings(
            action="ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        img = cp.random.randn(32, 8).astype(dtype)

    pyramid = pyramid_func(img)

    float_dtype = _supported_float_type(dtype)
    assert all([im.dtype == float_dtype for im in pyramid])


@pytest.mark.parametrize(
    "func",
    [
        pyramids.pyramid_reduce,
        pyramids.pyramid_expand,
        pyramids.pyramid_gaussian,
        pyramids.pyramid_laplacian,
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize(
    "mode,cval", [("reflect", 0), ("constant", 0), ("constant", 1.5)]
)
@pytest.mark.parametrize("channel_axis", [None, 0, -1])
@pytest.mark.filterwarnings(
    "ignore:cupyx.allow_synchronize will be removed:UserWarning"
)
def test_pyramids_assume_finite(
    monkeypatch, func, dtype, order, mode, cval, channel_axis
):
    image = cp.linspace(1, 2, 33 * 35 * 3, dtype=dtype).reshape(33, 35, 3)
    if channel_axis == 0:
        image = cp.moveaxis(image, -1, 0)
    kwargs = dict(order=order, mode=mode, cval=cval, channel_axis=channel_axis)
    is_generator = func in (
        pyramids.pyramid_gaussian,
        pyramids.pyramid_laplacian,
    )
    if is_generator:
        kwargs["max_layer"] = 3
    expected = func(image, **kwargs)
    expected = list(expected) if is_generator else [expected]

    def fail_nan_path(*args, **kwargs):
        pytest.fail("pyramid level did not reuse the finite guarantee")

    monkeypatch.setattr(_warps, "_get_warp_range", fail_nan_path)
    monkeypatch.setattr(_warps, "_clip_constant_warp_output", fail_nan_path)
    # Warmed caches above avoid waiting for their private initialization copies.
    with allow_synchronize(False):
        actual = func(image, **kwargs, assume_finite=True)
        actual = list(actual) if is_generator else [actual]
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        cp.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize(
    "func",
    [
        pyramids.pyramid_reduce,
        pyramids.pyramid_expand,
        pyramids.pyramid_gaussian,
        pyramids.pyramid_laplacian,
    ],
)
@pytest.mark.parametrize("cval", [float("nan"), float("inf"), -float("inf")])
def test_pyramids_assume_finite_requires_finite_cval(func, cval):
    with pytest.raises(ValueError, match="requires a finite cval"):
        result = func(
            cp.ones((8, 8)), mode="constant", cval=cval, assume_finite=True
        )
        if func in (pyramids.pyramid_gaussian, pyramids.pyramid_laplacian):
            list(result)
