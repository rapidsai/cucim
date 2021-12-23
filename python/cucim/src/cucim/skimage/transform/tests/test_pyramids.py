import math

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal
from skimage import data
from skimage._shared.testing import expected_warnings

from cucim.skimage._shared.testing import assert_almost_equal
from cucim.skimage.transform import pyramids

image = cp.asarray(data.astronaut())
image_gray = image[..., 0]


def test_pyramid_reduce_rgb_deprecated_multichannel():
    rows, cols, dim = image.shape
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        out = pyramids.pyramid_reduce(image, downscale=2, multichannel=True)
    assert_array_equal(out.shape, (rows / 2, cols / 2, dim))

    # repeat prior test, but check for positional multichannel warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        out = pyramids.pyramid_reduce(image, 2, None, 1, 'reflect', 0, True)
    assert_array_equal(out.shape, (rows / 2, cols / 2, dim))


@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_pyramid_reduce_rgb(channel_axis):
    rows, cols, dim = image.shape
    image_ = cp.moveaxis(image, -1, channel_axis)
    out_ = pyramids.pyramid_reduce(image_, downscale=2,
                                   channel_axis=channel_axis)
    out = cp.moveaxis(out_, channel_axis, -1)
    assert_array_equal(out.shape, (rows / 2, cols / 2, dim))


def test_pyramid_reduce_gray():
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_reduce(image_gray, downscale=2,
                                   channel_axis=None)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(float(out1.ptp()), 1.0, decimal=2)
    out2 = pyramids.pyramid_reduce(image_gray, downscale=2,
                                   preserve_range=True)
    assert_almost_equal(float(out2.ptp()) / float(image_gray.ptp()), 1.0,
                        decimal=2)


def test_pyramid_reduce_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*((8, ) * ndim))
        out = pyramids.pyramid_reduce(img, downscale=2, channel_axis=None)
        expected_shape = cp.asarray(img.shape) / 2
        assert_array_equal(out.shape, expected_shape)


def test_pyramid_expand_rgb():
    rows, cols, dim = image.shape
    out = pyramids.pyramid_expand(image, upscale=2, channel_axis=-1)
    assert_array_equal(out.shape, (rows * 2, cols * 2, dim))


def test_pyramid_expand_rgb_deprecated_multichannel():
    rows, cols, dim = image.shape
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        out = pyramids.pyramid_expand(image, upscale=2, multichannel=True)
    assert_array_equal(out.shape, (rows * 2, cols * 2, dim))

    # repeat prior test, but check for positional multichannel warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        out = pyramids.pyramid_expand(image, 2, None, 1, 'reflect', 0, True)
    assert_array_equal(out.shape, (rows * 2, cols * 2, dim))


def test_pyramid_expand_gray():
    rows, cols = image_gray.shape
    out = pyramids.pyramid_expand(image_gray, upscale=2, channel_axis=None)
    assert_array_equal(out.shape, (rows * 2, cols * 2))


def test_pyramid_expand_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*((4,) * ndim))
        out = pyramids.pyramid_expand(img, upscale=2,
                                      channel_axis=None)
        expected_shape = cp.asarray(img.shape) * 2
        assert_array_equal(out.shape, expected_shape)


def test_build_gaussian_pyramid_rgb():
    rows, cols, dim = image.shape
    pyramid = pyramids.pyramid_gaussian(image, downscale=2,
                                        channel_axis=-1)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_rgb_deprecated_multichannel():
    rows, cols, dim = image.shape
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        pyramid = pyramids.pyramid_gaussian(image, downscale=2,
                                            multichannel=True)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)

    # repeat prior test, but check for positional multichannel warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        pyramid = pyramids.pyramid_gaussian(image, -1, 2, None, 1, 'reflect',
                                            0, True)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_gray():
    rows, cols = image_gray.shape
    pyramid = pyramids.pyramid_gaussian(image_gray, downscale=2,
                                        channel_axis=None)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*((8,) * ndim))
        original_shape = cp.asarray(img.shape)
        pyramid = pyramids.pyramid_gaussian(img, downscale=2,
                                            channel_axis=None)
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2 ** layer
            assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_rgb():
    rows, cols, dim = image.shape
    pyramid = pyramids.pyramid_laplacian(image, downscale=2,
                                         channel_axis=-1)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_rgb_deprecated_multichannel():
    rows, cols, dim = image.shape
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        pyramid = pyramids.pyramid_laplacian(image, downscale=2,
                                             multichannel=True)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)

    # repeat prior test, but check for positional multichannel warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        pyramid = pyramids.pyramid_laplacian(image, -1, 2, None, 1, 'reflect',
                                             0, True)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = cp.random.randn(*(16,) * ndim)
        original_shape = cp.asarray(img.shape)
        pyramid = pyramids.pyramid_laplacian(img, downscale=2,
                                             channel_axis=None)
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2 ** layer
            assert_array_equal(out.shape, layer_shape)


def test_laplacian_pyramid_max_layers():
    for downscale in [2, 3, 5, 7]:
        img = cp.random.randn(32, 8)
        pyramid = pyramids.pyramid_laplacian(img, downscale=downscale,
                                             channel_axis=None)
        max_layer = int(np.ceil(math.log(np.max(img.shape), downscale)))
        for layer, out in enumerate(pyramid):
            if layer < max_layer:
                # should not reach all axes as size 1 prior to final level
                assert np.max(out.shape) > 1

        # total number of images is max_layer + 1
        assert max_layer == layer

        # final layer should be size 1 on all axes
        assert out.shape == (1, 1)


def test_check_factor():
    with pytest.raises(ValueError):
        pyramids._check_factor(0.99)
    with pytest.raises(ValueError):
        pyramids._check_factor(-2)


@pytest.mark.parametrize('dtype, expected',
                         zip(['float32', 'float64', 'uint8', 'int64'],
                             ['float32', 'float64', 'float64', 'float64']))
def test_pyramid_gaussian_dtype_support(dtype, expected):
    img = cp.random.randn(32, 8).astype(dtype)
    pyramid = pyramids.pyramid_gaussian(img)

    assert all([im.dtype == expected for im in pyramid])
