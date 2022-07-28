import math

import cupy as cp
import numpy as np
import pytest
from cupy.testing import (assert_allclose, assert_array_almost_equal,
                          assert_array_equal)
from cupyx.scipy.ndimage import map_coordinates
from numpy.testing import assert_almost_equal, assert_equal
from skimage.color.colorconv import rgb2gray
from skimage.data import astronaut, checkerboard
from skimage.draw.draw import circle_perimeter_aa

from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.feature.peak import peak_local_max
from cucim.skimage.transform._geometric import (AffineTransform,
                                                ProjectiveTransform,
                                                SimilarityTransform)
from cucim.skimage.transform._warps import (_linear_polar_mapping,
                                            _log_polar_mapping, _stackcopy,
                                            downscale_local_mean, rescale,
                                            resize, resize_local_mean, rotate,
                                            swirl, warp, warp_coords,
                                            warp_polar)
from cucim.skimage.util.dtype import _convert, img_as_float

# from skimage._shared.testing import test_parallel


cp.random.seed(0)


def test_stackcopy():
    layers = 4
    x = cp.empty((3, 3, layers))
    y = cp.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_array_almost_equal(x[..., i], y)


def test_warp_tform():
    x = cp.zeros((5, 5), dtype=cp.double)
    x[2, 2] = 1
    theta = -math.pi / 2
    tform = SimilarityTransform(
        scale=1, rotation=theta, translation=(0, 4), xp=cp
    )

    x90 = warp(x, tform, order=1)
    assert_array_almost_equal(x90, cp.rot90(x))

    x90 = warp(x, tform.inverse, order=1)
    assert_array_almost_equal(x90, cp.rot90(x))


def test_warp_callable():
    x = cp.zeros((5, 5), dtype=cp.double)
    x[2, 2] = 1
    refx = cp.zeros((5, 5), dtype=cp.double)
    refx[1, 1] = 1

    def shift(xy):
        return xy + 1

    outx = warp(x, shift, order=1)
    assert_array_almost_equal(outx, refx)


@cp.testing.with_requires('cupy>=9.0.0b2')
def test_warp_matrix():
    x = cp.zeros((5, 5), dtype=cp.double)
    x[2, 2] = 1
    refx = cp.zeros((5, 5), dtype=cp.double)
    refx[1, 1] = 1

    matrix = cp.asarray([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

    # _warp_fast
    outx = warp(x, matrix, order=1)
    assert_array_almost_equal(outx, refx)
    # check for ndimage.map_coordinates
    outx = warp(x, matrix, order=5)


def test_warp_nd():
    for dim in range(2, 8):
        shape = dim * (5,)

        x = cp.zeros(shape, dtype=cp.double)
        x_c = dim * (2,)
        x[x_c] = 1
        refx = cp.zeros(shape, dtype=cp.double)
        refx_c = dim * (1,)
        refx[refx_c] = 1

        coord_grid = dim * (slice(0, 5, 1),)
        coords = cp.array(cp.mgrid[coord_grid]) + 1

        outx = warp(x, coords, order=0, cval=0)

        assert_array_almost_equal(outx, refx)


@cp.testing.with_requires('cupy>=9.0.0b2')
def test_warp_clip():
    x = cp.zeros((5, 5), dtype=cp.double)
    x[2, 2] = 1

    outx = rescale(x, 3, order=3, clip=False, anti_aliasing=False,
                   mode='constant')
    assert outx.min() < 0

    outx = rescale(x, 3, order=3, clip=True, anti_aliasing=False,
                   mode='constant')
    assert_almost_equal(float(outx.min()), 0)
    assert_almost_equal(float(outx.max()), 1)


@pytest.mark.parametrize('order', [0, 1])
def test_warp_clip_image_containing_nans(order):
    # Test that clipping works as intended on an image with NaNs
    # Orders >1 do not produce good output when the input image has
    # NaNs, so those orders are not tested.

    x = cp.ones((15, 15), dtype=cp.float64)
    x[7, 7] = cp.nan

    outx = rotate(x, 45, order=order, cval=2, resize=True, clip=True)

    assert_almost_equal(cp.nanmin(outx).item(), 1)
    assert_almost_equal(cp.nanmax(outx).item(), 2)


@pytest.mark.parametrize('order', [0, 1])
def test_warp_clip_cval_is_nan(order):
    # Test that clipping works as intended when cval is NaN
    # Orders > 1 do not produce good output when cval is NaN, so those
    # orders are not tested.

    x = cp.ones((15, 15), dtype=cp.float64)
    x[5:-5, 5:-5] = 2

    outx = rotate(x, 45, order=order, cval=cp.nan, resize=True, clip=True)

    assert_almost_equal(cp.nanmin(outx).item(), 1)
    assert_almost_equal(cp.nanmax(outx).item(), 2)


@pytest.mark.parametrize('order', range(6))
def test_warp_clip_cval_outside_input_range(order):
    # Test that clipping behavior considers cval part of the input range

    x = cp.ones((15, 15), dtype=cp.float64)

    # Specify a cval that is outside the input range to check clipping
    outx = rotate(x, 45, order=order, cval=2, resize=True, clip=True)

    # The corners should be cval for all interpolation orders
    outx = cp.asnumpy(outx)
    assert_array_almost_equal([outx[0, 0], outx[0, -1],
                               outx[-1, 0], outx[-1, -1]], 2)

    # For all interpolation orders other than nearest-neighbor, the clipped
    # output should have some pixels with values between the input (1) and
    # cval (2) (i.e., clipping should not set them to 1)
    if order > 0:
        assert np.sum(np.less(1, outx) * np.less(outx, 2)) > 0


@pytest.mark.parametrize('order', range(6))
def test_warp_clip_cval_not_used(order):
    # Test that clipping does not consider cval part of the input range if it
    # is not used in the output image

    x = cp.ones((15, 15), dtype=cp.float64)
    x[5:-5, 5:-5] = 2

    # Transform the image by stretching it out by one pixel on each side so
    # that cval will not actually be used
    scale = 15 / (15 + 2)
    transform = AffineTransform(scale=scale, translation=(1, 1))
    outx = warp(x, transform, mode='constant', order=order, cval=0, clip=True)

    # At higher orders of interpolation, the transformed image has overshoots
    # beyond the input range that should be clipped to the range 1 to 2.  Even
    # though cval=0, the minimum value of the clipped output image should be
    # 1 and not affected by the unused cval.
    assert_array_almost_equal(outx.min(), 1)


def test_homography():
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1
    theta = -math.pi / 2
    # fmt: off
    M = cp.array([[math.cos(theta), - math.sin(theta), 0],   # noqa
                  [math.sin(theta),   math.cos(theta), 4],   # noqa
                  [0,               0,                 1]])  # noqa
    # fmt: on

    x90 = warp(x,
               inverse_map=ProjectiveTransform(M).inverse,
               order=1)
    assert_array_almost_equal(x90, cp.rot90(x))


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_rotate(dtype):
    x = cp.zeros((5, 5), dtype=dtype)
    x[1, 1] = 1
    x90 = rotate(x, 90)
    assert x90.dtype == _supported_float_type(dtype)
    assert_array_almost_equal(x90, cp.rot90(x))


def test_rotate_resize():
    x = cp.zeros((10, 10), dtype=cp.double)

    x45 = rotate(x, 45, resize=False)
    assert x45.shape == (10, 10)

    x45 = rotate(x, 45, resize=True)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)


def test_rotate_center():
    x = cp.zeros((10, 10), dtype=cp.double)
    x[4, 4] = 1
    refx = cp.zeros((10, 10), dtype=cp.double)
    refx[2, 5] = 1
    x20 = rotate(x, 20, order=0, center=(0, 0))
    assert_array_almost_equal(x20, refx)
    x0 = rotate(x20, -20, order=0, center=(0, 0))
    assert_array_almost_equal(x0, x)


def test_rotate_resize_center():
    x = cp.zeros((10, 10), dtype=cp.double)
    x[0, 0] = 1

    ref_x45 = cp.zeros((14, 14), dtype=cp.double)
    ref_x45[6, 0] = 1
    ref_x45[7, 0] = 1

    x45 = rotate(x, 45, resize=True, center=(3, 3), order=0)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)
    if False:
        # CuPy Backend: ubtle difference in rounding somewhere.
        # (visually rotations of real images look reasonable)
        assert_array_equal(x45, ref_x45)


def test_rotate_resize_90():
    x90 = rotate(cp.zeros((470, 230), dtype=cp.double), 90, resize=True)
    assert x90.shape == (230, 470)


def test_rescale():
    # same scale factor
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1
    scaled = rescale(x, 2, order=0,
                     channel_axis=None, anti_aliasing=False, mode='constant')
    ref = cp.zeros((10, 10))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(scaled, ref)

    # different scale factors
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1

    scaled = rescale(x, (2, 1), order=0,
                     channel_axis=None, anti_aliasing=False, mode='constant')
    ref = cp.zeros((10, 5))
    ref[2:4, 1] = 1
    assert_array_almost_equal(scaled, ref)


def test_rescale_invalid_scale():
    x = cp.zeros((10, 10, 3))
    with pytest.raises(ValueError):
        rescale(x, (2, 2),
                channel_axis=None, anti_aliasing=False, mode='constant')
    with pytest.raises(ValueError):
        rescale(x, (2, 2, 2),
                channel_axis=-1, anti_aliasing=False, mode='constant')


def test_rescale_multichannel():
    # 1D + channels
    x = cp.zeros((8, 3), dtype=cp.double)
    scaled = rescale(x, 2, order=0, channel_axis=-1, anti_aliasing=False,
                     mode='constant')
    assert scaled.shape == (16, 3)
    # 2D
    scaled = rescale(x, 2, order=0, channel_axis=None, anti_aliasing=False,
                     mode='constant')
    assert scaled.shape == (16, 6)

    # 2D + channels
    x = cp.zeros((8, 8, 3), dtype=cp.double)
    scaled = rescale(x, 2, order=0, channel_axis=-1, anti_aliasing=False,
                     mode='constant')
    assert scaled.shape == (16, 16, 3)
    # 3D
    scaled = rescale(x, 2, order=0, channel_axis=None, anti_aliasing=False,
                     mode='constant')
    assert scaled.shape == (16, 16, 6)

    # 3D + channels
    x = cp.zeros((8, 8, 8, 3), dtype=cp.double)
    scaled = rescale(x, 2, order=0, channel_axis=-1, anti_aliasing=False,
                     mode='constant')
    assert scaled.shape == (16, 16, 16, 3)
    # 4D
    scaled = rescale(x, 2, order=0, channel_axis=None, anti_aliasing=False,
                     mode='constant')
    assert scaled.shape == (16, 16, 16, 6)


def test_rescale_multichannel_deprecated_multiscale():
    x = cp.zeros((5, 5, 3), dtype=cp.double)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        scaled = rescale(x, (2, 1), order=0, multichannel=True,
                         anti_aliasing=False, mode='constant')
    assert scaled.shape == (10, 5, 3)

    # repeat prior test, but check for positional multichannel _warnings
    with expected_warnings(["Providing the `multichannel` argument"]):
        scaled = rescale(x, (2, 1), 0, 'constant', 0, True, False, True,
                         anti_aliasing=False)
    assert scaled.shape == (10, 5, 3)


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_rescale_channel_axis_multiscale(channel_axis):
    x = cp.zeros((5, 5, 3), dtype=cp.double)
    x = cp.moveaxis(x, -1, channel_axis)
    scaled = rescale(x, scale=(2, 1), order=0, channel_axis=channel_axis,
                     anti_aliasing=False, mode='constant')
    scaled = cp.moveaxis(scaled, channel_axis, -1)
    assert scaled.shape == (10, 5, 3)


def test_rescale_multichannel_defaults():
    x = cp.zeros((8, 3), dtype=cp.double)
    scaled = rescale(x, 2, order=0, anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (16, 6))

    x = cp.zeros((8, 8, 3), dtype=cp.double)
    scaled = rescale(x, 2, order=0, anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (16, 16, 6))


def test_resize2d():
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1
    resized = resize(x, (10, 10), order=0, anti_aliasing=False,
                     mode='constant')
    ref = cp.zeros((10, 10))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)


def test_resize3d_keep():
    # keep 3rd dimension
    x = cp.zeros((5, 5, 3), dtype=cp.double)
    x[1, 1, :] = 1
    resized = resize(x, (10, 10), order=0, anti_aliasing=False,
                     mode='constant')
    with pytest.raises(ValueError):
        # output_shape too short
        resize(x, (10,), order=0, anti_aliasing=False, mode='constant')
    ref = cp.zeros((10, 10, 3))
    ref[2:4, 2:4, :] = 1
    assert_array_almost_equal(resized, ref)
    resized = resize(x, (10, 10, 3), order=0, anti_aliasing=False,
                     mode='constant')
    assert_array_almost_equal(resized, ref)


def test_resize3d_resize():
    # resize 3rd dimension
    x = cp.zeros((5, 5, 3), dtype=cp.double)
    x[1, 1, :] = 1
    resized = resize(x, (10, 10, 1), order=0, anti_aliasing=False,
                     mode='constant')
    ref = cp.zeros((10, 10, 1))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)


def test_resize3d_2din_3dout():
    # 3D output with 2D input
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1
    resized = resize(x, (10, 10, 1), order=0, anti_aliasing=False,
                     mode='constant')
    ref = cp.zeros((10, 10, 1))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)


def test_resize2d_4d():
    # resize with extra output dimensions
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1
    out_shape = (10, 10, 1, 1)
    resized = resize(x, out_shape, order=0, anti_aliasing=False,
                     mode='constant')
    ref = cp.zeros(out_shape)
    ref[2:4, 2:4, ...] = 1
    assert_array_almost_equal(resized, ref)


def test_resize_nd():
    for dim in range(1, 6):
        shape = 2 + np.arange(dim) * 2
        x = cp.ones(tuple(shape))
        out_shape = cp.asarray(shape) * 1.5
        resized = resize(x, out_shape, order=0, mode='reflect',
                         anti_aliasing=False)
        expected_shape = 1.5 * shape
        assert_equal(resized.shape, expected_shape)
        assert cp.all(resized == 1)


def test_resize3d_bilinear():
    # bilinear 3rd dimension
    x = cp.zeros((5, 5, 2), dtype=cp.double)
    x[1, 1, 0] = 0
    x[1, 1, 1] = 1
    resized = resize(x, (10, 10, 1), order=1, mode='constant',
                     anti_aliasing=False)
    ref = cp.zeros((10, 10, 1))
    ref[1:5, 1:5, :] = 0.03125
    ref[1:5, 2:4, :] = 0.09375
    ref[2:4, 1:5, :] = 0.09375
    ref[2:4, 2:4, :] = 0.28125
    assert_array_almost_equal(resized, ref)


def test_resize_dtype():
    x = cp.zeros((5, 5))
    x_f32 = x.astype(cp.float32)
    x_u8 = x.astype(cp.uint8)
    x_b = x.astype(bool)

    assert resize(x, (10, 10), preserve_range=False).dtype == x.dtype
    assert resize(x, (10, 10), preserve_range=True).dtype == x.dtype
    assert resize(x_u8, (10, 10), preserve_range=False).dtype == cp.float32
    assert resize(x_u8, (10, 10), preserve_range=True).dtype == cp.float32
    assert resize(x_b, (10, 10), preserve_range=False).dtype == bool
    assert resize(x_b, (10, 10), preserve_range=True).dtype == bool
    assert resize(x_f32, (10, 10), preserve_range=False).dtype == x_f32.dtype
    assert resize(x_f32, (10, 10), preserve_range=True).dtype == x_f32.dtype


@pytest.mark.parametrize('order', [0, 1])
@pytest.mark.parametrize('preserve_range', [True, False])
@pytest.mark.parametrize('anti_aliasing', [True, False])
@pytest.mark.parametrize('dtype', [cp.float64, cp.uint8])
def test_resize_clip(order, preserve_range, anti_aliasing, dtype):
    # test if clip as expected
    if dtype == cp.uint8 and (preserve_range or order == 0):
        expected_max = 255
    else:
        expected_max = 1.0
    x = cp.ones((5, 5), dtype=dtype)
    if dtype == cp.uint8:
        x *= 255
    resized = resize(x, (3, 3), order=order, preserve_range=preserve_range,
                     anti_aliasing=anti_aliasing)

    assert abs(float(resized.max()) - expected_max) < 1e-14


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_swirl(dtype):
    image = img_as_float(cp.array(checkerboard())).astype(dtype, copy=False)
    float_dtype = _supported_float_type(dtype)

    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}

    # with expected_warnings(["Bi-quadratic.*bug"]):
    swirled = swirl(image, strength=10, **swirl_params)
    unswirled = swirl(swirled, strength=-10, **swirl_params)
    assert swirled.dtype == unswirled.dtype == float_dtype

    assert cp.mean(cp.abs(image - unswirled)) < 0.01

    swirl_params.pop('mode')

    # with expected_warnings(["Bi-quadratic.*bug"]):
    swirled = swirl(image, strength=10, **swirl_params)
    unswirled = swirl(swirled, strength=-10, **swirl_params)
    assert swirled.dtype == unswirled.dtype == float_dtype

    assert cp.mean(cp.abs(image[1:-1, 1:-1] - unswirled[1:-1, 1:-1])) < 0.01


def test_const_cval_out_of_range():
    img = cp.random.randn(100, 100)
    cval = -10
    warped = warp(img, AffineTransform(translation=(10, 10)), cval=cval)
    assert cp.sum(warped == cval) == (2 * 100 * 10 - 10 * 10)


def test_warp_identity():
    img = img_as_float(cp.array(rgb2gray(astronaut())))
    assert len(img.shape) == 2
    assert cp.allclose(img, warp(img, AffineTransform(rotation=0)))
    assert not cp.allclose(img, warp(img, AffineTransform(rotation=0.1)))

    img = cp.asnumpy(img)
    rgb_img = np.transpose(
        np.asarray(np.array([img, np.zeros_like(img), img])), (1, 2, 0)
    )
    rgb_img = cp.array(rgb_img)
    warped_rgb_img = warp(rgb_img, AffineTransform(rotation=0.1))
    assert cp.allclose(rgb_img, warp(rgb_img, AffineTransform(rotation=0)))
    assert not cp.allclose(rgb_img, warped_rgb_img)
    # assert no cross-talk between bands
    assert cp.all(0 == warped_rgb_img[:, :, 1])


@cp.testing.with_requires('cupy>=9.0.0b2')
def test_warp_coords_example():
    image = cp.array(astronaut().astype(cp.float32))
    assert 3 == image.shape[2]
    tform = SimilarityTransform(translation=(0, -10), xp=cp)
    coords = warp_coords(tform, (30, 30, 3))
    map_coordinates(image[:, :, 0], coords[:2])


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.int32, cp.float16, cp.float32, cp.float64]
)
def test_downsize(dtype):
    x = cp.zeros((10, 10), dtype=dtype)
    x[2:4, 2:4] = 1
    scaled = resize(x, (5, 5), order=0, anti_aliasing=False, mode='constant')
    expected_dtype = cp.float32 if dtype == cp.float16 else dtype
    assert scaled.dtype == expected_dtype
    assert_equal(scaled.shape, (5, 5))
    assert_equal(float(scaled[1, 1]), 1)
    assert_equal(float(scaled[2:, :].sum()), 0)
    assert_equal(float(scaled[:, 2:].sum()), 0)


def test_downsize_anti_aliasing():
    x = cp.zeros((10, 10), dtype=cp.double)
    x[2, 2] = 1
    scaled = resize(x, (5, 5), order=1, anti_aliasing=True, mode='constant')
    assert_equal(scaled.shape, (5, 5))
    assert cp.all(scaled[:3, :3] > 0)
    assert_equal(float(scaled[3:, :].sum()), 0)
    assert_equal(float(scaled[:, 3:].sum()), 0)

    sigma = 0.125
    out_size = (5, 5)
    resize(x, out_size, order=1, mode='constant',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='edge',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='symmetric',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='reflect',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='wrap',
           anti_aliasing=True, anti_aliasing_sigma=sigma)

    with pytest.raises(ValueError):  # Unknown mode, or cannot translate mode
        resize(x, out_size, order=1, mode='non-existent',
               anti_aliasing=True, anti_aliasing_sigma=sigma)


def test_downsize_anti_aliasing_invalid_stddev():
    x = cp.zeros((10, 10), dtype=cp.double)
    with pytest.raises(ValueError):
        resize(x, (5, 5), order=0, anti_aliasing=True, anti_aliasing_sigma=-1,
               mode='constant')
    with expected_warnings(["Anti-aliasing standard deviation greater"]):
        resize(x, (5, 15), order=0, anti_aliasing=True,
               anti_aliasing_sigma=(1, 1), mode="reflect")
        resize(x, (5, 15), order=0, anti_aliasing=True,
               anti_aliasing_sigma=(0, 1), mode="reflect")


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.int32, cp.float16, cp.float32, cp.float64]
)
def test_downscale(dtype):
    x = cp.zeros((10, 10), dtype=dtype)
    x[2:4, 2:4] = 1
    scaled = rescale(x, 0.5, order=0, anti_aliasing=False,
                     channel_axis=None, mode='constant')
    expected_dtype = cp.float32 if dtype == cp.float16 else dtype
    assert scaled.dtype == expected_dtype
    assert_equal(scaled.shape, (5, 5))
    assert_equal(float(scaled[1, 1]), 1)
    assert_equal(float(scaled[2:, :].sum()), 0)
    assert_equal(float(scaled[:, 2:].sum()), 0)


def test_downscale_anti_aliasing():
    x = cp.zeros((10, 10), dtype=cp.double)
    x[2, 2] = 1
    scaled = rescale(x, 0.5, order=1, anti_aliasing=True,
                     channel_axis=None, mode='constant')
    assert_equal(scaled.shape, (5, 5))
    assert cp.all(scaled[:3, :3] > 0)
    assert_equal(float(scaled[3:, :].sum()), 0)
    assert_equal(float(scaled[:, 3:].sum()), 0)


def test_downscale_to_the_limit():
    img = cp.random.rand(3, 4)
    out = rescale(img, 1e-3)

    assert out.size == 1


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.int32, cp.float16, cp.float32, cp.float64]
)
def test_downscale_local_mean(dtype):
    image1 = cp.arange(4 * 6, dtype=dtype).reshape(4, 6)
    out1 = downscale_local_mean(image1, (2, 3))
    float_dtype = dtype if cp.dtype(dtype).kind == 'f' else cp.float64
    assert out1.dtype == float_dtype

    expected1 = np.array([[4., 7.],
                          [16., 19.]])
    assert_array_equal(expected1, out1)

    image2 = cp.arange(5 * 8, dtype=dtype).reshape(5, 8)
    out2 = downscale_local_mean(image2, (4, 5))
    assert out2.dtype == float_dtype
    expected2 = np.array([[14., 10.8],
                          [8.5, 5.7]])
    rtol = 1e-3 if dtype == cp.float16 else 1e-7
    assert_allclose(expected2, out2, rtol=rtol)


def test_invalid():
    with pytest.raises(ValueError):
        warp(cp.ones((4, 3, 3, 3)), SimilarityTransform())


def test_inverse():
    tform = SimilarityTransform(scale=0.5, rotation=0.1)
    inverse_tform = SimilarityTransform(matrix=cp.linalg.inv(tform.params))
    image = cp.arange(10 * 10).reshape(10, 10).astype(cp.double)
    assert_array_equal(warp(image, inverse_tform), warp(image, tform.inverse))


def test_slow_warp_nonint_oshape():
    image = cp.random.rand(5, 5)

    with pytest.raises(ValueError):
        warp(image, lambda xy: xy, output_shape=(13.1, 19.5))

    warp(image, lambda xy: xy, output_shape=(13.0001, 19.9999))


def test_keep_range():
    image = cp.linspace(0, 2, 25).reshape(5, 5)
    out = rescale(image, 2, preserve_range=False, clip=True, order=0,
                  mode='constant', channel_axis=None, anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2

    out = rescale(image, 2, preserve_range=True, clip=True, order=0,
                  mode='constant', channel_axis=None, anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2

    out = rescale(image.astype(cp.uint8), 2, preserve_range=False,
                  mode='constant', channel_axis=None, anti_aliasing=False,
                  clip=True, order=0)
    assert out.min() == 0
    assert out.max() == 2


def test_zero_image_size():
    with pytest.raises(ValueError):
        warp(cp.zeros(0), SimilarityTransform())
    with pytest.raises(ValueError):
        warp(cp.zeros((0, 10)), SimilarityTransform())
    with pytest.raises(ValueError):
        warp(cp.zeros((10, 0)), SimilarityTransform())
    with pytest.raises(ValueError):
        warp(cp.zeros((10, 10, 0)), SimilarityTransform())


def test_linear_polar_mapping():
    # fmt: off
    output_coords = cp.array([[0, 0],
                             [0, 90],
                             [0, 180],
                             [0, 270],
                             [99, 0],
                             [99, 180],
                             [99, 270],
                             [99, 45]])
    ground_truth = cp.array([[100, 100],
                             [100, 100],
                             [100, 100],
                             [100, 100],
                             [199, 100],
                             [1, 100],
                             [100, 1],
                             [170.00357134, 170.00357134]])
    # fmt: on
    k_angle = 360 / (2 * math.pi)
    k_radius = 1
    center = (100, 100)
    coords = _linear_polar_mapping(output_coords, k_angle, k_radius, center)
    assert cp.allclose(coords, ground_truth)


def test_log_polar_mapping():
    # fmt: off
    output_coords = cp.array([[0, 0],
                              [0, 90],
                              [0, 180],
                              [0, 270],
                              [99, 0],
                              [99, 180],
                              [99, 270],
                              [99, 45]])
    ground_truth = cp.array([[101, 100],
                             [100, 101],
                             [99, 100],
                             [100, 99],
                             [195.4992586, 100],
                             [4.5007414, 100],
                             [100, 4.5007414],
                             [167.52817336, 167.52817336]])
    # fmt: on
    k_angle = 360 / (2 * math.pi)
    k_radius = 100 / cp.log(100)
    center = (100, 100)
    coords = _log_polar_mapping(output_coords, k_angle, k_radius, center)
    assert cp.allclose(coords, ground_truth)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_linear_warp_polar(dtype):
    radii = [5, 10, 15, 20]
    image = cp.zeros([51, 51])
    for rad in radii:
        rr, cc, val = circle_perimeter_aa(25, 25, rad)
        image[rr, cc] = val
    image = image.astype(dtype, copy=False)
    warped = warp_polar(image, radius=25)
    assert warped.dtype == _supported_float_type(dtype)
    profile = warped.mean(axis=0)
    peaks = cp.asnumpy(peak_local_max(profile))
    assert np.alltrue([peak in radii for peak in peaks])


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_log_warp_polar(dtype):
    radii = [math.exp(2), math.exp(3), math.exp(4), math.exp(5),
             math.exp(5) - 1, math.exp(5) + 1]
    radii = [int(x) for x in radii]
    image = cp.zeros([301, 301])
    for rad in radii:
        rr, cc, val = circle_perimeter_aa(150, 150, rad)
        image[rr, cc] = val
    image = image.astype(dtype, copy=False)
    warped = warp_polar(image, radius=200, scaling='log')
    assert warped.dtype == _supported_float_type(dtype)
    profile = warped.mean(axis=0)
    peaks_coord = peak_local_max(profile)
    peaks_coord.sort(axis=0)
    gaps = cp.asnumpy(peaks_coord[1:] - peaks_coord[:-1])
    assert np.alltrue([x >= 38 and x <= 40 for x in gaps])


def test_invalid_scaling_polar():
    with pytest.raises(ValueError):
        warp_polar(cp.zeros((10, 10)), (5, 5), scaling="invalid")
    with pytest.raises(ValueError):
        warp_polar(cp.zeros((10, 10)), (5, 5), scaling=None)


def test_invalid_dimensions_polar():
    with pytest.raises(ValueError):
        warp_polar(cp.zeros((10, 10, 3)), (5, 5))
    with pytest.raises(ValueError):
        warp_polar(cp.zeros((10, 10)), (5, 5), channel_axis=-1)
    with pytest.raises(ValueError):
        warp_polar(cp.zeros((10, 10, 10, 3)), (5, 5), channel_axis=-1)


def test_bool_img_rescale():
    img = cp.ones((12, 18), dtype=bool)
    img[2:-2, 4:-4] = False
    res = rescale(img, 0.5)

    expected = cp.ones((6, 9))
    expected[1:-1, 2:-2] = False

    assert_array_equal(res, expected)


def test_bool_img_resize():
    img = cp.ones((12, 18), dtype=bool)
    img[2:-2, 4:-4] = False
    res = resize(img, (6, 9))

    expected = cp.ones((6, 9))
    expected[1:-1, 2:-2] = False

    assert_array_equal(res, expected)


def test_bool_and_anti_aliasing_errors():
    img = cp.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError):
        rescale(img, 0.5, anti_aliasing=True)

    with pytest.raises(ValueError):
        resize(img, (5, 5), anti_aliasing=True)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_bool_nonzero_order_errors(order):
    img = cp.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError):
        rescale(img, 0.5, order=order)

    with pytest.raises(ValueError):
        resize(img, (5, 5), order=order)

    with pytest.raises(ValueError):
        warp(img, cp.eye(3), order=order)


@pytest.mark.parametrize('dtype', [cp.uint8, bool, cp.float32, cp.float64])
def test_order_0_warp_dtype(dtype):

    img = _convert(cp.array(astronaut()[:10, :10, 0]), dtype)

    assert resize(img, (12, 12), order=0).dtype == dtype
    assert rescale(img, 0.5, order=0).dtype == dtype
    assert rotate(img, 45, order=0).dtype == dtype
    assert warp_polar(img, order=0).dtype == dtype
    assert swirl(img, order=0).dtype == dtype


@pytest.mark.parametrize(
    'dtype',
    [cp.uint8, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize('order', [1, 3, 5])
def test_nonzero_order_warp_dtype(dtype, order):

    img = _convert(cp.array(astronaut()[:10, :10, 0]), dtype)

    float_dtype = _supported_float_type(dtype)

    assert resize(img, (12, 12), order=order).dtype == float_dtype
    assert rescale(img, 0.5, order=order).dtype == float_dtype
    assert rotate(img, 45, order=order).dtype == float_dtype
    assert warp_polar(img, order=order).dtype == float_dtype
    assert swirl(img, order=order).dtype == float_dtype


def test_resize_local_mean2d():
    x = cp.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    resized = resize_local_mean(x, (10, 10))
    ref = cp.zeros((10, 10))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_resize_local_mean3d_keep(channel_axis):
    # keep 3rd dimension
    nch = 3
    x = cp.zeros((5, 5, nch), dtype=np.double)
    x[1, 1, :] = 1
    # move channels to expected dimension
    x = cp.moveaxis(x, -1, channel_axis)
    resized = resize_local_mean(x, (10, 10), channel_axis=channel_axis)
    # move channels back to last axis to match the reference image
    resized = cp.moveaxis(resized, channel_axis, -1)
    with pytest.raises(ValueError):
        # output_shape too short
        resize_local_mean(x, (10, ))
    ref = cp.zeros((10, 10, nch))
    ref[2:4, 2:4, :] = 1
    assert_array_almost_equal(resized, ref)

    channel_axis = channel_axis % x.ndim
    spatial_shape = (10, 10)
    out_shape = (
        spatial_shape[:channel_axis] + (nch,) + spatial_shape[channel_axis:]
    )
    resized = resize_local_mean(x, out_shape)
    # move channels back to last axis to match the reference image
    resized = cp.moveaxis(resized, channel_axis, -1)
    assert_array_almost_equal(resized, ref)


def test_resize_local_mean3d_resize():
    # resize 3rd dimension
    x = cp.zeros((5, 5, 3), dtype=cp.double)
    x[1, 1, :] = 1
    resized = resize_local_mean(x, (10, 10, 1))
    ref = cp.zeros((10, 10, 1))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)

    # can't resize along specified channel axis
    with pytest.raises(ValueError):
        resize_local_mean(x, (10, 10, 1), channel_axis=-1)


def test_resize_local_mean3d_2din_3dout():
    # 3D output with 2D input
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1
    resized = resize_local_mean(x, (10, 10, 1))
    ref = cp.zeros((10, 10, 1))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)


def test_resize_local_mean2d_4d():
    # resize with extra output dimensions
    x = cp.zeros((5, 5), dtype=cp.double)
    x[1, 1] = 1
    out_shape = (10, 10, 1, 1)
    resized = resize_local_mean(x, out_shape)
    ref = cp.zeros(out_shape)
    ref[2:4, 2:4, ...] = 1
    assert_array_almost_equal(resized, ref)


@pytest.mark.parametrize("dim", range(1, 6))
def test_resize_local_mean_nd(dim):
    shape = tuple(2 + np.arange(dim) * 2)
    x = cp.ones(shape)
    out_shape = (np.asarray(shape) * 1.5).astype(int)
    resized = resize_local_mean(x, out_shape)
    expected_shape = tuple(1.5 * np.asarray(shape))
    assert_array_equal(resized.shape, expected_shape)
    assert_array_equal(resized, 1)


def test_resize_local_mean3d():
    x = cp.zeros((5, 5, 2), dtype=cp.double)
    x[1, 1, 0] = 0
    x[1, 1, 1] = 1
    resized = resize_local_mean(x, (10, 10, 1))
    ref = cp.zeros((10, 10, 1))
    ref[2:4, 2:4, :] = 0.5
    assert_array_almost_equal(resized, ref)
    resized = resize_local_mean(x, (10, 10, 1), grid_mode=False)
    ref[1, 1, :] = 0.0703125
    ref[2, 2, :] = 0.5
    ref[3, 3, :] = 0.3828125
    ref[1, 2, :] = ref[2, 1, :] = 0.1875
    ref[1, 3, :] = ref[3, 1, :] = 0.1640625
    ref[2, 3, :] = ref[3, 2, :] = 0.4375
    assert_array_almost_equal(resized, ref)


def test_resize_local_mean_dtype():
    x = cp.zeros((5, 5))
    x_f32 = x.astype(cp.float32)
    x_u8 = x.astype(cp.uint8)
    x_b = x.astype(bool)

    assert resize_local_mean(x, (10, 10),
                             preserve_range=False).dtype == x.dtype
    assert resize_local_mean(x, (10, 10),
                             preserve_range=True).dtype == x.dtype
    assert resize_local_mean(x_u8, (10, 10),
                             preserve_range=False).dtype == cp.float32
    assert resize_local_mean(x_u8, (10, 10),
                             preserve_range=True).dtype == cp.float32
    assert resize_local_mean(x_b, (10, 10),
                             preserve_range=False).dtype == cp.float32
    assert resize_local_mean(x_b, (10, 10),
                             preserve_range=True).dtype == cp.float32
    assert resize_local_mean(x_f32, (10, 10),
                             preserve_range=False).dtype == x_f32.dtype
    assert resize_local_mean(x_f32, (10, 10),
                             preserve_range=True).dtype == x_f32.dtype
