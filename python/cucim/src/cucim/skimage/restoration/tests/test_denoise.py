import functools

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal
from skimage import color, data, img_as_float

from cucim.skimage import restoration
from cucim.skimage._shared.testing import expected_warnings
from cucim.skimage._shared.utils import _supported_float_type, slice_at_axis
from cucim.skimage.metrics import structural_similarity

cp.random.seed(1234)


astro = img_as_float(data.astronaut()[:128, :128])
astro_gray = color.rgb2gray(astro)
checkerboard_gray = img_as_float(data.checkerboard())
checkerboard = color.gray2rgb(checkerboard_gray)
# versions with one odd-sized dimension
astro_gray_odd = astro_gray[:, :-1]
astro_odd = astro[:, :-1]

# transfer test images to the GPU
astro = cp.asarray(astro)
astro_gray = cp.asarray(astro_gray)
astro_gray_odd = cp.asarray(astro_gray_odd)
astro_odd = cp.asarray(astro_odd)
checkerboard = cp.asarray(checkerboard)
checkerboard_gray = cp.asarray(checkerboard_gray)

float_dtypes = [cp.float16, cp.float32, cp.float64]
try:
    float_dtypes += [cp.float128]
except AttributeError:
    pass


@pytest.mark.parametrize('dtype', float_dtypes)
def test_denoise_tv_chambolle_2d(dtype):
    # astronaut image
    img = astro_gray.astype(dtype, copy=True)
    # add noise to astronaut
    img += 0.5 * img.std() * cp.random.rand(*img.shape)
    # clip noise so that it does not exceed allowed range for float images.
    img = cp.clip(img, 0, 1)
    # denoise
    denoised_astro = restoration.denoise_tv_chambolle(img, weight=0.1)
    assert denoised_astro.dtype == _supported_float_type(img.dtype)

    # TODO: remove device to host transfers if cuda
    #       morphological_gradient is implemented
    from scipy import ndimage as ndi  # noqa

    # Convert to a floating point type supported by scipy.ndimage
    float_dtype = _supported_float_type(img.dtype)
    img = img.astype(float_dtype, copy=False)

    grad = ndi.morphological_gradient(cp.asnumpy(img), size=((3, 3)))
    grad_denoised = ndi.morphological_gradient(
        cp.asnumpy(denoised_astro), size=((3, 3)))
    # test if the total variation has decreased
    assert grad_denoised.dtype == float_dtype
    assert np.sqrt((grad_denoised ** 2).sum()) < np.sqrt((grad ** 2).sum())


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_denoise_tv_chambolle_multichannel(channel_axis):
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=0.1)

    img = cp.moveaxis(astro, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(img, weight=0.1,
                                                channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis, axis=channel_axis % img.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)

    # tile astronaut subset to generate 3D+channels data
    astro3 = cp.tile(astro[:64, :64, cp.newaxis, :], [1, 1, 2, 1])
    # modify along tiled dimension to give non-zero gradient on 3rd axis
    astro3[:, :, 0, :] = 2 * astro3[:, :, 0, :]
    denoised0 = restoration.denoise_tv_chambolle(astro3[..., 0], weight=0.1)

    astro3 = cp.moveaxis(astro3, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(astro3, weight=0.1,
                                                channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis,
                            axis=channel_axis % astro3.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)


def test_denoise_tv_chambolle_multichannel_deprecation():
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=0.1)

    with expected_warnings(["`multichannel` is a deprecated argument"]):
        denoised = restoration.denoise_tv_chambolle(astro, weight=0.1,
                                                    multichannel=True)

    # providing multichannel argument positionally also warns
    with expected_warnings(["Providing the `multichannel` argument"]):
        denoised = restoration.denoise_tv_chambolle(astro, 0.1, 2e-4, 200,
                                                    True)

    assert_array_equal(denoised[..., 0], denoised0)


def test_denoise_tv_chambolle_float_result_range():
    # astronaut image
    img = astro_gray
    int_astro = cp.multiply(img, 255).astype(np.uint8)
    assert cp.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_chambolle(int_astro,
                                                          weight=0.1)
    # test if the value range of output float data is within [0.0:1.0]
    assert denoised_int_astro.dtype == _supported_float_type(int_astro.dtype)
    assert cp.max(denoised_int_astro) <= 1.0
    assert cp.min(denoised_int_astro) >= 0.0


def test_denoise_tv_chambolle_3d():
    """Apply the TV denoising algorithm on a 3D image representing a sphere."""
    x, y, z = cp.ogrid[0:40, 0:40, 0:40]
    mask = (x - 22) ** 2 + (y - 20) ** 2 + (z - 17) ** 2 < 8 ** 2
    mask = 100 * mask.astype(float)
    mask += 60
    mask += 20 * cp.random.rand(*mask.shape)
    mask[mask < 0] = 0
    mask[mask > 255] = 255
    mask = mask.astype(np.uint8)
    res = restoration.denoise_tv_chambolle(mask, weight=0.1)
    assert res.dtype == _supported_float_type(mask.dtype)
    assert res.std() * 255 < mask.std()


def test_denoise_tv_chambolle_1d():
    """Apply the TV denoising algorithm on a 1D sinusoid."""
    x = 125 + 100 * cp.sin(cp.linspace(0, 8 * cp.pi, 1000))
    x += 20 * cp.random.rand(x.size)
    x = cp.clip(x, 0, 255)
    x = x.astype(np.uint8)
    res = restoration.denoise_tv_chambolle(x, weight=0.1)
    assert res.dtype == _supported_float_type(x.dtype)
    assert res.std() * 255 < x.std()


def test_denoise_tv_chambolle_4d():
    """ TV denoising for a 4D input."""
    im = 255 * cp.random.rand(8, 8, 8, 8)
    im = im.astype(np.uint8)
    res = restoration.denoise_tv_chambolle(im, weight=0.1)
    assert res.dtype == _supported_float_type(im.dtype)
    assert res.std() * 255 < im.std()


def test_denoise_tv_chambolle_weighting():
    # make sure a specified weight gives consistent results regardless of
    # the number of input image dimensions
    rstate = cp.random.RandomState(1234)
    img2d = astro_gray.copy()
    img2d += 0.15 * rstate.standard_normal(img2d.shape)
    img2d = cp.clip(img2d, 0, 1)

    # generate 4D image by tiling
    img4d = cp.tile(img2d[..., None, None], (1, 1, 2, 2))

    w = 0.2
    denoised_2d = restoration.denoise_tv_chambolle(img2d, weight=w)
    denoised_4d = restoration.denoise_tv_chambolle(img4d, weight=w)
    assert (structural_similarity(denoised_2d,
                                  denoised_4d[:, :, 0, 0]) > 0.99)
