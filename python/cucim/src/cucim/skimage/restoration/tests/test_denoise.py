# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import functools

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal
from skimage import color, data, img_as_float

from cucim.skimage import restoration
from cucim.skimage._shared.utils import _supported_float_type, slice_at_axis
from cucim.skimage.metrics import peak_signal_noise_ratio, structural_similarity

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


@pytest.mark.parametrize("dtype", float_dtypes)
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
        cp.asnumpy(denoised_astro), size=((3, 3))
    )
    # test if the total variation has decreased
    assert grad_denoised.dtype == float_dtype
    assert np.sqrt((grad_denoised**2).sum()) < np.sqrt((grad**2).sum())


@pytest.mark.parametrize("channel_axis", [0, 1, 2, -1])
def test_denoise_tv_chambolle_multichannel(channel_axis):
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=0.1)

    img = cp.moveaxis(astro, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(
        img, weight=0.1, channel_axis=channel_axis
    )
    _at = functools.partial(slice_at_axis, axis=channel_axis % img.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)

    # tile astronaut subset to generate 3D+channels data
    astro3 = cp.tile(astro[:64, :64, cp.newaxis, :], [1, 1, 2, 1])
    # modify along tiled dimension to give non-zero gradient on 3rd axis
    astro3[:, :, 0, :] = 2 * astro3[:, :, 0, :]
    denoised0 = restoration.denoise_tv_chambolle(astro3[..., 0], weight=0.1)

    astro3 = cp.moveaxis(astro3, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(
        astro3, weight=0.1, channel_axis=channel_axis
    )
    _at = functools.partial(slice_at_axis, axis=channel_axis % astro3.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)


def test_denoise_tv_chambolle_float_result_range():
    # astronaut image
    img = astro_gray
    int_astro = cp.multiply(img, 255).astype(np.uint8)
    assert cp.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_chambolle(int_astro, weight=0.1)
    # test if the value range of output float data is within [0.0:1.0]
    assert denoised_int_astro.dtype == _supported_float_type(int_astro.dtype)
    assert cp.max(denoised_int_astro) <= 1.0
    assert cp.min(denoised_int_astro) >= 0.0


def test_denoise_tv_chambolle_3d():
    """Apply the TV denoising algorithm on a 3D image representing a sphere."""
    x, y, z = cp.ogrid[0:40, 0:40, 0:40]
    mask = (x - 22) ** 2 + (y - 20) ** 2 + (z - 17) ** 2 < 8**2
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
    """TV denoising for a 4D input."""
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
    assert (
        structural_similarity(
            denoised_2d, denoised_4d[:, :, 0, 0], data_range=1.0
        )
        > 0.98
    )


@pytest.mark.parametrize("fast_mode", [False, True])
def test_denoise_nl_means_2d(fast_mode):
    img = cp.zeros((40, 40))
    img[10:-10, 10:-10] = 1.0
    sigma = 0.3
    img += sigma * cp.random.standard_normal(img.shape)
    img_f32 = img.astype("float32")
    for s in [sigma, 0]:
        denoised = restoration.denoise_nl_means(
            img, 7, 5, 0.2, fast_mode=fast_mode, channel_axis=None, sigma=s
        )
        # make sure noise is reduced
        assert img.std() > denoised.std()

        denoised_f32 = restoration.denoise_nl_means(
            img_f32, 7, 5, 0.2, fast_mode=fast_mode, channel_axis=None, sigma=s
        )
        # make sure noise is reduced
        assert img.std() > denoised_f32.std()

        # Check single precision result
        assert cp.allclose(denoised_f32, denoised, atol=1e-2)


@pytest.mark.parametrize("fast_mode", [False, True])
@pytest.mark.parametrize("n_channels", [2, 3, 6])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_denoise_nl_means_2d_multichannel(fast_mode, n_channels, dtype):
    # reduce image size because nl means is slow
    img = cp.copy(astro[:50, :50])
    img = cp.concatenate((img,) * 2, axis=-1)  # 6 channels
    img = img.astype(dtype)

    # add some random noise
    sigma = 0.1
    imgn = img + sigma * cp.random.standard_normal(img.shape)
    imgn = cp.clip(imgn, 0, 1)
    imgn = imgn.astype(dtype)

    for s in [sigma, 0]:
        psnr_noisy = peak_signal_noise_ratio(
            img[..., :n_channels], imgn[..., :n_channels]
        )
        denoised = restoration.denoise_nl_means(
            imgn[..., :n_channels],
            3,
            5,
            h=0.75 * sigma,
            fast_mode=fast_mode,
            channel_axis=-1,
            sigma=s,
        )
        psnr_denoised = peak_signal_noise_ratio(
            denoised[..., :n_channels], img[..., :n_channels]
        )

        # make sure noise is reduced
        assert psnr_denoised > psnr_noisy


@pytest.mark.parametrize("fast_mode", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_denoise_nl_means_3d(fast_mode, dtype):
    img = cp.zeros((12, 12, 8), dtype=dtype)
    img[5:-5, 5:-5, 2:-2] = 1.0
    sigma = 0.3
    imgn = img + sigma * cp.random.standard_normal(img.shape)
    imgn = imgn.astype(dtype)
    psnr_noisy = peak_signal_noise_ratio(img, imgn)
    for s in [sigma, 0]:
        denoised = restoration.denoise_nl_means(
            imgn,
            3,
            4,
            h=0.75 * sigma,
            fast_mode=fast_mode,
            channel_axis=None,
            sigma=s,
        )
        # make sure noise is reduced
        assert peak_signal_noise_ratio(img, denoised) > psnr_noisy


@pytest.mark.parametrize("fast_mode", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("channel_axis", [0, -1])
def test_denoise_nl_means_multichannel(fast_mode, dtype, channel_axis):
    # for true 3D data, 3D denoising is better than denoising as 2D+channels

    # synthetic 3d volume
    img = cp.asarray(data.binary_blobs(length=32, n_dim=3, rng=5))
    img = img[:, :24, :16].astype(dtype, copy=False)

    sigma = 0.2
    rng = cp.random.default_rng(5)
    imgn = img + sigma * rng.standard_normal(img.shape)
    imgn = imgn.astype(dtype)

    # test 3D denoising (channel_axis = None)
    denoised_ok_multichannel = restoration.denoise_nl_means(
        imgn.copy(),
        3,
        2,
        h=0.6 * sigma,
        sigma=sigma,
        fast_mode=fast_mode,
        channel_axis=None,
    )

    # set a channel axis: one dimension is (incorrectly) considered "channels"
    imgn = cp.moveaxis(imgn, -1, channel_axis)
    denoised_wrong_multichannel = restoration.denoise_nl_means(
        imgn.copy(),
        3,
        2,
        h=0.6 * sigma,
        sigma=sigma,
        fast_mode=fast_mode,
        channel_axis=channel_axis,
    )
    denoised_wrong_multichannel = cp.moveaxis(
        denoised_wrong_multichannel, channel_axis, -1
    )

    img = img.astype(denoised_wrong_multichannel.dtype)
    psnr_wrong = peak_signal_noise_ratio(img, denoised_wrong_multichannel)
    psnr_ok = peak_signal_noise_ratio(img, denoised_ok_multichannel)
    assert psnr_ok > psnr_wrong


def test_denoise_nl_means_4d():
    rng = cp.random.default_rng(5)
    img = cp.zeros((10, 10, 8, 5))
    img[2:-2, 2:-2, 2:-2, :2] = 0.5
    img[2:-2, 2:-2, 2:-2, 2:] = 1.0
    sigma = 0.3
    imgn = img + sigma * rng.standard_normal(img.shape)

    nlmeans_kwargs = dict(
        patch_size=3,
        patch_distance=2,
        h=0.3 * sigma,
        sigma=sigma,
        fast_mode=True,
    )

    psnr_noisy = peak_signal_noise_ratio(img, imgn, data_range=1.0)

    # denoise by looping over 3D slices
    denoised_3d = cp.zeros_like(imgn)
    for ch in range(img.shape[-1]):
        denoised_3d[..., ch] = restoration.denoise_nl_means(
            imgn[..., ch], channel_axis=None, **nlmeans_kwargs
        )
    psnr_3d = peak_signal_noise_ratio(img, denoised_3d, data_range=1.0)
    assert psnr_3d > psnr_noisy

    # denoise as 4D
    denoised_4d = restoration.denoise_nl_means(
        imgn, channel_axis=None, **nlmeans_kwargs
    )
    psnr_4d = peak_signal_noise_ratio(img, denoised_4d, data_range=1.0)
    assert psnr_4d > psnr_3d

    # denoise as 3D + channels instead
    denoised_3dmc = restoration.denoise_nl_means(
        imgn, channel_axis=-1, **nlmeans_kwargs
    )
    psnr_3dmc = peak_signal_noise_ratio(img, denoised_3dmc, data_range=1.0)
    assert psnr_3dmc > psnr_3d


def test_denoise_nl_means_4d_multichannel():
    img = cp.zeros((8, 8, 8, 4, 4))
    img[2:-2, 2:-2, 2:-2, 1:-1, :] = 1.0
    sigma = 0.3
    imgn = img + sigma * cp.random.standard_normal(img.shape)

    psnr_noisy = peak_signal_noise_ratio(img, imgn, data_range=1.0)

    denoised_4dmc = restoration.denoise_nl_means(
        imgn,
        3,
        3,
        h=0.35 * sigma,
        fast_mode=True,
        channel_axis=-1,
        sigma=sigma,
    )
    psnr_4dmc = peak_signal_noise_ratio(img, denoised_4dmc, data_range=1.0)
    assert psnr_4dmc > psnr_noisy


def test_denoise_nl_means_wrong_dimension():
    # 1D not implemented
    img = cp.zeros((5,))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None)

    img = cp.zeros((5, 3))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1)

    # 3D + channels only for fast mode
    img = cp.zeros((5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1, fast_mode=False)

    # 4D only for fast mode
    img = cp.zeros((5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None, fast_mode=False)

    # 4D + channels only for fast mode
    img = cp.zeros((5, 5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1, fast_mode=False)

    # 5D not implemented
    img = cp.zeros((5, 5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None)


@pytest.mark.parametrize("fast_mode", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_no_denoising_for_small_h(fast_mode, dtype):
    img = cp.zeros((40, 40))
    img[10:-10, 10:-10] = 1.0
    img += 0.3 * cp.random.standard_normal(img.shape)
    img = img.astype(dtype)
    # very small h should result in no averaging with other patches
    denoised = restoration.denoise_nl_means(
        img, 7, 5, 0.01, fast_mode=fast_mode, channel_axis=None
    )
    assert cp.allclose(denoised, img)
    denoised = restoration.denoise_nl_means(
        img, 7, 5, 0.01, fast_mode=fast_mode, channel_axis=None
    )
    assert cp.allclose(denoised, img)


@pytest.mark.parametrize("fast_mode", [False, True])
def test_denoise_nl_means_2d_dtype(fast_mode):
    img = cp.zeros((40, 40), dtype=int)
    img_f32 = img.astype("float32")
    img_f64 = img.astype("float64")

    assert (
        restoration.denoise_nl_means(img, fast_mode=fast_mode).dtype
        == "float64"
    )

    assert (
        restoration.denoise_nl_means(img_f32, fast_mode=fast_mode).dtype
        == img_f32.dtype
    )

    assert (
        restoration.denoise_nl_means(img_f64, fast_mode=fast_mode).dtype
        == img_f64.dtype
    )


@pytest.mark.parametrize("fast_mode", [False, True])
def test_denoise_nl_means_3d_dtype(fast_mode):
    img = cp.zeros((12, 12, 8), dtype=int)
    img_f32 = img.astype("float32")
    img_f64 = img.astype("float64")

    assert (
        restoration.denoise_nl_means(
            img, patch_distance=2, fast_mode=fast_mode
        ).dtype
        == "float64"
    )

    assert (
        restoration.denoise_nl_means(
            img_f32, patch_distance=2, fast_mode=fast_mode
        ).dtype
        == img_f32.dtype
    )

    assert (
        restoration.denoise_nl_means(
            img_f64, patch_distance=2, fast_mode=fast_mode
        ).dtype
        == img_f64.dtype
    )
