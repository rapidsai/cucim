import itertools

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose
from cupyx.scipy.ndimage import fourier_shift
from skimage.data import camera, eagle

from cucim.skimage import img_as_float
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage._shared.fft import fftmodule as fft
from cucim.skimage.data import binary_blobs
from cucim.skimage.registration._phase_cross_correlation import (
    _upsampled_dft,
    phase_cross_correlation,
)


@pytest.mark.parametrize("normalization", [None, "phase"])
def test_correlation(normalization):
    reference_image = fft.fftn(cp.array(camera()))
    shift = (-7, 12)
    shifted_image = fourier_shift(reference_image, shift)

    # pixel precision
    result, _, _ = phase_cross_correlation(
        reference_image,
        shifted_image,
        space="fourier",
        normalization=normalization,
    )
    assert_allclose(result[:2], -cp.array(shift))


@pytest.mark.parametrize("normalization", ["nonexisting"])
def test_correlation_invalid_normalization(normalization):
    reference_image = fft.fftn(cp.array(camera()))
    shift = (-7, 12)
    shifted_image = fourier_shift(reference_image, shift)

    # pixel precision
    with pytest.raises(ValueError):
        phase_cross_correlation(
            reference_image,
            shifted_image,
            space="fourier",
            normalization=normalization,
        )


@pytest.mark.parametrize("normalization", [None, "phase"])
def test_subpixel_precision(normalization):
    reference_image = fft.fftn(cp.array(camera()))
    subpixel_shift = (-2.4, 1.32)
    shifted_image = fourier_shift(reference_image, subpixel_shift)

    # subpixel precision
    result, _, _ = phase_cross_correlation(
        reference_image,
        shifted_image,
        upsample_factor=100,
        space="fourier",
        normalization=normalization,
    )
    assert_allclose(result[:2], -cp.array(subpixel_shift), atol=0.05)


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_real_input(dtype):
    reference_image = cp.array(camera()).astype(dtype, copy=False)
    subpixel_shift = (-2.4, 1.32)
    shifted_image = fourier_shift(fft.fftn(reference_image), subpixel_shift)
    shifted_image = fft.ifftn(shifted_image).real.astype(dtype, copy=False)

    # subpixel precision
    result, error, diffphase = phase_cross_correlation(
        reference_image, shifted_image, upsample_factor=100
    )
    assert isinstance(result, tuple)
    assert all(isinstance(s, float) for s in result)
    assert_allclose(result[:2], -cp.array(subpixel_shift), atol=0.05)


def test_size_one_dimension_input():
    # take a strip of the input image
    reference_image = fft.fftn(cp.array(camera())[:, 15]).reshape((-1, 1))
    subpixel_shift = (-2.4, 4)
    shifted_image = fourier_shift(reference_image, subpixel_shift)

    # subpixel precision
    result, error, diffphase = phase_cross_correlation(
        reference_image, shifted_image, upsample_factor=20, space="fourier"
    )
    assert_allclose(result[:2], -cp.array((-2.4, 0)), atol=0.05)


def test_3d_input():
    phantom = img_as_float(binary_blobs(length=32, n_dim=3))
    reference_image = fft.fftn(phantom)
    shift = (-2.0, 1.0, 5.0)
    shifted_image = fourier_shift(reference_image, shift)

    result, error, diffphase = phase_cross_correlation(
        reference_image, shifted_image, space="fourier"
    )
    assert_allclose(result, -cp.array(shift), atol=0.05)

    # subpixel precision now available for 3-D data

    subpixel_shift = (-2.3, 1.7, 5.4)
    shifted_image = fourier_shift(reference_image, subpixel_shift)
    result, error, diffphase = phase_cross_correlation(
        reference_image, shifted_image, upsample_factor=100, space="fourier"
    )
    assert_allclose(result, -cp.array(subpixel_shift), atol=0.05)


def test_unknown_space_input():
    image = cp.ones((5, 5))
    with pytest.raises(ValueError):
        phase_cross_correlation(image, image, space="frank")


def test_wrong_input():
    # Dimensionality mismatch
    image = cp.ones((5, 5, 1))
    template = cp.ones((5, 5))
    with pytest.raises(ValueError):
        phase_cross_correlation(template, image)

    # Size mismatch
    image = cp.ones((5, 5))
    template = cp.ones((4, 4))
    with pytest.raises(ValueError):
        phase_cross_correlation(template, image)

    # NaN values in data
    image = cp.ones((5, 5))
    image[0][0] = cp.nan
    template = cp.ones((5, 5))
    with expected_warnings([r"invalid value encountered in true_divide|\A\Z"]):
        with pytest.raises(ValueError):
            phase_cross_correlation(template, image)


def test_4d_input_pixel():
    phantom = img_as_float(binary_blobs(length=32, n_dim=4))
    reference_image = fft.fftn(phantom)
    shift = (-2.0, 1.0, 5.0, -3)
    shifted_image = fourier_shift(reference_image, shift)
    result, error, diffphase = phase_cross_correlation(
        reference_image, shifted_image, space="fourier"
    )
    assert_allclose(result, -cp.array(shift), atol=0.05)


def test_4d_input_subpixel():
    phantom = img_as_float(binary_blobs(length=32, n_dim=4))
    reference_image = fft.fftn(phantom)
    subpixel_shift = (-2.3, 1.7, 5.4, -3.2)
    shifted_image = fourier_shift(reference_image, subpixel_shift)
    result, error, diffphase = phase_cross_correlation(
        reference_image, shifted_image, upsample_factor=10, space="fourier"
    )
    assert_allclose(result, -cp.array(subpixel_shift), atol=0.05)


def test_mismatch_upsampled_region_size():
    with pytest.raises(ValueError):
        _upsampled_dft(cp.ones((4, 4)), upsampled_region_size=[3, 2, 1, 4])


def test_mismatch_offsets_size():
    with pytest.raises(ValueError):
        _upsampled_dft(cp.ones((4, 4)), 3, axis_offsets=[3, 2, 1, 4])


@pytest.mark.parametrize(
    ("shift0", "shift1"),
    itertools.product((100, -100, 350, -350), (100, -100, 350, -350)),
)
@cp.testing.with_requires("scikit-image>=0.20")
def test_disambiguate_2d(shift0, shift1):
    image = cp.array(eagle()[500:, 900:])  # use a highly textured image region

    # Protect against some versions of scikit-image + imagio loading as
    # RGB instead of grayscale.
    if image.ndim == 3:
        image = image[..., 0]

    shift = (shift0, shift1)
    origin0 = []
    for s in shift:
        if s > 0:
            origin0.append(0)
        else:
            origin0.append(-s)
    origin1 = np.array(origin0) + shift
    slice0 = tuple(slice(o, o + 450) for o in origin0)
    slice1 = tuple(slice(o, o + 450) for o in origin1)
    reference = image[slice0]
    moving = image[slice1]
    computed_shift, _, _ = phase_cross_correlation(
        reference,
        moving,
        disambiguate=True,
    )
    np.testing.assert_equal(shift, computed_shift)


def test_disambiguate_zero_shift():
    """When the shift is 0, disambiguation becomes degenerate.
    Some quadrants become size 0, which prevents computation of
    cross-correlation. This test ensures that nothing bad happens in that
    scenario.
    """
    image = cp.array(camera())
    computed_shift, _, _ = phase_cross_correlation(
        image,
        image,
        disambiguate=True,
        return_error="always",
    )
    assert computed_shift == (0, 0)
