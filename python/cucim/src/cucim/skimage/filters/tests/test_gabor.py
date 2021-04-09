import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

from cucim.skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel


def test_gabor_kernel_size():
    sigma_x = 5
    sigma_y = 10
    # Sizes cut off at +/- three sigma + 1 for the center
    size_x = sigma_x * 6 + 1
    size_y = sigma_y * 6 + 1

    kernel = gabor_kernel(0, theta=0, sigma_x=sigma_x, sigma_y=sigma_y)
    assert kernel.shape == (size_y, size_x)

    kernel = gabor_kernel(0, theta=np.pi / 2, sigma_x=sigma_x, sigma_y=sigma_y)
    assert kernel.shape == (size_x, size_y)


def test_gabor_kernel_bandwidth():
    kernel = gabor_kernel(1, bandwidth=1)
    assert kernel.shape == (5, 5)

    kernel = gabor_kernel(1, bandwidth=0.5)
    assert kernel.shape == (9, 9)

    kernel = gabor_kernel(0.5, bandwidth=1)
    assert kernel.shape == (9, 9)


def test_sigma_prefactor():
    assert_almost_equal(_sigma_prefactor(1), 0.56, 2)
    assert_almost_equal(_sigma_prefactor(0.5), 1.09, 2)


def test_gabor_kernel_sum():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                kernel = gabor_kernel(frequency + 0.1, theta=0,
                                      sigma_x=sigma_x, sigma_y=sigma_y)
                # make sure gaussian distribution is covered nearly 100%
                assert_almost_equal(float(cp.abs(kernel).sum()), 1, 2)


def test_gabor_kernel_theta():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                for theta in range(0, 10, 2):
                    kernel0 = gabor_kernel(frequency + 0.1, theta=theta,
                                           sigma_x=sigma_x, sigma_y=sigma_y)
                    kernel180 = gabor_kernel(frequency, theta=theta + np.pi,
                                             sigma_x=sigma_x, sigma_y=sigma_y)

                    assert_array_almost_equal(cp.abs(kernel0),
                                              cp.abs(kernel180))


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_gabor(dtype):
    Y, X = cp.mgrid[:40, :40]
    frequencies = (0.1, 0.3)
    wave_images = [cp.sin(2 * np.pi * X * f) for f in frequencies]

    wave_images = [w.astype(dtype, copy=False) for w in wave_images]

    def match_score(image, frequency):
        gabor_responses = gabor(image, frequency)
        assert all(r.dtype == dtype for r in gabor_responses)
        return float(cp.mean(cp.hypot(*gabor_responses)))

    # Gabor scores: diagonals are frequency-matched, off-diagonals are not.
    responses = np.array(
        [[match_score(image, f) for f in frequencies] for image in wave_images]
    )
    assert responses[0, 0] > responses[0, 1]
    assert responses[1, 1] > responses[0, 1]
    assert responses[0, 0] > responses[1, 0]
    assert responses[1, 1] > responses[1, 0]
