import cupy as cp
import numpy as np
import pytest

from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.filters._gaussian import difference_of_gaussians, gaussian


def test_negative_sigma():
    a = cp.zeros((3, 3))
    a[1, 1] = 1.0
    with pytest.raises(ValueError):
        gaussian(a, sigma=-1.0)
    with pytest.raises(ValueError):
        gaussian(a, sigma=[-1.0, 1.0])
    with pytest.raises(ValueError):
        gaussian(a, sigma=cp.asarray([-1.0, 1.0]))


def test_null_sigma():
    a = cp.zeros((3, 3))
    a[1, 1] = 1.0
    assert cp.all(gaussian(a, 0) == a)


def test_default_sigma():
    a = cp.zeros((3, 3))
    a[1, 1] = 1.0
    assert cp.all(gaussian(a) == gaussian(a, sigma=1))


def test_energy_decrease():
    a = cp.zeros((3, 3))
    a[1, 1] = 1.0
    gaussian_a = gaussian(a, sigma=1, mode="reflect")
    assert gaussian_a.std() < a.std()


@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_multichannel(channel_axis):
    a = np.zeros((5, 5, 3))
    a[1, 1] = np.arange(1, 4)
    a = np.moveaxis(a, -1, channel_axis)
    a = cp.asarray(a)
    gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect', preserve_range=True,
                              channel_axis=channel_axis)
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    spatial_axes = tuple(
        [ax for ax in range(a.ndim) if ax != channel_axis % a.ndim]
    )
    assert cp.allclose(a.mean(axis=spatial_axes),
                       gaussian_rgb_a.mean(axis=spatial_axes))

    if channel_axis % a.ndim == 2:
        # Test legacy behavior equivalent to old (multichannel = None)
        with expected_warnings(['multichannel']):
            gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect',
                                      preserve_range=True)

        # Check that the mean value is conserved in each channel
        # (color channels are not mixed together)
        assert cp.allclose(a.mean(axis=spatial_axes),
                           gaussian_rgb_a.mean(axis=spatial_axes))
    # Iterable sigma
    gaussian_rgb_a = gaussian(a, sigma=[1, 2], mode='reflect',
                              channel_axis=channel_axis,
                              preserve_range=True)
    assert cp.allclose(a.mean(axis=spatial_axes),
                       gaussian_rgb_a.mean(axis=spatial_axes))


def test_deprecated_multichannel():
    a = np.zeros((5, 5, 3))
    a[1, 1] = np.arange(1, 4)
    a = cp.asarray(a)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect',
                                  multichannel=True)
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert cp.allclose(a.mean(axis=(0, 1)), gaussian_rgb_a.mean(axis=(0, 1)))

    # check positional multichannel argument warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        gaussian_rgb_a = gaussian(a, 1, None, 'reflect', 0, True)


def test_preserve_range():
    """Test preserve_range parameter."""
    ones = cp.ones((2, 2), dtype=np.int64)
    filtered_ones = gaussian(ones, preserve_range=False)
    assert cp.all(filtered_ones == filtered_ones[0, 0])
    assert filtered_ones[0, 0] < 1e-10

    filtered_preserved = gaussian(ones, preserve_range=True)
    cp.testing.assert_array_almost_equal(
        filtered_preserved, cp.ones_like(filtered_preserved)
    )

    img = cp.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1)


def test_1d_ok():
    """Testing Gaussian Filter for 1D array.
    With any array consisting of positive integers and only one zero - it
    should filter all values to be greater than 0.1
    """
    nums = cp.arange(7)
    filtered = gaussian(nums, preserve_range=True)
    assert cp.all(filtered > 0.1)


def test_4d_ok():
    img = cp.zeros((5,) * 4)
    img[2, 2, 2, 2] = 1
    res = gaussian(img, 1, mode="reflect", preserve_range=True)
    assert cp.allclose(res.sum(), 1)


@pytest.mark.parametrize(
    "dtype", [cp.float32, cp.float64]
)
def test_preserve_output(dtype):
    image = cp.arange(9, dtype=dtype).reshape((3, 3))
    output = cp.zeros_like(image, dtype=dtype)
    gaussian_image = gaussian(image, sigma=1, output=output,
                              preserve_range=True)
    assert gaussian_image is output


def test_output_error():
    image = cp.arange(9, dtype=cp.float32).reshape((3, 3))
    output = cp.zeros_like(image, dtype=cp.uint8)
    with pytest.raises(ValueError):
        gaussian(image, sigma=1, output=output,
                 preserve_range=True)


@pytest.mark.parametrize("s", [1, (2, 3)])
@pytest.mark.parametrize("s2", [4, (5, 6)])
@pytest.mark.parametrize("channel_axis", [None, 0, 1, -1])
def test_difference_of_gaussians(s, s2, channel_axis):
    image = np.random.rand(10, 10)
    if channel_axis is not None:
        n_channels = 5
        image = np.stack((image,) * n_channels, channel_axis)
    image = cp.asarray(image)
    im1 = gaussian(image, s, preserve_range=True, channel_axis=channel_axis)
    im2 = gaussian(image, s2, preserve_range=True, channel_axis=channel_axis)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2, channel_axis=channel_axis)
    assert cp.allclose(dog, dog2)


@pytest.mark.parametrize("s", [1, (1, 2)])
def test_auto_sigma2(s):
    image = cp.random.rand(10, 10)
    im1 = gaussian(image, s)
    s2 = 1.6 * np.array(s)
    im2 = gaussian(image, s2)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2)
    assert cp.allclose(dog, dog2)


def test_dog_invalid_sigma_dims():
    image = cp.ones((5, 5, 3))
    with pytest.raises(ValueError):
        difference_of_gaussians(image, (1, 2))
    with pytest.raises(ValueError):
        difference_of_gaussians(image, 1, (3, 4))
    with pytest.raises(ValueError):
        with expected_warnings(["`multichannel` is a deprecated argument"]):
            difference_of_gaussians(image, (1, 2, 3), multichannel=True)
    with pytest.raises(ValueError):
        difference_of_gaussians(image, (1, 2, 3), channel_axis=-1)


def test_dog_invalid_sigma2():
    image = cp.ones((3, 3))
    with pytest.raises(ValueError):
        difference_of_gaussians(image, 3, 2)
    with pytest.raises(ValueError):
        difference_of_gaussians(image, (1, 5), (2, 4))
