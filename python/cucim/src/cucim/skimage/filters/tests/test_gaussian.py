import cupy as cp
import numpy as np
import pytest

from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.filters._gaussian import (_guess_spatial_dimensions,
                                             difference_of_gaussians, gaussian)


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


def test_multichannel():
    a = cp.zeros((5, 5, 3))
    a[1, 1] = cp.arange(1, 4)
    gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect', multichannel=True)
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert cp.allclose([a[..., i].mean() for i in range(3)],
                       [gaussian_rgb_a[..., i].mean() for i in range(3)])
    # Test multichannel = None
    with expected_warnings(["multichannel"]):
        gaussian_rgb_a = gaussian(a, sigma=1, mode="reflect")
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert cp.allclose([a[..., i].mean() for i in range(3)],
                       [gaussian_rgb_a[..., i].mean() for i in range(3)])
    # Iterable sigma
    gaussian_rgb_a = gaussian(a, sigma=[1, 2], mode='reflect',
                              multichannel=True)
    assert cp.allclose([a[..., i].mean() for i in range(3)],
                       [gaussian_rgb_a[..., i].mean() for i in range(3)])


def test_preserve_range():
    img = cp.array([[10.0, -10.0], [-4, 3]], dtype=cp.float32)
    gaussian(img, 1, preserve_range=True)


def test_4d_ok():
    img = cp.zeros((5,) * 4)
    img[2, 2, 2, 2] = 1
    res = gaussian(img, 1, mode="reflect")
    assert cp.allclose(res.sum(), 1)


def test_guess_spatial_dimensions():
    im1 = cp.zeros((5, 5))
    im2 = cp.zeros((5, 5, 5))
    im3 = cp.zeros((5, 5, 3))
    im4 = cp.zeros((5, 5, 5, 3))
    im5 = cp.zeros((5,))
    assert _guess_spatial_dimensions(im1) == 2
    assert _guess_spatial_dimensions(im2) == 3
    assert _guess_spatial_dimensions(im3) is None
    assert _guess_spatial_dimensions(im4) == 3
    with pytest.raises(ValueError):
        _guess_spatial_dimensions(im5)


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
def test_difference_of_gaussians(s, s2):
    image = cp.random.rand(10, 10)
    im1 = gaussian(image, s)
    im2 = gaussian(image, s2)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2)
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
        difference_of_gaussians(image, (1, 2, 3), multichannel=True)


def test_dog_invalid_sigma2():
    image = cp.ones((3, 3))
    with pytest.raises(ValueError):
        difference_of_gaussians(image, 3, 2)
    with pytest.raises(ValueError):
        difference_of_gaussians(image, (1, 5), (2, 4))
