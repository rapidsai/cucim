import cupy as cp
import numpy as np
import pytest
from cupy.testing import (assert_allclose, assert_array_almost_equal,
                          assert_array_equal)
from cupyx.scipy import ndimage as ndi
from numpy.testing import assert_almost_equal
from skimage import draw

from cucim.skimage.measure import (centroid, inertia_tensor,
                                   inertia_tensor_eigvals, moments,
                                   moments_central, moments_coords,
                                   moments_coords_central, moments_normalized)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_moments(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    m = moments(image)
    assert m.dtype == dtype
    assert_array_equal(m[0, 0], 3)
    assert_almost_equal(m[1, 0] / m[0, 0], 14.5)
    assert_almost_equal(m[0, 1] / m[0, 0], 14.5)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_moments_central(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    mu = moments_central(image, (14.5, 14.5))
    assert mu.dtype == dtype

    # check for proper centroid computation
    mu_calc_centroid = moments_central(image)
    assert_array_equal(mu, mu_calc_centroid)

    # shift image by dx=2, dy=2
    image2 = cp.zeros((20, 20), dtype=dtype)
    image2[16, 16] = 1
    image2[17, 17] = 1
    image2[16, 17] = 0.5
    image2[17, 16] = 0.5
    mu2 = moments_central(image2, (14.5 + 2, 14.5 + 2))
    assert mu2.dtype == dtype
    # central moments must be translation invariant
    assert_array_equal(mu, mu2)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_moments_coords(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[13:17, 13:17] = 1
    mu_image = moments(image)
    assert mu_image.dtype == dtype

    coords = cp.asarray(
        np.array([[r, c] for r in range(13, 17)
                  for c in range(13, 17)], dtype=dtype)
    )
    mu_coords = moments_coords(coords)
    assert mu_coords.dtype == dtype
    assert_array_almost_equal(mu_coords, mu_image)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_moments_central_coords(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[13:17, 13:17] = 1
    mu_image = moments_central(image, (14.5, 14.5))
    assert mu_image.dtype == dtype

    coords = cp.asarray(
        np.array(
            [[r, c] for r in range(13, 17) for c in range(13, 17)],
            dtype=dtype,
        )
    )
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    assert mu_coords.dtype == dtype
    assert_array_almost_equal(mu_coords, mu_image)

    # ensure that center is being calculated normally
    mu_coords_calc_centroid = moments_coords_central(coords)
    assert_array_almost_equal(mu_coords_calc_centroid, mu_coords)

    # shift image by dx=3 dy=3
    image = cp.zeros((20, 20), dtype=dtype)
    image[16:20, 16:20] = 1
    mu_image = moments_central(image, (14.5, 14.5))
    assert mu_image.dtype == dtype

    coords = cp.asarray(
        np.array([[r, c] for r in range(16, 20)
                  for c in range(16, 20)], dtype=dtype)
    )
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    assert mu_coords.dtype == dtype
    decimal = 3 if dtype == np.float32 else 6
    assert_array_almost_equal(mu_coords, mu_image, decimal=decimal)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_moments_normalized(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[13:17, 13:17] = 1
    mu = moments_central(image, (14.5, 14.5))
    nu = moments_normalized(mu)
    # shift image by dx=-3, dy=-3 and scale by 0.5
    image2 = cp.zeros((20, 20), dtype=dtype)
    image2[11:13, 11:13] = 1
    mu2 = moments_central(image2, (11.5, 11.5))
    nu2 = moments_normalized(mu2)
    # central moments must be translation and scale invariant
    assert_array_almost_equal(nu, nu2, decimal=1)


def test_moments_normalized_3d():
    image = cp.asarray(draw.ellipsoid(1, 1, 10))
    mu_image = moments_central(image)
    nu = moments_normalized(mu_image)
    assert nu[0, 0, 2] > nu[0, 2, 0]
    assert_almost_equal(nu[0, 2, 0], nu[2, 0, 0])

    coords = cp.where(image)
    mu_coords = moments_coords_central(coords)
    assert_array_almost_equal(mu_coords, mu_image)


def test_moments_normalized_invalid():
    with pytest.raises(ValueError):
        moments_normalized(cp.zeros((3, 3)), 3)
    with pytest.raises(ValueError):
        moments_normalized(cp.zeros((3, 3)), 4)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_moments_hu(dtype):
    moments_hu = pytest.importorskip("skimage.measure.moments_hu")

    image = cp.zeros((20, 20), dtype=dtype)
    image[13:15, 13:17] = 1
    mu = moments_central(image, (13.5, 14.5))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    assert hu.dtype == image.dtype
    # shift image by dx=2, dy=3, scale by 0.5 and rotate by 90deg
    image2 = cp.zeros((20, 20), dtype=dtype)
    image2[11, 11:13] = 1
    image2 = image2.T
    mu2 = moments_central(image2, (11.5, 11))
    nu2 = moments_normalized(mu2)
    hu2 = moments_hu(nu2)
    assert hu2.dtype == image2.dtype
    # central moments must be translation and scale invariant
    assert_array_almost_equal(hu, hu2, decimal=1)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_centroid(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[14, 14:16] = 1
    image[15, 14:16] = 1 / 3
    image_centroid = centroid(image)
    assert image_centroid.dtype == image.dtype
    assert_allclose(image_centroid, (14.25, 14.5))


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_inertia_tensor_2d(dtype):
    image = cp.zeros((40, 40), dtype=dtype)
    image[15:25, 5:35] = 1  # big horizontal rectangle (aligned with axis 1)
    T = inertia_tensor(image)
    assert T.dtype == image.dtype
    assert T[0, 0] > T[1, 1]
    cp.testing.assert_allclose(T[0, 1], 0)
    v0, v1 = inertia_tensor_eigvals(image, T=T)
    assert v0.dtype == image.dtype
    cp.testing.assert_allclose(cp.sqrt(v0 / v1), 3, rtol=0.01, atol=0.05)


def test_inertia_tensor_3d():
    image = cp.asarray(draw.ellipsoid(10, 5, 3))
    T0 = inertia_tensor(image)
    eig0, V0 = np.linalg.eig(cp.asnumpy(T0))
    # principal axis of ellipse = eigenvector of smallest eigenvalue
    v0 = cp.asarray(V0[:, np.argmin(eig0)])

    assert cp.allclose(v0, [1, 0, 0]) or cp.allclose(-v0, [1, 0, 0])

    imrot = ndi.rotate(image.astype(float), 30, axes=(0, 1), order=1)
    Tr = inertia_tensor(imrot)
    eigr, Vr = np.linalg.eig(cp.asnumpy(Tr))
    vr = cp.asarray(Vr[:, np.argmin(eigr)])

    # Check that axis has rotated by expected amount
    pi, cos, sin = np.pi, np.cos, np.sin
    # fmt: off
    R = cp.array([[cos(pi/6), -sin(pi/6), 0],   # noqa
                  [sin(pi/6),  cos(pi/6), 0],   # noqa
                  [        0,          0, 1]])  # noqa
    # fmt: on
    expected_vr = R @ v0
    assert (cp.allclose(vr, expected_vr, atol=1e-3, rtol=0.01) or
            cp.allclose(-vr, expected_vr, atol=1e-3, rtol=0.01))


def test_inertia_tensor_eigvals():
    # Floating point precision problems could make a positive
    # semidefinite matrix have an eigenvalue that is very slightly
    # negative.  Check that we have caught and fixed this problem.
    # fmt: off
    image = cp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    # fmt: on
    # mu = np.array([[3, 0, 98], [0, 14, 0], [2, 0, 98]])
    eigvals = inertia_tensor_eigvals(image=image)
    assert min(eigvals) >= 0
