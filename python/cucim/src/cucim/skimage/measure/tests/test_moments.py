import itertools

import cupy as cp
import numpy as np
import pytest
from cupy.testing import (assert_allclose, assert_array_almost_equal,
                          assert_array_equal)
from cupyx.scipy import ndimage as ndi
from numpy.testing import assert_almost_equal
from skimage import draw

from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.measure import (centroid, inertia_tensor,
                                   inertia_tensor_eigvals, moments,
                                   moments_central, moments_coords,
                                   moments_coords_central, moments_hu,
                                   moments_normalized)


def compare_moments(m1, m2, thresh=1e-8):
    """Compare two moments arrays.

    Compares only values in the upper-left triangle of m1, m2 since
    values below the diagonal exceed the specified order and are not computed
    when the analytical computation is used.

    Also, there the first-order central moments will be exactly zero with the
    analytical calculation, but will not be zero due to limited floating point
    precision when using a numerical computation. Here we just specify the
    tolerance as a fraction of the maximum absolute value in the moments array.
    """
    m1 = cp.asnumpy(m1)
    m2 = cp.asnumpy(m2)

    # make sure location of any NaN values match and then ignore the NaN values
    # in the subsequent comparisons
    nan_idx1 = np.where(np.isnan(m1.ravel()))[0]
    nan_idx2 = np.where(np.isnan(m2.ravel()))[0]
    assert len(nan_idx1) == len(nan_idx2)
    assert np.all(nan_idx1 == nan_idx2)
    m1[np.isnan(m1)] = 0
    m2[np.isnan(m2)] = 0

    max_val = np.abs(m1[m1 != 0]).max()
    for orders in itertools.product(*((range(m1.shape[0]),) * m1.ndim)):
        if sum(orders) > m1.shape[0] - 1:
            m1[orders] = 0
            m2[orders] = 0
            continue
        abs_diff = abs(m1[orders] - m2[orders])
        rel_diff = abs_diff / max_val
        assert rel_diff < thresh


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
@pytest.mark.parametrize('anisotropic', [False, True, None])
def test_moments(anisotropic, dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    if anisotropic:
        spacing = (1.4, 2)
    else:
        spacing = (1, 1)
    if anisotropic is None:
        m = moments(image)
    else:
        m = moments(image, spacing=spacing)
    assert m.dtype == dtype
    assert_array_equal(m[0, 0], 3)
    decimal = 5 if dtype == np.float32 else 12
    assert_almost_equal(m[1, 0] / m[0, 0], 14.5 * spacing[0], decimal=decimal)
    assert_almost_equal(m[0, 1] / m[0, 0], 14.5 * spacing[1], decimal=decimal)


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
@pytest.mark.parametrize('anisotropic', [False, True, None])
def test_moments_central(anisotropic, dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    if anisotropic:
        spacing = (2, 1)
    else:
        spacing = (1, 1)
    if anisotropic is None:
        mu = moments_central(image, (14.5, 14.5))
        # check for proper centroid computation
        mu_calc_centroid = moments_central(image)
    else:
        mu = moments_central(image, (14.5 * spacing[0], 14.5 * spacing[1]),
                             spacing=spacing)
        # check for proper centroid computation
        mu_calc_centroid = moments_central(image, spacing=spacing)
    assert mu.dtype == dtype
    thresh = 1e-6 if dtype == np.float32 else 1e-14
    compare_moments(mu, mu_calc_centroid, thresh=thresh)

    # shift image by dx=2, dy=2
    image2 = cp.zeros((20, 20), dtype=dtype)
    image2[16, 16] = 1
    image2[17, 17] = 1
    image2[16, 17] = 0.5
    image2[17, 16] = 0.5
    if anisotropic is None:
        mu2 = moments_central(image2, (14.5 + 2, 14.5 + 2))
    else:
        mu2 = moments_central(
            image2,
            ((14.5 + 2) * spacing[0], (14.5 + 2) * spacing[1]),
            spacing=spacing
        )
    assert mu2.dtype == dtype
    # central moments must be translation invariant
    compare_moments(mu, mu2, thresh=thresh)


def test_moments_coords():
    image = cp.zeros((20, 20), dtype=cp.float64)
    image[13:17, 13:17] = 1
    mu_image = moments(image)

    coords = cp.array([[r, c] for r in range(13, 17)
                       for c in range(13, 17)], dtype=cp.float64)
    mu_coords = moments_coords(coords)
    assert_array_almost_equal(mu_coords, mu_image)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_moments_coords_dtype(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[13:17, 13:17] = 1

    expected_dtype = _supported_float_type(dtype)
    mu_image = moments(image)
    assert mu_image.dtype == expected_dtype

    coords = cp.asarray(
        np.array([[r, c] for r in range(13, 17)
                  for c in range(13, 17)], dtype=dtype)
    )
    mu_coords = moments_coords(coords)
    assert mu_coords.dtype == expected_dtype

    assert_array_almost_equal(mu_coords, mu_image)


def test_moments_central_coords():
    image = cp.zeros((20, 20), dtype=float)
    image[13:17, 13:17] = 1
    mu_image = moments_central(image, (14.5, 14.5))

    coords = cp.asarray(
        np.array(
            [[r, c] for r in range(13, 17) for c in range(13, 17)],
            dtype=float,
        )
    )
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    assert_array_almost_equal(mu_coords, mu_image)

    # ensure that center is being calculated normally
    mu_coords_calc_centroid = moments_coords_central(coords)
    assert_array_almost_equal(mu_coords_calc_centroid, mu_coords)

    # shift image by dx=3 dy=3
    image = cp.zeros((20, 20), dtype=float)
    image[16:20, 16:20] = 1
    mu_image = moments_central(image, (14.5, 14.5))

    coords = cp.asarray(
        np.array([[r, c] for r in range(16, 20)
                  for c in range(16, 20)], dtype=float)
    )
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    decimal = 6
    assert_array_almost_equal(mu_coords, mu_image, decimal=decimal)


def test_moments_normalized():
    image = cp.zeros((20, 20), dtype=float)
    image[13:17, 13:17] = 1
    mu = moments_central(image, (14.5, 14.5))
    nu = moments_normalized(mu)
    # shift image by dx=-2, dy=-2 and scale non-zero extent by 0.5
    image2 = cp.zeros((20, 20), dtype=float)
    # scale amplitude by 0.7
    image2[11:13, 11:13] = 0.7
    mu2 = moments_central(image2, (11.5, 11.5))
    nu2 = moments_normalized(mu2)
    # central moments must be translation and scale invariant
    assert_array_almost_equal(nu, nu2, decimal=1)


@pytest.mark.parametrize('anisotropic', [False, True])
def test_moments_normalized_spacing(anisotropic):
    image = cp.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1

    if not anisotropic:
        spacing1 = (1, 1)
        spacing2 = (3, 3)
    else:
        spacing1 = (1, 2)
        spacing2 = (2, 4)

    mu = moments_central(image, spacing=spacing1)
    nu = moments_normalized(mu, spacing=spacing1)

    mu2 = moments_central(image, spacing=spacing2)
    nu2 = moments_normalized(mu2, spacing=spacing2)

    # result should be invariant to absolute scale of spacing
    compare_moments(nu, nu2)


def test_moments_normalized_3d():
    image = cp.asarray(draw.ellipsoid(1, 1, 10))
    mu_image = moments_central(image)
    nu = moments_normalized(mu_image)
    assert nu[0, 0, 2] > nu[0, 2, 0]
    assert_almost_equal(nu[0, 2, 0], nu[2, 0, 0])

    coords = cp.where(image)
    mu_coords = moments_coords_central(coords)
    assert_array_almost_equal(mu_coords, mu_image)


@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float32, np.float64])
@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize('ndim', [2, 3, 4])
def test_analytical_moments_calculation(dtype, order, ndim):
    if ndim == 2:
        shape = (256, 256)
    elif ndim == 3:
        shape = (64, 64, 64)
    else:
        shape = (16, ) * ndim
    rng = np.random.default_rng(1234)
    if np.dtype(dtype).kind in 'iu':
        x = rng.integers(0, 256, shape, dtype=dtype)
    else:
        x = rng.standard_normal(shape, dtype=dtype)
    x = cp.asarray(x)
    # setting center=None will use the analytical expressions
    m1 = moments_central(x, center=None, order=order)
    # providing explicit centroid will bypass the analytical code path
    m2 = moments_central(x, center=centroid(x), order=order)

    # ensure numeric and analytical central moments are close
    thresh = 5e-4 if _supported_float_type(x.dtype) == np.float32 else 1e-11
    compare_moments(m1, m2, thresh=thresh)


def test_moments_normalized_invalid():
    with pytest.raises(ValueError):
        moments_normalized(cp.zeros((3, 3)), 3)
    with pytest.raises(ValueError):
        moments_normalized(cp.zeros((3, 3)), 4)


def test_moments_hu():
    image = cp.zeros((20, 20), dtype=float)
    image[13:15, 13:17] = 1
    mu = moments_central(image, (13.5, 14.5))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    # shift image by dx=2, dy=3, scale by 0.5 and rotate by 90deg
    image2 = cp.zeros((20, 20), dtype=float)
    image2[11, 11:13] = 1
    image2 = image2.T
    mu2 = moments_central(image2, (11.5, 11))
    nu2 = moments_normalized(mu2)
    hu2 = moments_hu(nu2)
    # central moments must be translation and scale invariant
    assert_array_almost_equal(hu, hu2, decimal=1)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_moments_dtype(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[13:15, 13:17] = 1

    expected_dtype = _supported_float_type(dtype)
    mu = moments_central(image, (13.5, 14.5))
    assert mu.dtype == expected_dtype

    nu = moments_normalized(mu)
    assert nu.dtype == expected_dtype

    hu = moments_hu(nu)
    assert hu.dtype == expected_dtype


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_centroid(dtype):
    image = cp.zeros((20, 20), dtype=dtype)
    image[14, 14:16] = 1
    image[15, 14:16] = 1 / 3
    image_centroid = centroid(image)
    if dtype == cp.float16:
        rtol = 1e-3
    elif dtype == cp.float32:
        rtol = 1e-5
    else:
        rtol = 1e-7
    assert_allclose(image_centroid, (14.25, 14.5), rtol=rtol)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_inertia_tensor_2d(dtype):
    image = cp.zeros((40, 40), dtype=dtype)
    image[15:25, 5:35] = 1  # big horizontal rectangle (aligned with axis 1)
    expected_dtype = _supported_float_type(image.dtype)

    T = inertia_tensor(image)
    assert T.dtype == expected_dtype
    assert T[0, 0] > T[1, 1]
    cp.testing.assert_allclose(T[0, 1], 0)
    v0, v1 = inertia_tensor_eigvals(image, T=T)
    assert v0.dtype == expected_dtype
    assert v1.dtype == expected_dtype
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
