import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose, assert_array_equal, assert_array_less
from skimage.data import camera, retina

from cucim.skimage import img_as_float, img_as_float64
from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.color import rgb2gray
from cucim.skimage.filters import frangi, hessian, meijering, sato
from cucim.skimage.util import crop, invert


def test_2d_null_matrix():

    a_black = cp.zeros((3, 3)).astype(cp.uint8)
    a_white = invert(a_black)

    zeros = cp.zeros((3, 3))
    ones = cp.ones((3, 3))

    assert_array_equal(meijering(a_black, black_ridges=True), zeros)
    assert_array_equal(meijering(a_white, black_ridges=False), zeros)

    assert_array_equal(sato(a_black, black_ridges=True, mode='reflect'),
                       zeros)
    assert_array_equal(sato(a_white, black_ridges=False, mode='reflect'),
                       zeros)
    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_array_equal(hessian(a_black, black_ridges=False, mode='reflect'),
                       ones)
    assert_array_equal(hessian(a_white, black_ridges=True, mode='reflect'),
                       ones)


def test_3d_null_matrix():

    # Note: last axis intentionally not size 3 to avoid 2D+RGB autodetection
    #       warning from an internal call to `skimage.filters.gaussian`.
    a_black = cp.zeros((3, 3, 5)).astype(cp.uint8)
    a_white = invert(a_black)

    zeros = cp.zeros((3, 3, 5))
    ones = cp.ones((3, 3, 5))

    assert_allclose(meijering(a_black, black_ridges=True), zeros, atol=1e-1)
    assert_allclose(meijering(a_white, black_ridges=False), zeros, atol=1e-1)

    assert_array_equal(sato(a_black, black_ridges=True, mode='reflect'),
                       zeros)
    assert_array_equal(sato(a_white, black_ridges=False, mode='reflect'),
                       zeros)

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_array_equal(hessian(a_black, black_ridges=False, mode='reflect'),
                       ones)
    assert_array_equal(hessian(a_white, black_ridges=True, mode='reflect'),
                       ones)


def test_2d_energy_decrease():

    a_black = cp.zeros((5, 5)).astype(np.uint8)
    a_black[2, 2] = 255
    a_white = invert(a_black)

    assert_array_less(meijering(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(meijering(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(sato(a_black, black_ridges=True, mode='reflect').std(),
                      a_black.std())
    assert_array_less(sato(a_white, black_ridges=False, mode='reflect').std(),
                      a_white.std())

    assert_array_less(frangi(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(frangi(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(hessian(a_black, black_ridges=True,
                              mode='reflect').std(), a_black.std())
    assert_array_less(hessian(a_white, black_ridges=False,
                              mode='reflect').std(), a_white.std())


def test_3d_energy_decrease():

    a_black = cp.zeros((5, 5, 5)).astype(np.uint8)
    a_black[2, 2, 2] = 255
    a_white = invert(a_black)

    assert_array_less(meijering(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(meijering(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(sato(a_black, black_ridges=True, mode='reflect').std(),
                      a_black.std())
    assert_array_less(sato(a_white, black_ridges=False, mode='reflect').std(),
                      a_white.std())

    assert_array_less(frangi(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(frangi(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(hessian(a_black, black_ridges=True,
                              mode='reflect').std(), a_black.std())
    assert_array_less(hessian(a_white, black_ridges=False,
                              mode='reflect').std(), a_white.std())


def test_2d_linearity():

    a_black = cp.ones((3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    assert_allclose(meijering(1 * a_black, black_ridges=True),
                    meijering(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(meijering(1 * a_white, black_ridges=False),
                    meijering(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(sato(1 * a_black, black_ridges=True, mode='reflect'),
                    sato(10 * a_black, black_ridges=True, mode='reflect'),
                    atol=1e-3)
    assert_allclose(sato(1 * a_white, black_ridges=False, mode='reflect'),
                    sato(10 * a_white, black_ridges=False, mode='reflect'),
                    atol=1e-3)

    assert_allclose(frangi(1 * a_black, black_ridges=True),
                    frangi(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(frangi(1 * a_white, black_ridges=False),
                    frangi(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(hessian(1 * a_black, black_ridges=True, mode='reflect'),
                    hessian(10 * a_black, black_ridges=True, mode='reflect'),
                    atol=1e-3)
    assert_allclose(hessian(1 * a_white, black_ridges=False, mode='reflect'),
                    hessian(10 * a_white, black_ridges=False, mode='reflect'),
                    atol=1e-3)


def test_3d_linearity():

    # Note: last axis intentionally not size 3 to avoid 2D+RGB autodetection
    #       warning from an internal call to `skimage.filters.gaussian`.
    a_black = cp.ones((3, 3, 5)).astype(np.uint8)
    a_white = invert(a_black)

    assert_allclose(meijering(1 * a_black, black_ridges=True),
                    meijering(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(meijering(1 * a_white, black_ridges=False),
                    meijering(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(sato(1 * a_black, black_ridges=True, mode='reflect'),
                    sato(10 * a_black, black_ridges=True, mode='reflect'),
                    atol=1e-3)
    assert_allclose(sato(1 * a_white, black_ridges=False, mode='reflect'),
                    sato(10 * a_white, black_ridges=False, mode='reflect'),
                    atol=1e-3)

    assert_allclose(frangi(1 * a_black, black_ridges=True),
                    frangi(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(frangi(1 * a_white, black_ridges=False),
                    frangi(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(hessian(1 * a_black, black_ridges=True, mode='reflect'),
                    hessian(10 * a_black, black_ridges=True, mode='reflect'),
                    atol=1e-3)
    assert_allclose(hessian(1 * a_white, black_ridges=False, mode='reflect'),
                    hessian(10 * a_white, black_ridges=False, mode='reflect'),
                    atol=1e-3)


@pytest.mark.parametrize('dtype', ['float64', 'uint8'])
def test_2d_cropped_camera_image(dtype):
    a_black = crop(cp.array(camera()), ((200, 212), (100, 312)))
    assert a_black.dtype == cp.uint8
    if dtype == 'float64':
        a_black = img_as_float64(a_black)
    a_white = invert(a_black)

    ones = cp.ones((100, 100))

    tol = 1e-7 if dtype == 'float64' else 1e-5

    assert_allclose(meijering(a_black, black_ridges=True),
                    meijering(a_white, black_ridges=False), atol=tol, rtol=tol)

    assert_allclose(sato(a_black, black_ridges=True, mode='reflect'),
                    sato(a_white, black_ridges=False, mode='reflect'),
                    atol=tol, rtol=tol)

    assert_allclose(frangi(a_black, black_ridges=True),
                    frangi(a_white, black_ridges=False), atol=tol, rtol=tol)

    assert_allclose(hessian(a_black, black_ridges=True, mode='reflect'),
                    ones, atol=1 - 1e-7)
    assert_allclose(hessian(a_white, black_ridges=False, mode='reflect'),
                    ones, atol=1 - 1e-7)


@pytest.mark.parametrize('func', [meijering, sato, frangi, hessian])
@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_ridge_output_dtype(func, dtype):
    img = img_as_float(cp.array(camera()).astype(dtype, copy=False))
    assert func(img).dtype == _supported_float_type(img.dtype)


@pytest.mark.parametrize('dtype', ['float64', 'uint8'])
@pytest.mark.parametrize('uniform_stack', [False, True])
def test_3d_cropped_camera_image(dtype, uniform_stack):

    a_black = crop(cp.asarray(camera()), ((200, 212), (100, 312)))
    assert a_black.dtype == cp.uint8
    if dtype == 'float64':
        a_black = img_as_float64(a_black)
    if uniform_stack:
        # Hessian along last axis will be 0 due to identical image content
        a_black = cp.dstack([a_black, a_black, a_black, a_black, a_black])
    else:
        # stack using shift to give a non-zero Hessian on the last axis
        a_black = cp.stack(
            [cp.roll(a_black, shift=n, axis=0) for n in range(5)],
            axis=-1
        )
    tol = 1e-10 if dtype == 'float64' else 4e-3
    a_white = invert(a_black)

    ones = cp.ones(a_black.shape)

    assert_allclose(meijering(a_black, black_ridges=True),
                    meijering(a_white, black_ridges=False), atol=tol, rtol=tol)

    assert_allclose(sato(a_black, black_ridges=True, mode='reflect'),
                    sato(a_white, black_ridges=False, mode='reflect'),
                    atol=tol, rtol=tol)

    assert_allclose(frangi(a_black, black_ridges=True),
                    frangi(a_white, black_ridges=False), atol=tol, rtol=tol)

    assert_allclose(hessian(a_black, black_ridges=True, mode='reflect'),
                    ones, atol=1 - 1e-7)
    assert_allclose(hessian(a_white, black_ridges=False, mode='reflect'),
                    ones, atol=1 - 1e-7)


@pytest.mark.parametrize('func, tol', [(frangi, 1e-2),
                                       (meijering, 1e-2),
                                       (sato, 2e-3),
                                       (hessian, 1e-2)])
def test_border_management(func, tol):
    img = rgb2gray(cp.array(retina()[300:500, 700:900]))
    out = func(img, sigmas=[1], mode='mirror')

    full_std = out.std()
    full_mean = out.mean()
    inside_std = out[4:-4, 4:-4].std()
    inside_mean = out[4:-4, 4:-4].mean()
    border_std = cp.stack([out[:4, :], out[-4:, :],
                           out[:, :4].T, out[:, -4:].T]).std()
    border_mean = cp.stack([out[:4, :], out[-4:, :],
                            out[:, :4].T, out[:, -4:].T]).mean()

    assert abs(full_std - inside_std) < tol
    assert abs(full_std - border_std) < tol
    assert abs(inside_std - border_std) < tol
    assert abs(full_mean - inside_mean) < tol
    assert abs(full_mean - border_mean) < 8 * tol
    assert abs(inside_mean - border_mean) < 8 * tol
