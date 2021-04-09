import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose, assert_array_equal, assert_array_less
from skimage.data import camera, retina

from cucim.skimage._shared.testing import expected_warnings
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

    a_black = cp.zeros((3, 3, 3)).astype(cp.uint8)
    a_white = invert(a_black)

    zeros = cp.zeros((3, 3, 3))
    ones = cp.ones((3, 3, 3))

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

    a_black = cp.ones((3, 3, 3)).astype(np.uint8)
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


def test_2d_cropped_camera_image():

    a_black = crop(cp.array(camera()), ((200, 212), (100, 312)))
    a_white = invert(a_black)

    zeros = cp.zeros((100, 100))
    ones = cp.ones((100, 100))

    assert_allclose(meijering(a_black, black_ridges=True),
                    meijering(a_white, black_ridges=False))

    assert_allclose(sato(a_black, black_ridges=True, mode='mirror'),
                    sato(a_white, black_ridges=False, mode='mirror'))

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_allclose(hessian(a_black, black_ridges=True, mode='mirror'),
                    ones, atol=1 - 1e-7)
    assert_allclose(hessian(a_white, black_ridges=False, mode='mirror'),
                    ones, atol=1 - 1e-7)


def test_3d_cropped_camera_image():

    a_black = crop(cp.asarray(camera()), ((200, 212), (100, 312)))
    a_black = cp.dstack([a_black, a_black, a_black])
    a_white = invert(a_black)

    zeros = cp.zeros((100, 100, 3))
    ones = cp.ones((100, 100, 3))

    # TODO: determine why the following allclose checks occassionally fail
    assert_allclose(meijering(a_black, black_ridges=True),
                    meijering(a_white, black_ridges=False))

    assert_allclose(sato(a_black, black_ridges=True, mode='mirror'),
                    sato(a_white, black_ridges=False, mode='mirror'))

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_allclose(hessian(a_black, black_ridges=True, mode='mirror'),
                    ones, atol=1 - 1e-7)
    assert_allclose(hessian(a_white, black_ridges=False, mode='mirror'),
                    ones, atol=1 - 1e-7)


@pytest.mark.parametrize('func, tol', [(frangi, 1e-7),
                                       (meijering, 2e-2),
                                       (sato, 1e-3),
                                       (hessian, 2e-2)])
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
    assert abs(full_mean - border_mean) < tol
    assert abs(inside_mean - border_mean) < tol


@pytest.mark.parametrize('func', [sato, hessian])
def test_border_warning(func):
    img = rgb2gray(cp.array(retina()[300:500, 700:900]))

    with expected_warnings(["implicitly used 'constant' as the border mode"]):
        func(img, sigmas=[1])


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
@pytest.mark.parametrize('func', [sato, hessian, meijering, frangi])
def test_output_dtype(func, dtype):
    img = rgb2gray(cp.array(retina()[300:500, 700:900], dtype=dtype))

    out = func(img, sigmas=[1], mode='reflect')
    assert out.dtype == dtype
