import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal, assert_array_equal
from numpy.testing import assert_equal
from skimage import data, draw

from cucim.skimage import img_as_float
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.color import rgb2gray
from cucim.skimage.feature import (corner_foerstner, corner_harris,
                                   corner_kitchen_rosenfeld, corner_peaks,
                                   corner_shi_tomasi, hessian_matrix,
                                   hessian_matrix_det, hessian_matrix_eigvals,
                                   peak_local_max, shape_index,
                                   structure_tensor,
                                   structure_tensor_eigenvalues,
                                   structure_tensor_eigvals)
from cucim.skimage.morphology import cube


@pytest.fixture
def im3d():
    r = 10
    pad = 10
    im3 = draw.ellipsoid(r, r, r)
    im3 = np.pad(im3, pad, mode='constant').astype(np.uint8)
    return cp.asarray(im3)


def test_structure_tensor():
    square = cp.zeros((5, 5))
    square[2, 2] = 1
    Arr, Arc, Acc = structure_tensor(square, sigma=0.1, order='rc')
    # fmt: off
    assert_array_equal(Acc, cp.asarray([[0, 0, 0, 0, 0],
                                        [0, 1, 0, 1, 0],
                                        [0, 4, 0, 4, 0],
                                        [0, 1, 0, 1, 0],
                                        [0, 0, 0, 0, 0]]))
    assert_array_equal(Arc, cp.asarray([[0, 0, 0, 0, 0],
                                        [0, 1, 0, -1, 0],
                                        [0, 0, 0, -0, 0],
                                        [0, -1, -0, 1, 0],
                                        [0, 0, 0, 0, 0]]))
    assert_array_equal(Arr, cp.asarray([[0, 0, 0, 0, 0],
                                        [0, 1, 4, 1, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 1, 4, 1, 0],
                                        [0, 0, 0, 0, 0]]))
    # fmt: on


def test_structure_tensor_3d():
    cube = cp.zeros((5, 5, 5))
    cube[2, 2, 2] = 1
    A_elems = structure_tensor(cube, sigma=0.1)
    assert_equal(len(A_elems), 6)
    # fmt: off
    assert_array_equal(A_elems[0][:, 1, :], cp.asarray([[0, 0, 0, 0, 0],
                                                        [0, 1, 4, 1, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 1, 4, 1, 0],
                                                        [0, 0, 0, 0, 0]]))
    assert_array_equal(A_elems[0][1], cp.asarray([[0, 0, 0, 0, 0],
                                                  [0, 1, 4, 1, 0],
                                                  [0, 4, 16, 4, 0],
                                                  [0, 1, 4, 1, 0],
                                                  [0, 0, 0, 0, 0]]))
    assert_array_equal(A_elems[3][2], cp.asarray([[0, 0, 0, 0, 0],
                                                  [0, 4, 16, 4, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 4, 16, 4, 0],
                                                  [0, 0, 0, 0, 0]]))
    # fmt: on


def test_structure_tensor_3d_rc_only():
    cube = cp.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        structure_tensor(cube, sigma=0.1, order='xy')
    A_elems_rc = structure_tensor(cube, sigma=0.1, order='rc')
    A_elems_none = structure_tensor(cube, sigma=0.1)
    for a_rc, a_none in zip(A_elems_rc, A_elems_none):
        assert_array_equal(a_rc, a_none)


def test_structure_tensor_orders():
    square = cp.zeros((5, 5))
    square[2, 2] = 1
    with expected_warnings(['the default order of the structure']):
        A_elems_default = structure_tensor(square, sigma=0.1)
    A_elems_xy = structure_tensor(square, sigma=0.1, order='xy')
    A_elems_rc = structure_tensor(square, sigma=0.1, order='rc')
    for elem_xy, elem_def in zip(A_elems_xy, A_elems_default):
        assert_array_equal(elem_xy, elem_def)
    for elem_xy, elem_rc in zip(A_elems_xy, A_elems_rc[::-1]):
        assert_array_equal(elem_xy, elem_rc)


def test_hessian_matrix():
    square = cp.zeros((5, 5))
    square[2, 2] = 4
    Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order="rc")
    # fmt: off
    assert_array_almost_equal(Hrr, cp.asarray([[0, 0,  0, 0, 0],    # noqa
                                               [0, 0,  0, 0, 0],    # noqa
                                               [2, 0, -2, 0, 2],    # noqa
                                               [0, 0,  0, 0, 0],    # noqa
                                               [0, 0,  0, 0, 0]]))  # noqa

    assert_array_almost_equal(Hrc, cp.asarray([[0,  0, 0,  0, 0],    # noqa
                                               [0,  1, 0, -1, 0],    # noqa
                                               [0,  0, 0,  0, 0],    # noqa
                                               [0, -1, 0,  1, 0],    # noqa
                                               [0,  0, 0,  0, 0]]))  # noqa

    assert_array_almost_equal(Hcc, cp.asarray([[0, 0,  2, 0, 0],    # noqa
                                               [0, 0,  0, 0, 0],    # noqa
                                               [0, 0, -2, 0, 0],    # noqa
                                               [0, 0,  0, 0, 0],    # noqa
                                               [0, 0,  2, 0, 0]]))  # noqa
    # fmt: on


def test_hessian_matrix_3d():
    cube = cp.zeros((5, 5, 5))
    cube[2, 2, 2] = 4
    Hs = hessian_matrix(cube, sigma=0.1, order='rc')
    assert len(Hs) == 6, "incorrect number of Hessian images (%i) for 3D" % len(
        Hs
    )
    # fmt: off
    assert_array_almost_equal(
        Hs[2][:, 2, :], cp.asarray([[0,  0,  0,  0,  0],    # noqa
                                    [0,  1,  0, -1,  0],    # noqa
                                    [0,  0,  0,  0,  0],    # noqa
                                    [0, -1,  0,  1,  0],    # noqa
                                    [0,  0,  0,  0,  0]]))  # noqa
    # fmt: on


def test_structure_tensor_eigenvalues():
    square = cp.zeros((5, 5))
    square[2, 2] = 1
    A_elems = structure_tensor(square, sigma=0.1, order='rc')
    l1, l2 = structure_tensor_eigenvalues(A_elems)
    assert_array_equal(l1, cp.asarray([[0, 0, 0, 0, 0],
                                       [0, 2, 4, 2, 0],
                                       [0, 4, 0, 4, 0],
                                       [0, 2, 4, 2, 0],
                                       [0, 0, 0, 0, 0]]))
    assert_array_equal(l2, cp.asarray([[0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]))


def test_structure_tensor_eigenvalues_3d():
    image = cp.pad(cube(9), 5, mode='constant') * 1000
    boundary = (cp.pad(cube(9), 5, mode='constant')
                - cp.pad(cube(7), 6, mode='constant')).astype(bool)
    A_elems = structure_tensor(image, sigma=0.1)
    e0, e1, e2 = structure_tensor_eigenvalues(A_elems)
    # e0 should detect facets
    assert np.all(e0[boundary] != 0)


def test_structure_tensor_eigvals():
    square = cp.zeros((5, 5))
    square[2, 2] = 1
    A_elems = structure_tensor(square, sigma=0.1, order='rc')
    with expected_warnings(['structure_tensor_eigvals is deprecated']):
        eigvals = structure_tensor_eigvals(*A_elems)
    eigenvalues = structure_tensor_eigenvalues(A_elems)
    for ev1, ev2 in zip(eigvals, eigenvalues):
        assert_array_equal(ev1, ev2)


def test_hessian_matrix_eigvals():
    square = cp.zeros((5, 5))
    square[2, 2] = 4
    H = hessian_matrix(square, sigma=0.1, order='rc')
    l1, l2 = hessian_matrix_eigvals(H)
    # fmt: off
    assert_array_almost_equal(l1, cp.asarray([[0, 0,  2, 0, 0],      # noqa
                                              [0, 1,  0, 1, 0],      # noqa
                                              [2, 0, -2, 0, 2],      # noqa
                                              [0, 1,  0, 1, 0],      # noqa
                                              [0, 0,  2, 0, 0]]))    # noqa
    assert_array_almost_equal(l2, cp.asarray([[0,  0,  0,  0, 0],    # noqa
                                              [0, -1,  0, -1, 0],    # noqa
                                              [0,  0, -2,  0, 0],    # noqa
                                              [0, -1,  0, -1, 0],    # noqa
                                              [0,  0,  0,  0, 0]]))  # noqa

    # fmt: on


def test_hessian_matrix_eigvals_3d(im3d):
    H = hessian_matrix(im3d)
    E = hessian_matrix_eigvals(H)
    E = cp.asnumpy(E)
    # test descending order:
    e0, e1, e2 = E
    assert np.all(e0 >= e1) and np.all(e1 >= e2)

    E0, E1, E2 = E[:, E.shape[1] // 2]  # cross section
    row_center, col_center = np.asarray(E0.shape) // 2
    circles = [
        draw.circle_perimeter(row_center, col_center, radius, shape=E0.shape)
        for radius in range(1, E0.shape[1] // 2 - 1)
    ]
    response0 = np.array([np.mean(E0[c]) for c in circles])
    response2 = np.array([np.mean(E2[c]) for c in circles])

    # eigenvalues are negative just inside the sphere, positive just outside
    assert np.argmin(response2) < np.argmax(response0)
    assert np.min(response2) < 0
    assert np.max(response0) > 0


def test_hessian_matrix_det():
    image = cp.zeros((5, 5))
    image[2, 2] = 1
    # TODO: approximate=True case not implemented
    det = hessian_matrix_det(image, 5, approximate=False)
    assert_array_almost_equal(det, 0, decimal=3)


def test_hessian_matrix_det_3d(im3d):
    D = hessian_matrix_det(im3d)
    D = cp.asnumpy(D)
    D0 = D[D.shape[0] // 2]
    row_center, col_center = np.asarray(D0.shape) // 2
    # testing in 3D is hard. We test this by showing that you get the
    # expected flat-then-low-then-high 2nd derivative response in a circle
    # around the midplane of the sphere.
    circles = [
        draw.circle_perimeter(row_center, col_center, r, shape=D0.shape)
        for r in range(1, D0.shape[1] // 2 - 1)
    ]
    response = np.array([np.mean(D0[c]) for c in circles])
    lowest = np.argmin(response)
    highest = np.argmax(response)
    assert lowest < highest
    assert response[lowest] < 0
    assert response[highest] > 0


def test_shape_index():
    # software floating point arm doesn't raise a warning on divide by zero
    # https://github.com/scikit-image/scikit-image/issues/3335
    square = cp.zeros((5, 5))
    square[2, 2] = 4
    with expected_warnings([r"divide by zero|\A\Z", r"invalid value|\A\Z"]):
        s = shape_index(square, sigma=0.1)
    # fmt: off
    assert_array_almost_equal(
        s,
        cp.asarray(
            [
                [cp.nan, cp.nan,   -0.5, cp.nan, cp.nan],  # noqa
                [cp.nan,      0, cp.nan,      0, cp.nan],  # noqa
                [  -0.5, cp.nan,     -1, cp.nan,   -0.5],  # noqa
                [cp.nan,      0, cp.nan,      0, cp.nan],  # noqa
                [cp.nan, cp.nan,   -0.5, cp.nan, cp.nan],  # noqa
            ]
        )
    )
    # fmt: on


# @test_parallel()
def test_square_image():
    im = cp.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.0

    # # Moravec
    # results = peak_local_max(corner_moravec(im),
    #                          min_distance=10, threshold_rel=0)
    # # interest points along edge
    # assert len(results) == 57

    # Harris
    results = peak_local_max(corner_harris(im, method='k'),
                             min_distance=10, threshold_rel=0)
    # interest at corner
    assert len(results) == 1

    results = peak_local_max(corner_harris(im, method='eps'),
                             min_distance=10, threshold_rel=0)
    # interest at corner
    assert len(results) == 1

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im),
                             min_distance=10, threshold_rel=0)
    # interest at corner
    assert len(results) == 1


def test_noisy_square_image():
    im = cp.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.0
    np.random.seed(seed=1234)  # result is specic to this NumPy seed
    im = im + cp.asarray(np.random.uniform(size=im.shape)) * 0.2

    # # Moravec
    # results = peak_local_max(corner_moravec(im),
    #                          min_distance=10, threshold_rel=0)
    # # undefined number of interest points
    # assert results.any()

    # Harris
    results = peak_local_max(corner_harris(im, method='k'),
                             min_distance=10, threshold_rel=0)
    assert len(results) == 1
    results = peak_local_max(corner_harris(im, method='eps'),
                             min_distance=10, threshold_rel=0)
    assert len(results) == 1

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im, sigma=1.5),
                             min_distance=10, threshold_rel=0)
    assert len(results) == 1


def test_squared_dot():
    im = cp.zeros((50, 50))
    im[4:8, 4:8] = 1
    im = img_as_float(im)

    # Moravec fails

    # Harris
    results = peak_local_max(corner_harris(im),
                             min_distance=10, threshold_rel=0)

    assert (results == cp.asarray([[6, 6]])).all()

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im),
                             min_distance=10, threshold_rel=0)

    assert (results == cp.asarray([[6, 6]])).all()


def test_rotated_img():
    """
    The harris filter should yield the same results with an image and it's
    rotation.
    """
    im = img_as_float(cp.asarray(data.astronaut().mean(axis=2)))
    im_rotated = im.T

    # # Moravec
    # results = peak_local_max(corner_moravec(im),
    #                          min_distance=10, threshold_rel=0)
    # results_rotated = peak_local_max(corner_moravec(im_rotated),
    #                                  min_distance=10, threshold_rel=0)
    # assert (cp.sort(results[:, 0]) == cp.sort(results_rotated[:, 1])).all()
    # assert (cp.sort(results[:, 1]) == cp.sort(results_rotated[:, 0])).all()

    # Harris
    results = cp.nonzero(corner_harris(im))
    results_rotated = cp.nonzero(corner_harris(im_rotated))

    assert (cp.sort(results[0]) == cp.sort(results_rotated[1])).all()
    assert (cp.sort(results[1]) == cp.sort(results_rotated[0])).all()

    # Shi-Tomasi
    results = cp.nonzero(corner_shi_tomasi(im))
    results_rotated = cp.nonzero(corner_shi_tomasi(im_rotated))

    assert (cp.sort(results[0]) == cp.sort(results_rotated[1])).all()
    assert (cp.sort(results[1]) == cp.sort(results_rotated[0])).all()


# def test_subpix_edge():
#     img = cp.zeros((50, 50))
#     img[:25, :25] = 255
#     img[25:, 25:] = 255
#     corner = peak_local_max(corner_harris(img),
#                             min_distance=10, threshold_rel=0, num_peaks=1)
#     subpix = corner_subpix(img, corner)
#     assert_array_equal(subpix[0], (24.5, 24.5))


# def test_subpix_dot():
#     img = cp.zeros((50, 50))
#     img[25, 25] = 255
#     corner = peak_local_max(corner_harris(img),
#                             min_distance=10, threshold_rel=0, num_peaks=1)
#     subpix = corner_subpix(img, corner)
#     assert_array_equal(subpix[0], (25, 25))


# def test_subpix_no_class():
#     img = cp.zeros((50, 50))
#     subpix = corner_subpix(img, cp.asarray([[25, 25]]))
#     assert_array_equal(subpix[0], (cp.nan, cp.nan))

#     img[25, 25] = 1e-10
#     corner = peak_local_max(corner_harris(img),
#                             min_distance=10, threshold_rel=0, num_peaks=1)
#     subpix = corner_subpix(img, cp.asarray([[25, 25]]))
#     assert_array_equal(subpix[0], (cp.nan, cp.nan))


# def test_subpix_border():
#     img = cp.zeros((50, 50))
#     img[1:25, 1:25] = 255
#     img[25:-1, 25:-1] = 255
#     corner = corner_peaks(corner_harris(img), threshold_rel=0)
#     subpix = corner_subpix(img, corner, window_size=11)
#     ref = cp.asarray([[24.5, 24.5],
#                     [0.52040816, 0.52040816],
#                     [0.52040816, 24.47959184],
#                     [24.47959184, 0.52040816],
#                     [24.52040816, 48.47959184],
#                     [48.47959184, 24.52040816],
#                     [48.47959184, 48.47959184]])

#     assert_array_almost_equal(subpix, ref)


def test_num_peaks():
    """For a bunch of different values of num_peaks, check that
    peak_local_max returns exactly the right amount of peaks. Test
    is run on the astronaut image in order to produce a sufficient number of
    corners
    """

    img_corners = corner_harris(rgb2gray(cp.asarray(data.astronaut())))

    for i in range(20):
        n = cp.random.randint(1, 21)
        results = peak_local_max(img_corners,
                                 min_distance=10, threshold_rel=0, num_peaks=n)
        assert results.shape[0] == n


def test_corner_peaks():
    response = cp.zeros((10, 10))
    response[2:5, 2:5] = 1
    response[8:10, 0:2] = 1

    corners = corner_peaks(response, exclude_border=False, min_distance=10,
                           threshold_rel=0)
    assert corners.shape == (1, 2)

    corners = corner_peaks(response, exclude_border=False, min_distance=5,
                           threshold_rel=0)
    assert corners.shape == (2, 2)

    corners = corner_peaks(response, exclude_border=False, min_distance=1)
    assert corners.shape == (5, 2)

    corners = corner_peaks(response, exclude_border=False, min_distance=1,
                           indices=False)
    assert cp.sum(corners) == 5


def test_blank_image_nans():
    """Some of the corner detectors had a weakness in terms of returning
    NaN when presented with regions of constant intensity. This should
    be fixed by now. We test whether each detector returns something
    finite in the case of constant input"""

    #    detectors = [corner_moravec, corner_harris, corner_shi_tomasi,
    #                 corner_kitchen_rosenfeld]
    detectors = [
        corner_harris,
        corner_shi_tomasi,
        corner_kitchen_rosenfeld,
    ]
    constant_image = cp.zeros((20, 20))

    for det in detectors:
        response = det(constant_image)
        assert cp.all(cp.isfinite(response))

    w, q = corner_foerstner(constant_image)
    assert cp.all(cp.isfinite(w))
    assert cp.all(cp.isfinite(q))


# def test_corner_fast_image_unsupported_error():
#     img = cp.zeros((20, 20, 3))
#     with pytest.raises(ValueError):
#         corner_fast(img)


# # @test_parallel()
# def test_corner_fast_astronaut():
#     img = rgb2gray(cp.asarray(data.astronaut()))
#     expected = cp.asarray([[444, 310],
#                          [374, 171],
#                          [249, 171],
#                          [492, 139],
#                          [403, 162],
#                          [496, 266],
#                          [362, 328],
#                          [476, 250],
#                          [353, 172],
#                          [346, 279],
#                          [494, 169],
#                          [177, 156],
#                          [413, 181],
#                          [213, 117],
#                          [390, 149],
#                          [140, 205],
#                          [232, 266],
#                          [489, 155],
#                          [387, 195],
#                          [101, 198],
#                          [363, 192],
#                          [364, 147],
#                          [300, 244],
#                          [325, 245],
#                          [141, 242],
#                          [401, 197],
#                          [197, 148],
#                          [339, 242],
#                          [188, 113],
#                          [362, 252],
#                          [379, 183],
#                          [358, 307],
#                          [245, 137],
#                          [369, 159],
#                          [464, 251],
#                          [305,  57],
#                          [223, 375]])
#     actual = corner_peaks(corner_fast(img, 12, 0.3),
#                           min_distance=10, threshold_rel=0)
#     assert_array_equal(actual, expected)


# def test_corner_orientations_image_unsupported_error():
#     img = cp.zeros((20, 20, 3))
#     with pytest.raises(ValueError):
#         corner_orientations(
#             img,
#             cp.asarray([[7, 7]]), cp.ones((3, 3)))


# def test_corner_orientations_even_shape_error():
#     img = cp.zeros((20, 20))
#     with pytest.raises(ValueError):
#         corner_orientations(
#             img,
#             cp.asarray([[7, 7]]), cp.ones((4, 4)))


# # @test_parallel()
# def test_corner_orientations_astronaut():
#     img = rgb2gray(cp.asarray(data.astronaut()))
#     corners = corner_peaks(corner_fast(img, 11, 0.35),
#                            min_distance=10, threshold_abs=0,
#                            threshold_rel=0.1)
#     expected = cp.asarray([-4.40598471e-01, -1.46554357e+00,
#                          2.39291733e+00, -1.63869275e+00,
#                          1.45931342e+00, -1.64397304e+00,
#                          -1.76069982e+00, 1.09650167e+00,
#                          -1.65449964e+00, 1.19134149e+00,
#                          5.46905279e-02, 2.17103132e+00,
#                          8.12701702e-01, -1.22091334e-01,
#                          -2.01162417e+00, 1.25854853e+00,
#                          3.05330950e+00, 2.01197383e+00,
#                          1.07812134e+00, 3.09780364e+00,
#                          -3.49561988e-01, 2.43573659e+00,
#                          3.14918803e-01, -9.88548213e-01,
#                          -1.88247204e-01, 2.47305654e+00,
#                          -2.99143370e+00, 1.47154532e+00,
#                          -6.61151410e-01, -1.68885773e+00,
#                          -3.09279990e-01, -2.81524886e+00,
#                          -1.75220190e+00, -1.69230287e+00,
#                          -7.52950306e-04])

#     actual = corner_orientations(img, corners, octagon(3, 2))
#     assert_array_almost_equal(actual, expected)


# def test_corner_orientations_square():
#     square = cp.zeros((12, 12))
#     square[3:9, 3:9] = 1
#     corners = corner_peaks(corner_fast(square, 9),
#                            min_distance=1, threshold_rel=0)
#     actual_orientations = corner_orientations(square, corners, octagon(3, 2))
#     actual_orientations_degrees = cp.rad2deg(actual_orientations)
#     expected_orientations_degree = cp.asarray([45, 135, -45, -135])
#     assert_array_equal(actual_orientations_degrees,
#                        expected_orientations_degree)
