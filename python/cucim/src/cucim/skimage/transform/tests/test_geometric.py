import re
import textwrap

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal, assert_array_equal

from cucim.skimage.transform import (AffineTransform, EssentialMatrixTransform,
                                     EuclideanTransform,
                                     FundamentalMatrixTransform,
                                     PiecewiseAffineTransform,
                                     PolynomialTransform, ProjectiveTransform,
                                     SimilarityTransform, estimate_transform,
                                     matrix_transform)
from cucim.skimage.transform._geometric import (GeometricTransform,
                                                _affine_matrix_from_vector,
                                                _center_and_normalize_points,
                                                _euler_rotation_matrix)

# fmt: off
SRC = cp.array([
    [-12.3705, -10.5075],
    [-10.7865, 15.4305],
    [8.6985, 10.8675],
    [11.4975, -9.5715],
    [7.8435, 7.4835],
    [-5.3325, 6.5025],
    [6.7905, -6.3765],
    [-6.1695, -0.8235],
])
DST = cp.array([
    [0, 0],
    [0, 5800],
    [4900, 5800],
    [4900, 0],
    [4479, 4580],
    [1176, 3660],
    [3754, 790],
    [1024, 1931],
])
# fmt: on


def test_estimate_transform():
    for tform in ('euclidean', 'similarity', 'affine', 'projective',
                  'polynomial'):
        estimate_transform(tform, SRC[:2, :], DST[:2, :])
    with pytest.raises(ValueError):
        estimate_transform('foobar', SRC[:2, :], DST[:2, :])


def test_matrix_transform():
    tform = AffineTransform(scale=(0.1, 0.5), rotation=2)
    assert_array_equal(tform(SRC), matrix_transform(SRC, tform.params))


def test_euclidean_estimation():
    # exact solution
    tform = estimate_transform('euclidean', SRC[:2, :], SRC[:2, :] + 10)
    assert_array_almost_equal(tform(SRC[:2, :]), SRC[:2, :] + 10)
    assert_array_almost_equal(tform.params[0, 0], tform.params[1, 1])
    assert_array_almost_equal(tform.params[0, 1], -tform.params[1, 0])

    # over-determined
    tform2 = estimate_transform('euclidean', SRC, DST)
    assert_array_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_array_almost_equal(tform2.params[0, 0], tform2.params[1, 1])
    assert_array_almost_equal(tform2.params[0, 1], -tform2.params[1, 0])

    # via estimate method
    tform3 = EuclideanTransform()
    assert tform3.estimate(SRC, DST)
    assert_array_almost_equal(tform3.params, tform2.params)


def test_euclidean_init():
    # init with implicit parameters
    rotation = 1
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = EuclideanTransform(tform.params)
    assert_array_almost_equal(tform2.rotation, rotation)
    assert_array_almost_equal(tform2.translation, translation)

    # test special case for scale if rotation=0
    rotation = 0
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)

    # test special case for scale if rotation=90deg
    rotation = np.pi / 2
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)


def test_similarity_estimation():
    # exact solution
    tform = estimate_transform('similarity', SRC[:2, :], DST[:2, :])
    assert_array_almost_equal(tform(SRC[:2, :]), DST[:2, :])
    assert_array_almost_equal(tform.params[0, 0], tform.params[1, 1])
    assert_array_almost_equal(tform.params[0, 1], -tform.params[1, 0])

    # over-determined
    tform2 = estimate_transform('similarity', SRC, DST)
    assert_array_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_array_almost_equal(tform2.params[0, 0], tform2.params[1, 1])
    assert_array_almost_equal(tform2.params[0, 1], -tform2.params[1, 0])

    # via estimate method
    tform3 = SimilarityTransform()
    assert tform3.estimate(SRC, DST)
    assert_array_almost_equal(tform3.params, tform2.params)


def test_similarity_init():
    # init with implicit parameters
    scale = 0.1
    rotation = 1
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = SimilarityTransform(tform.params)
    assert_array_almost_equal(tform2.scale, scale)
    assert_array_almost_equal(tform2.rotation, rotation)
    assert_array_almost_equal(tform2.translation, translation)

    # test special case for scale if rotation=0
    scale = 0.1
    rotation = 0
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)

    # test special case for scale if rotation=90deg
    scale = 0.1
    rotation = np.pi / 2
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)

    # test special case for scale where the rotation isn't exactly 90deg,
    # but very close
    scale = 1.0
    rotation = np.pi / 2
    translation = (0, 0)
    params = np.array([[0, -1, 1.33226763e-15],
                       [1, 2.22044605e-16, -1.33226763e-15],
                       [0, 0, 1]])
    tform = SimilarityTransform(params)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)


def test_affine_estimation():
    # exact solution
    tform = estimate_transform('affine', SRC[:3, :], DST[:3, :])
    assert_array_almost_equal(tform(SRC[:3, :]), DST[:3, :])

    # over-determined
    tform2 = estimate_transform('affine', SRC, DST)
    assert_array_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = AffineTransform()
    assert tform3.estimate(SRC, DST)
    assert_array_almost_equal(tform3.params, tform2.params)


def test_affine_init():
    # init with implicit parameters
    scale = (0.1, 0.13)
    rotation = 1
    shear = 0.1
    translation = (1, 1)
    tform = AffineTransform(scale=scale, rotation=rotation, shear=shear,
                            translation=translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.shear, shear)
    assert_array_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = AffineTransform(tform.params)
    assert_array_almost_equal(tform2.scale, scale)
    assert_array_almost_equal(tform2.rotation, rotation)
    assert_array_almost_equal(tform2.shear, shear)
    assert_array_almost_equal(tform2.translation, translation)

    # scalar vs. tuple scale arguments
    assert_array_almost_equal(AffineTransform(scale=0.5).scale,
                              AffineTransform(scale=(0.5, 0.5)).scale)


def test_piecewise_affine():
    tform = PiecewiseAffineTransform()
    assert tform.estimate(SRC, DST)
    # make sure each single affine transform is exactly estimated
    assert_array_almost_equal(tform(SRC), DST)
    assert_array_almost_equal(tform.inverse(DST), SRC)


@pytest.mark.parametrize("xp", [np, cp])
def test_fundamental_matrix_estimation(xp):
    # fmt: off
    src = xp.array([1.839035, 1.924743, 0.543582,  0.375221,  # noqa
                    0.473240, 0.142522, 0.964910,  0.598376,  # noqa
                    0.102388, 0.140092, 15.994343, 9.622164,  # noqa
                    0.285901, 0.430055, 0.091150,  0.254594]).reshape(-1, 2)  # noqa
    dst = xp.array([1.002114, 1.129644, 1.521742, 1.846002,  # noqa
                    1.084332, 0.275134, 0.293328, 0.588992,  # noqa
                    0.839509, 0.087290, 1.779735, 1.116857,  # noqa
                    0.878616, 0.602447, 0.642616, 1.028681]).reshape(-1, 2)  # noqa
    # fmt: on

    tform = estimate_transform('fundamental', src, dst)

    # Reference values obtained using COLMAP SfM library.
    # fmt: off
    tform_ref = xp.array([[-0.217859,  0.419282, -0.0343075],   # noqa
                          [-0.0717941, 0.0451643, 0.0216073],   # noqa
                          [ 0.248062, -0.429478,  0.0221019]])  # noqa

    # fmt: on
    if xp == cp:
        # TODO: grlee77: why is there a sign difference here for CuPy?
        tform_ref = -tform_ref
    assert_array_almost_equal(tform.params, tform_ref, 6)


@pytest.mark.parametrize("xp", [np, cp])
def test_fundamental_matrix_residuals(xp):
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=xp.eye(3), translation=xp.array([1, 0, 0]), xp=xp)
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = xp.array([[0, 0], [0, 0], [0, 0]])
    dst = xp.array([[2, 0], [2, 1], [2, 2]])
    assert_array_almost_equal(
        tform.residuals(src, dst) ** 2, xp.array([0, 0.5, 2]))


@pytest.mark.parametrize("xp", [np, cp])
def test_fundamental_matrix_forward(xp):
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=xp.eye(3), translation=xp.array([1, 0, 0]), xp=xp)
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = xp.array([[0, 0], [0, 1], [1, 1]])
    assert_array_almost_equal(
        tform(src), xp.array([[0, -1, 0], [0, -1, 1], [0, -1, 1]]))


@pytest.mark.parametrize("xp", [np, cp])
def test_fundamental_matrix_inverse(xp):
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=xp.eye(3), translation=xp.array([1, 0, 0]), xp=xp)
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = xp.array([[0, 0], [0, 1], [1, 1]])
    assert_array_almost_equal(
        tform.inverse(src), xp.array([[0, 1, 0], [0, 1, -1], [0, 1, -1]]))


@pytest.mark.parametrize("xp", [np, cp])
def test_essential_matrix_init(xp):
    tform = EssentialMatrixTransform(rotation=xp.eye(3),
                                     translation=xp.array([0, 0, 1]), xp=xp)

    assert_array_equal(
        tform.params, xp.array([0, -1, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3))


@pytest.mark.parametrize("xp", [np, cp])
def test_essential_matrix_estimation(xp):
    # fmt: off
    src = xp.array([1.839035, 1.924743, 0.543582,  0.375221,  # noqa
                    0.473240, 0.142522, 0.964910,  0.598376,  # noqa
                    0.102388, 0.140092, 15.994343, 9.622164,  # noqa
                    0.285901, 0.430055, 0.091150,  0.254594]).reshape(-1, 2)  # noqa
    dst = xp.array([1.002114, 1.129644, 1.521742, 1.846002,  # noqa
                    1.084332, 0.275134, 0.293328, 0.588992,  # noqa
                    0.839509, 0.087290, 1.779735, 1.116857,  # noqa
                    0.878616, 0.602447, 0.642616, 1.028681]).reshape(-1, 2)  # noqa
    # fmt: on
    tform = estimate_transform('essential', src, dst)

    # Reference values obtained using COLMAP SfM library.
    # fmt: off
    tform_ref = xp.array([[-0.0811666, 0.255449, -0.0478999],  # noqa
                          [-0.192392, -0.0531675, 0.119547],  # noqa
                          [ 0.177784, -0.22008,  -0.015203]])  # noqa

    # fmt: on
    if xp == cp:
        # TODO: grlee77: why is there a sign difference here for CuPy?
        tform_ref = -tform_ref
    assert_array_almost_equal(tform.params, tform_ref, 6)


@pytest.mark.parametrize("xp", [np, cp])
def test_essential_matrix_forward(xp):
    tform = EssentialMatrixTransform(rotation=xp.eye(3),
                                     translation=xp.array([1, 0, 0]), xp=xp)
    src = xp.array([[0, 0], [0, 1], [1, 1]])
    assert_array_almost_equal(
        tform(src), xp.array([[0, -1, 0], [0, -1, 1], [0, -1, 1]]))


@pytest.mark.parametrize("xp", [np, cp])
def test_essential_matrix_inverse(xp):
    tform = EssentialMatrixTransform(rotation=xp.eye(3),
                                     translation=xp.array([1, 0, 0]), xp=xp)
    src = xp.array([[0, 0], [0, 1], [1, 1]])
    assert_array_almost_equal(tform.inverse(src),
                              xp.array([[0, 1, 0], [0, 1, -1], [0, 1, -1]]))


@pytest.mark.parametrize("xp", [np, cp])
def test_essential_matrix_residuals(xp):
    tform = EssentialMatrixTransform(rotation=xp.eye(3),
                                     translation=xp.array([1, 0, 0]), xp=xp)
    src = xp.array([[0, 0], [0, 0], [0, 0]])
    dst = xp.array([[2, 0], [2, 1], [2, 2]])
    assert_array_almost_equal(
        tform.residuals(src, dst) ** 2, xp.array([0, 0.5, 2]))


def test_projective_estimation():
    # exact solution
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    assert_array_almost_equal(tform(SRC[:4, :]), DST[:4, :])

    # over-determined
    tform2 = estimate_transform('projective', SRC, DST)
    assert_array_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = ProjectiveTransform()
    assert tform3.estimate(SRC, DST)
    assert_array_almost_equal(tform3.params, tform2.params)


def test_projective_weighted_estimation():

    # Exact solution with same points, and unity weights
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    tform_w = estimate_transform('projective',
                                 SRC[:4, :], DST[:4, :], cp.ones(4))
    assert_array_almost_equal(tform.params, tform_w.params)

    # Over-determined solution with same points, and unity weights
    tform = estimate_transform('projective', SRC, DST)
    tform_w = estimate_transform('projective',
                                 SRC, DST, cp.ones(SRC.shape[0]))
    assert_array_almost_equal(tform.params, tform_w.params)

    # Repeating a point, but setting its weight small, should give nearly
    # the same result.
    point_weights = cp.ones(SRC.shape[0] + 1)
    point_weights[0] = 1.0e-15
    tform1 = estimate_transform('projective', SRC, DST)
    tform2 = estimate_transform('projective',
                                SRC[cp.arange(-1, SRC.shape[0]), :],
                                DST[cp.arange(-1, SRC.shape[0]), :],
                                point_weights)
    assert_array_almost_equal(tform1.params, tform2.params, decimal=3)


def test_projective_init():
    tform = estimate_transform('projective', SRC, DST)
    # init with transformation matrix
    tform2 = ProjectiveTransform(tform.params)
    assert_array_almost_equal(tform2.params, tform.params)


def test_polynomial_estimation():
    # over-determined
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    assert_array_almost_equal(tform(SRC), DST, 6)

    # via estimate method
    tform2 = PolynomialTransform()
    assert tform2.estimate(SRC, DST, order=10)
    assert_array_almost_equal(tform2.params, tform.params)


def test_polynomial_weighted_estimation():
    # Over-determined solution with same points, and unity weights
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    tform_w = estimate_transform('polynomial',
                                 SRC,
                                 DST,
                                 order=10,
                                 weights=cp.ones(SRC.shape[0]))
    assert_array_almost_equal(tform.params, tform_w.params)

    # Repeating a point, but setting its weight small, should give nearly
    # the same result.
    point_weights = cp.ones(SRC.shape[0] + 1)
    point_weights[0] = 1.0e-15
    tform1 = estimate_transform('polynomial', SRC, DST, order=10)
    tform2 = estimate_transform('polynomial',
                                SRC[cp.arange(-1, SRC.shape[0]), :],
                                DST[cp.arange(-1, SRC.shape[0]), :],
                                order=10,
                                weights=point_weights)
    assert_array_almost_equal(tform1.params, tform2.params, decimal=4)


def test_polynomial_init():
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    # init with transformation parameters
    tform2 = PolynomialTransform(tform.params)
    assert_array_almost_equal(tform2.params, tform.params)


def test_polynomial_default_order():
    tform = estimate_transform('polynomial', SRC, DST)
    tform2 = estimate_transform('polynomial', SRC, DST, order=2)
    assert_array_almost_equal(tform2.params, tform.params)


def test_polynomial_inverse():
    with pytest.raises(Exception):
        PolynomialTransform().inverse(0)


def test_union():
    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 ** 2, rotation=0.3 + 0.9)
    tform = tform1 + tform2
    assert_array_almost_equal(tform.params, tform3.params)

    tform1 = AffineTransform(scale=(0.1, 0.1), rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 ** 2, rotation=0.3 + 0.9)
    tform = tform1 + tform2
    assert_array_almost_equal(tform.params, tform3.params)
    assert tform.__class__ == ProjectiveTransform

    tform = AffineTransform(scale=(0.1, 0.1), rotation=0.3)
    assert_array_almost_equal((tform + tform.inverse).params, cp.eye(3))

    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 * 1 / 0.1, rotation=0.3 - 0.9)
    tform = tform1 + tform2.inverse
    assert_array_almost_equal(tform.params, tform3.params)


def test_union_differing_types():
    tform1 = SimilarityTransform()
    tform2 = PolynomialTransform()
    with pytest.raises(TypeError):
        tform1.__add__(tform2)


@pytest.mark.parametrize("xp", [np, cp])
def test_geometric_tform(xp):
    tform = GeometricTransform()
    with pytest.raises(NotImplementedError):
        tform(0)
    with pytest.raises(NotImplementedError):
        tform.inverse(0)
    with pytest.raises(NotImplementedError):
        tform.__add__(0)

    # See gh-3926 for discussion details
    for i in range(20):
        # Generate random Homography
        H = np.random.rand(3, 3) * 100
        H[2, H[2] == 0] += np.finfo(float).eps
        H /= H[2, 2]

        # Craft some src coords
        # fmt: off
        src = np.array([
            [(H[2, 1] + 1) / -H[2, 0], 1],
            [1, (H[2, 0] + 1) / -H[2, 1]],
            [1, 1],
        ])
        # fmt: on
        H = xp.asarray(H)
        src = xp.asarray(src)

        # Prior to gh-3926, under the above circumstances,
        # destination coordinates could be returned with nan/inf values.
        tform = ProjectiveTransform(H)  # Construct the transform
        dst = tform(src)  # Obtain the dst coords
        # Ensure dst coords are finite numeric values
        assert xp.isfinite(dst).all()


@pytest.mark.parametrize("xp", [np, cp])
def test_invalid_input(xp):
    with pytest.raises(ValueError):
        ProjectiveTransform(xp.zeros((2, 3)))
    with pytest.raises(ValueError):
        AffineTransform(xp.zeros((2, 3)))
    with pytest.raises(ValueError):
        SimilarityTransform(xp.zeros((2, 3)))
    with pytest.raises(ValueError):
        EuclideanTransform(xp.zeros((2, 3)))
    with pytest.raises(ValueError):
        AffineTransform(matrix=xp.zeros((2, 3)), scale=1)
    with pytest.raises(ValueError):
        SimilarityTransform(matrix=xp.zeros((2, 3)), scale=1)
    with pytest.raises(ValueError):
        EuclideanTransform(matrix=xp.zeros((2, 3)), translation=(0, 0))
    with pytest.raises(ValueError):
        PolynomialTransform(xp.zeros((3, 3)))
    with pytest.raises(ValueError):
        FundamentalMatrixTransform(matrix=xp.zeros((3, 2)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(matrix=xp.zeros((3, 2)))

    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=xp.zeros((3, 2)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=xp.zeros((3, 3)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=xp.eye(3))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=xp.eye(3),
                                 translation=xp.zeros((2,)), xp=xp)
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=xp.eye(3),
                                 translation=xp.zeros((2,)), xp=xp)
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=xp.eye(3),
                                 translation=xp.zeros((3,)), xp=xp)


def test_degenerate(xp=cp):
    src = dst = xp.zeros((10, 2))

    tform = SimilarityTransform()
    assert not tform.estimate(src, dst)
    assert xp.all(xp.isnan(tform.params))

    tform = EuclideanTransform()
    assert not tform.estimate(src, dst)
    assert xp.all(xp.isnan(tform.params))

    tform = AffineTransform()
    assert not tform.estimate(src, dst)
    assert xp.all(xp.isnan(tform.params))

    tform = ProjectiveTransform()
    assert not tform.estimate(src, dst)
    assert xp.all(xp.isnan(tform.params))

    # See gh-3926 for discussion details
    tform = ProjectiveTransform()
    for i in range(20):
        # Some random coordinates
        src = xp.random.rand(4, 2) * 100
        dst = xp.random.rand(4, 2) * 100

        # Degenerate the case by arranging points on a single line
        src[:, 1] = xp.random.rand()
        # Prior to gh-3926, under the above circumstances,
        # a transform could be returned with nan values.
        assert not tform.estimate(src, dst) or xp.isfinite(tform.params).all()

    src = xp.array([[0, 2, 0], [0, 2, 0], [0, 4, 0]])
    dst = xp.array([[0, 1, 0], [0, 1, 0], [0, 3, 0]])
    tform = AffineTransform()
    assert not tform.estimate(src, dst)
    assert xp.all(xp.isnan(tform.params))

    # The tesselation on the following points produces one degenerate affine
    # warp within PiecewiseAffineTransform.
    src = xp.asarray([
        [0, 192, 256], [0, 256, 256], [5, 0, 192], [5, 64, 0], [5, 64, 64],
        [5, 64, 256], [5, 192, 192], [5, 256, 256], [0, 192, 256],
    ])

    dst = xp.asarray([
        [0, 142, 206], [0, 206, 206], [5, -50, 142], [5, 14, 0], [5, 14, 64],
        [5, 14, 206], [5, 142, 142], [5, 206, 206], [0, 142, 206],
    ])
    tform = PiecewiseAffineTransform()
    assert not tform.estimate(src, dst)
    assert np.all(np.isnan(tform.affines[4].params))  # degenerate affine
    for idx, affine in enumerate(tform.affines):
        if idx != 4:
            assert not xp.all(xp.isnan(affine.params))
    for affine in tform.inverse_affines:
        assert not xp.all(xp.isnan(affine.params))


@pytest.mark.parametrize('xp', [np, cp])
def test_normalize_degenerate_points(xp):
    """Return nan matrix *of appropriate size* when point is repeated."""
    pts = xp.array([[73.42834308, 94.2977623]] * 3)
    mat, pts_tf, _ = _center_and_normalize_points(pts)
    assert xp.all(xp.isnan(mat))
    assert xp.all(xp.isnan(pts_tf))
    assert mat.shape == (3, 3)
    assert pts_tf.shape == pts.shape


@pytest.mark.parametrize("xp", [np, cp])
def test_projective_repr(xp):
    tform = ProjectiveTransform(xp=xp)
    # fmt: off
    want = re.escape(textwrap.dedent(
        '''
        <ProjectiveTransform(matrix=
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]) at
        ''').strip()) + ' 0x[a-f0-9]+' + re.escape('>')
    # fmt: on
    # Hack the escaped regex to allow whitespace before each number for
    # compatibility with different numpy versions.
    want = want.replace('0\\.', ' *0\\.')
    want = want.replace('1\\.', ' *1\\.')
    assert re.match(want, repr(tform))


@pytest.mark.parametrize("xp", [np, cp])
def test_projective_str(xp):
    tform = ProjectiveTransform(xp=xp)
    # fmt: off
    want = re.escape(textwrap.dedent(
        '''
        <ProjectiveTransform(matrix=
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]])>
        ''').strip())
    # fmt: on
    # Hack the escaped regex to allow whitespace before each number for
    # compatibility with different numpy versions.
    want = want.replace('0\\.', ' *0\\.')
    want = want.replace('1\\.', ' *1\\.')
    print(want)
    assert re.match(want, str(tform))


def _assert_least_squares(tf, src, dst, xp=cp):
    baseline = xp.sum((tf(src) - dst) ** 2)
    for i in range(tf.params.size):
        for update in [0.001, -0.001]:
            params = xp.copy(tf.params).ravel()
            params[i] += update
            params = xp.reshape(params, tf.params.shape)
            new_tf = tf.__class__(matrix=params)
            new_ssq = np.sum((new_tf(src) - dst) ** 2)
            assert new_ssq > baseline


@pytest.mark.parametrize("xp", [np, cp])
def test_estimate_affine_3d(xp):
    ndim = 3
    src = np.random.random((25, ndim)) * 2 ** np.arange(7, 7 + ndim)
    src = xp.asarray(src)
    matrix = xp.array(
        [[4.8, 0.1, 0.2, 25],
         [0.0, 1.0, 0.1, 30],
         [0.0, 0.0, 1.0, -2],
         [0.0, 0.0, 0.0, 1.]]
    )
    tf = AffineTransform(matrix=matrix)
    dst = tf(src)
    dst_noisy = dst + xp.asarray(np.random.random((25, ndim)))
    tf2 = AffineTransform(dimensionality=ndim)
    assert tf2.estimate(src, dst_noisy)
    # we check rot/scale/etc more tightly than translation because translation
    # estimation is on the 1 pixel scale
    assert_array_almost_equal(tf2.params[:, :-1], matrix[:, :-1], decimal=2)
    assert_array_almost_equal(tf2.params[:, -1], matrix[:, -1], decimal=0)
    _assert_least_squares(tf2, src, dst_noisy)


def test_fundamental_3d_not_implemented():
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(dimensionality=3)
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(np.eye(4))


# array protocol not expected to work for CuPy arrays
# def test_array_protocol():
#     mat = cp.eye(4)
#     tf = ProjectiveTransform(mat)
#     assert_array_equal(cp.array(tf), mat)
#     assert_array_equal(cp.array(tf, dtype=int), mat.astype(int))


def test_affine_transform_from_linearized_parameters():
    mat = np.concatenate(
        (np.random.random((3, 4)), np.eye(4)[-1:]), axis=0
    )
    v = mat[:-1].ravel()
    mat_from_v = _affine_matrix_from_vector(v)
    tf = AffineTransform(matrix=mat_from_v)
    assert_array_equal(tf.params, mat)
    # incorrect number of parameters
    with pytest.raises(ValueError):
        _ = _affine_matrix_from_vector(v[:-1])
    with pytest.raises(ValueError):
        _ = AffineTransform(matrix=v[:-1])


def test_affine_params_nD_error():
    with pytest.raises(ValueError):
        _ = AffineTransform(scale=5, dimensionality=3)


def test_euler_rotation():
    v = [0, 10, 0]
    angles = np.radians([90, 45, 45])
    expected = [-5, -5, 7.1]
    R = _euler_rotation_matrix(angles)
    assert_array_almost_equal(R @ v, expected, decimal=1)


@pytest.mark.parametrize("xp", [np, cp])
def test_euclidean_param_defaults(xp):
    # 2D rotation is 0 when only translation is given
    tf = EuclideanTransform(translation=(5, 5))
    assert tf.params[0, 1] == 0
    # off diagonals are 0 when only translation is given
    tf = EuclideanTransform(translation=(4, 5, 9), dimensionality=3)
    assert xp.all(tf.params[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] == 0)
    with pytest.raises(ValueError):
        # specifying parameters for D>3 is not supported
        _ = EuclideanTransform(translation=(5, 6, 7, 8), dimensionality=4)
    with pytest.raises(ValueError):
        # incorrect number of angles for given dimensionality
        _ = EuclideanTransform(rotation=(4, 8), dimensionality=3)
    # translation is 0 when rotation is given
    tf = EuclideanTransform(rotation=np.pi * cp.arange(3), dimensionality=3)
    assert xp.all(tf.params[:-1, 3] == 0)


@pytest.mark.parametrize("xp", [np, cp])
def test_similarity_transform_params(xp):
    with pytest.raises(ValueError):
        _ = SimilarityTransform(translation=(4, 5, 6, 7), dimensionality=4,
                                xp=xp)
    tf = SimilarityTransform(scale=4, dimensionality=3, xp=xp)
    assert_array_equal(tf(xp.asarray([[1, 1, 1]])),
                       xp.asarray([[4, 4, 4]]))


@pytest.mark.parametrize("xp", [np, cp])
def test_euler_angle_consistency(xp):
    angles = xp.random.random((3,)) * 2 * np.pi - np.pi
    euclid = EuclideanTransform(rotation=angles, dimensionality=3, xp=xp)
    similar = SimilarityTransform(rotation=angles, dimensionality=3, xp=xp)
    assert_array_almost_equal(euclid.params, similar.params)


@pytest.mark.parametrize("xp", [np, cp])
def test_2D_only_implementations(xp):
    with pytest.raises(NotImplementedError):
        _ = PolynomialTransform(dimensionality=3, xp=xp)
    tf = AffineTransform(dimensionality=3, xp=xp)
    with pytest.raises(NotImplementedError):
        _ = tf.rotation
    with pytest.raises(NotImplementedError):
        _ = tf.shear
