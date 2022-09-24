import cupy as cp
import pytest
from cupy.testing import assert_almost_equal, assert_array_equal, assert_equal

# from cucim.skimage._shared.testing import test_parallel
from cucim.skimage._shared._dependency_checks import has_mpl
from cucim.skimage.draw import disk  # circle_perimeter, circle_perimeter_aa,
from cucim.skimage.draw import \
    ellipse  # set_color, line, line_aa, polygon, polygon_perimeter,; _bezier_segment, bezier_curve, rectangle,; rectangle_perimeter); ellipse_perimeter,
from cucim.skimage.measure import regionprops

#
#
# def test_set_color():
#     img = cp.zeros((10, 10))
#
#     rr, cc = line(0, 0, 0, 30)
#     set_color(img, (rr, cc), 1)
#
#     img_ = cp.zeros((10, 10))
#     img_[0, :] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_set_color_with_alpha():
#     img = cp.zeros((10, 10))
#
#     rr, cc, alpha = line_aa(0, 0, 0, 30)
#     set_color(img, (rr, cc), 1, alpha=alpha)
#
#     # Wrong dimensionality color
#     with pytest.raises(ValueError):
#         set_color(img, (rr, cc), (255, 0, 0), alpha=alpha)
#
#     img = cp.zeros((10, 10, 3))
#
#     rr, cc, alpha = line_aa(0, 0, 0, 30)
#     set_color(img, (rr, cc), (1, 0, 0), alpha=alpha)
#
#
# @test_parallel()
# def test_line_horizontal():
#     img = cp.zeros((10, 10))
#
#     rr, cc = line(0, 0, 0, 9)
#     img[rr, cc] = 1
#
#     img_ = cp.zeros((10, 10))
#     img_[0, :] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_line_vertical():
#     img = cp.zeros((10, 10))
#
#     rr, cc = line(0, 0, 9, 0)
#     img[rr, cc] = 1
#
#     img_ = cp.zeros((10, 10))
#     img_[:, 0] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_line_reverse():
#     img = cp.zeros((10, 10))
#
#     rr, cc = line(0, 9, 0, 0)
#     img[rr, cc] = 1
#
#     img_ = cp.zeros((10, 10))
#     img_[0, :] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_line_diag():
#     img = cp.zeros((5, 5))
#
#     rr, cc = line(0, 0, 4, 4)
#     img[rr, cc] = 1
#
#     img_ = cp.eye(5)
#
#     assert_array_equal(img, img_)
#
#
# def test_line_aa_horizontal():
#     img = cp.zeros((10, 10))
#
#     rr, cc, val = line_aa(0, 0, 0, 9)
#     set_color(img, (rr, cc), 1, alpha=val)
#
#     img_ = cp.zeros((10, 10))
#     img_[0, :] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_line_aa_vertical():
#     img = cp.zeros((10, 10))
#
#     rr, cc, val = line_aa(0, 0, 9, 0)
#     img[rr, cc] = val
#
#     img_ = cp.zeros((10, 10))
#     img_[:, 0] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_line_aa_diagonal():
#     img = cp.zeros((10, 10))
#
#     rr, cc, val = line_aa(0, 0, 9, 6)
#     img[rr, cc] = 1
#
#     # Check that each pixel belonging to line,
#     # also belongs to line_aa
#     r, c = line(0, 0, 9, 6)
#     for r_i, c_i in zip(r, c):
#         assert_equal(img[r_i, c_i], 1)
#
#
# def test_line_equal_aliasing_horizontally_vertically():
#     img0 = cp.zeros((25, 25))
#     img1 = cp.zeros((25, 25))
#
#     # Near-horizontal line
#     rr, cc, val = line_aa(10, 2, 12, 20)
#     img0[rr, cc] = val
#
#     # Near-vertical (transpose of prior)
#     rr, cc, val = line_aa(2, 10, 20, 12)
#     img1[rr, cc] = val
#
#     # Difference - should be zero
#     assert_array_equal(img0, img1.T)
#
#
# def test_polygon_rectangle():
#     img = cp.zeros((10, 10), 'uint8')
#     poly = cp.array(((1, 1), (4, 1), (4, 4), (1, 4), (1, 1)))
#
#     rr, cc = polygon(poly[:, 0], poly[:, 1])
#     img[rr, cc] = 1
#
#     img_ = cp.zeros((10, 10), 'uint8')
#     img_[1:5, 1:5] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_polygon_rectangle_angular():
#     img = cp.zeros((10, 10), 'uint8')
#     poly = cp.array(((0, 3), (4, 7), (7, 4), (3, 0), (0, 3)))
#
#     rr, cc = polygon(poly[:, 0], poly[:, 1])
#     img[rr, cc] = 1
#
#     img_ = cp.array(
#         [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
#          [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#          [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#          [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
#          [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'uint8'
#     )
#
#     assert_array_equal(img, img_)
#
#
# def test_polygon_parallelogram():
#     img = cp.zeros((10, 10), 'uint8')
#     poly = cp.array(((1, 1), (5, 1), (7, 6), (3, 6), (1, 1)))
#
#     rr, cc = polygon(poly[:, 0], poly[:, 1])
#     img[rr, cc] = 1
#
#     img_ = cp.array(
#         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#          [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#          [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#          [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'uint8'
#     )
#
#     assert_array_equal(img, img_)
#
#
# def test_polygon_exceed():
#     img = cp.zeros((10, 10), 'uint8')
#     poly = cp.array(((1, -1), (100, -1), (100, 100), (1, 100), (1, 1)))
#
#     rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
#     img[rr, cc] = 1
#
#     img_ = cp.zeros((10, 10))
#     img_[1:, :] = 1
#
#     assert_array_equal(img, img_)
#
#
# def test_polygon_0d_input():
#     # Bug reported in #4938: 0d input causes segfault.
#     rr, cc = polygon(0, 1)
#
#     assert rr.size == cc.size == 1


def test_disk():
    img = cp.zeros((15, 15), 'uint8')

    rr, cc = disk((7, 7), 6)
    img[rr, cc] = 1

    img_ = cp.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)


# def test_circle_perimeter_bresenham():
#     img = cp.zeros((15, 15), 'uint8')
#     rr, cc = circle_perimeter(7, 7, 0, method='bresenham')
#     img[rr, cc] = 1
#     assert(cp.sum(img) == 1)
#     assert(img[7][7] == 1)
#
#     img = cp.zeros((17, 15), 'uint8')
#     rr, cc = circle_perimeter(7, 7, 7, method='bresenham')
#     img[rr, cc] = 1
#     img_ = cp.array(
#         [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#     )
#     assert_array_equal(img, img_)
#
#
# def test_circle_perimeter_bresenham_shape():
#     img = cp.zeros((15, 20), 'uint8')
#     rr, cc = circle_perimeter(7, 10, 9, method='bresenham', shape=(15, 20))
#     img[rr, cc] = 1
#     shift = 5
#     img_ = cp.zeros((15 + 2 * shift, 20), 'uint8')
#     rr, cc = circle_perimeter(7 + shift, 10, 9, method='bresenham', shape=None)
#     img_[rr, cc] = 1
#     assert_array_equal(img, img_[shift:-shift, :])
#
#
# def test_circle_perimeter_andres():
#     img = cp.zeros((15, 15), 'uint8')
#     rr, cc = circle_perimeter(7, 7, 0, method='andres')
#     img[rr, cc] = 1
#     assert(cp.sum(img) == 1)
#     assert(img[7][7] == 1)
#
#     img = cp.zeros((17, 15), 'uint8')
#     rr, cc = circle_perimeter(7, 7, 7, method='andres')
#     img[rr, cc] = 1
#     img_ = cp.array(
#         [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#          [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
#          [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#          [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
#          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#     )
#     assert_array_equal(img, img_)
#
#
# def test_circle_perimeter_aa():
#     img = cp.zeros((15, 15), 'uint8')
#     rr, cc, val = circle_perimeter_aa(7, 7, 0)
#     img[rr, cc] = 1
#     assert(cp.sum(img) == 1)
#     assert(img[7][7] == 1)
#
#     img = cp.zeros((17, 17), 'uint8')
#     rr, cc, val = circle_perimeter_aa(8, 8, 7)
#     img[rr, cc] = val * 255
#     img_ = cp.array(
#         [[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
#          [  0,   0,   0,   0,   0,  82, 180, 236, 255, 236, 180,  82,   0,   0,   0,   0,   0],
#          [  0,   0,   0,   0, 189, 172,  74,  18,   0,  18,  74, 172, 189,   0,   0,   0,   0],
#          [  0,   0,   0, 229,  25,   0,   0,   0,   0,   0,   0,   0,  25, 229,   0,   0,   0],
#          [  0,   0, 189,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0,  25, 189,   0,   0],
#          [  0,  82, 172,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 172,  82,   0],
#          [  0, 180,  74,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  74, 180,   0],
#          [  0, 236,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  18, 236,   0],
#          [  0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0],
#          [  0, 236,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  18, 236,   0],
#          [  0, 180,  74,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  74, 180,   0],
#          [  0,  82, 172,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 172,  82,   0],
#          [  0,   0, 189,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0,  25, 189,   0,   0],
#          [  0,   0,   0, 229,  25,   0,   0,   0,   0,   0,   0,   0,  25, 229,   0,   0,   0],
#          [  0,   0,   0,   0, 189, 172,  74,  18,   0,  18,  74, 172, 189,   0,   0,   0,   0],
#          [  0,   0,   0,   0,   0,  82, 180, 236, 255, 236, 180,  82,   0,   0,   0,   0,   0],
#          [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]
#     )
#     assert_array_equal(img, img_)
#
#
# def test_circle_perimeter_aa_shape():
#     img = cp.zeros((15, 20), 'uint8')
#     rr, cc, val = circle_perimeter_aa(7, 10, 9, shape=(15, 20))
#     img[rr, cc] = val * 255
#
#     shift = 5
#     img_ = cp.zeros((15 + 2 * shift, 20), 'uint8')
#     rr, cc, val = circle_perimeter_aa(7 + shift, 10, 9, shape=None)
#     img_[rr, cc] = val * 255
#     assert_array_equal(img, img_[shift:-shift, :])


def test_ellipse_trivial():
    img = cp.zeros((2, 2), 'uint8')
    rr, cc = ellipse(0.5, 0.5, 0.5, 0.5)
    img[rr, cc] = 1
    img_correct = cp.array([
        [0, 0],
        [0, 0]
    ])
    assert_array_equal(img, img_correct)

    img = cp.zeros((2, 2), 'uint8')
    rr, cc = ellipse(0.5, 0.5, 1.1, 1.1)
    img[rr, cc] = 1
    img_correct = cp.array([
        [1, 1],
        [1, 1],
    ])
    assert_array_equal(img, img_correct)

    img = cp.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 0.9, 0.9)
    img[rr, cc] = 1
    img_correct = cp.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    assert_array_equal(img, img_correct)

    img = cp.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 1.1, 1.1)
    img[rr, cc] = 1
    img_correct = cp.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    assert_array_equal(img, img_correct)

    img = cp.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 1.5, 1.5)
    img[rr, cc] = 1
    img_correct = cp.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    assert_array_equal(img, img_correct)


def test_ellipse_generic():
    img = cp.zeros((4, 4), 'uint8')
    rr, cc = ellipse(1.5, 1.5, 1.1, 1.7)
    img[rr, cc] = 1
    img_ = cp.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = cp.zeros((5, 5), 'uint8')
    rr, cc = ellipse(2, 2, 1.7, 1.7)
    img[rr, cc] = 1
    img_ = cp.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = cp.zeros((10, 10), 'uint8')
    rr, cc = ellipse(5, 5, 3, 4)
    img[rr, cc] = 1
    img_ = cp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = cp.zeros((10, 10), 'uint8')
    rr, cc = ellipse(4.5, 5, 3.5, 4)
    img[rr, cc] = 1
    img_ = cp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = cp.zeros((15, 15), 'uint8')
    rr, cc = ellipse(7, 7, 3, 7)
    img[rr, cc] = 1
    img_ = cp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)


def test_ellipse_with_shape():
    img = cp.zeros((15, 15), 'uint8')

    rr, cc = ellipse(7, 7, 3, 10, shape=img.shape)
    img[rr, cc] = 1

    img_ = cp.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)

    img = cp.zeros((10, 9, 3), 'uint8')

    rr, cc = ellipse(7, 7, 3, 10, shape=img.shape)
    img[rr, cc, 0] = 1

    img_ = cp.zeros_like(img)
    img_[..., 0] = cp.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
    )

    assert_array_equal(img, img_)


def test_ellipse_negative():
    rr, cc = ellipse(-3, -3, 1.7, 1.7)
    rr_, cc_ = cp.nonzero(cp.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]))

    assert_array_equal(rr, rr_ - 5)
    assert_array_equal(cc, cc_ - 5)


def test_ellipse_rotation_symmetry():
    img1 = cp.zeros((150, 150), dtype=cp.uint8)
    img2 = cp.zeros((150, 150), dtype=cp.uint8)
    for angle in range(0, 180, 15):
        img1.fill(0)
        rr, cc = ellipse(80, 70, 60, 40, rotation=cp.deg2rad(angle))
        img1[rr, cc] = 1
        img2.fill(0)
        rr, cc = ellipse(80, 70, 60, 40, rotation=cp.deg2rad(angle + 180))
        img2[rr, cc] = 1
        assert_array_equal(img1, img2)


def test_ellipse_rotated():
    img = cp.zeros((1000, 1200), dtype=cp.uint8)
    for rot in range(0, 180, 10):
        img.fill(0)
        angle = cp.deg2rad(rot)
        rr, cc = ellipse(500, 600, 200, 400, rotation=angle)
        img[rr, cc] = 1
        # estimate orientation of ellipse
        angle_estim_raw = regionprops(img)[0].orientation
        angle_estim = cp.round(angle_estim_raw, 3) % (cp.pi / 2)
        assert_almost_equal(angle_estim, angle % (cp.pi / 2), 2)


# def test_ellipse_perimeter_dot_zeroangle():
#     # dot, angle == 0
#     img = cp.zeros((30, 15), 'uint8')
#     rr, cc = ellipse_perimeter(15, 7, 0, 0, 0)
#     img[rr, cc] = 1
#     assert(cp.sum(img) == 1)
#     assert(img[15][7] == 1)
#
#
# def test_ellipse_perimeter_dot_nzeroangle():
#     # dot, angle != 0
#     img = cp.zeros((30, 15), 'uint8')
#     rr, cc = ellipse_perimeter(15, 7, 0, 0, 1)
#     img[rr, cc] = 1
#     assert(cp.sum(img) == 1)
#     assert(img[15][7] == 1)
#
#
# def test_ellipse_perimeter_flat_zeroangle():
#     # flat ellipse
#     img = cp.zeros((20, 18), 'uint8')
#     img_ = cp.zeros((20, 18), 'uint8')
#     rr, cc = ellipse_perimeter(6, 7, 0, 5, 0)
#     img[rr, cc] = 1
#     rr, cc = line(6, 2, 6, 12)
#     img_[rr, cc] = 1
#     assert_array_equal(img, img_)
#
#
# def test_ellipse_perimeter_zeroangle():
#     # angle == 0
#     img = cp.zeros((30, 15), 'uint8')
#     rr, cc = ellipse_perimeter(15, 7, 14, 6, 0)
#     img[rr, cc] = 1
#     img_ = cp.array(
#         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
#     )
#
#     assert_array_equal(img, img_)
#
#
# def test_ellipse_perimeter_nzeroangle():
#     # angle != 0
#     img = cp.zeros((30, 25), 'uint8')
#     rr, cc = ellipse_perimeter(15, 11, 12, 6, 1.1)
#     img[rr, cc] = 1
#     img_ = cp.array(
#         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#     )
#     assert_array_equal(img, img_)
#
#
# def test_ellipse_perimeter_shape():
#     img = cp.zeros((15, 20), 'uint8')
#     rr, cc = ellipse_perimeter(7, 10, 9, 9, 0, shape=(15, 20))
#     img[rr, cc] = 1
#     shift = 5
#     img_ = cp.zeros((15 + 2 * shift, 20), 'uint8')
#     rr, cc = ellipse_perimeter(7 + shift, 10, 9, 9, 0, shape=None)
#     img_[rr, cc] = 1
#     assert_array_equal(img, img_[shift:-shift, :])
#
#
# def test_bezier_segment_straight():
#     image = cp.zeros((200, 200), dtype=int)
#     r0, r1, r2 = 50, 150, 150
#     c0, c1, c2 = 50, 50, 150
#     rr, cc = _bezier_segment(r0, c0, r1, c1, r2, c2, 0)
#     image[rr, cc] = 1
#
#     image2 = cp.zeros((200, 200), dtype=int)
#     rr, cc = line(r0, c0, r2, c2)
#     image2[rr, cc] = 1
#     assert_array_equal(image, image2)
#
#
# def test_bezier_segment_curved():
#     img = cp.zeros((25, 25), 'uint8')
#     r0, c0 = 20, 20
#     r1, c1 = 20, 2
#     r2, c2 = 2, 2
#     rr, cc = _bezier_segment(r0, c0, r1, c1, r2, c2, 1)
#     img[rr, cc] = 1
#     img_ = cp.array(
#         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#     )
#     assert_equal(img[r0, c0], 1)
#     assert_equal(img[r2, c2], 1)
#     assert_array_equal(img, img_)
#
#
# def test_bezier_curve_straight():
#     image = cp.zeros((200, 200), dtype=int)
#     r0, c0 = 50, 50
#     r1, c1 = 150, 50
#     r2, c2 = 150, 150
#     rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 0)
#     image[rr, cc] = 1
#
#     image2 = cp.zeros((200, 200), dtype=int)
#     rr, cc = line(r0, c0, r2, c2)
#     image2[rr, cc] = 1
#     assert_array_equal(image, image2)
#
#
# def test_bezier_curved_weight_eq_1():
#     img = cp.zeros((23, 8), 'uint8')
#     r0, c0 = 1, 1
#     r1, c1 = 11, 11
#     r2, c2 = 21, 1
#     rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 1)
#     img[rr, cc] = 1
#     assert_equal(img[r0, c0], 1)
#     assert_equal(img[r2, c2], 1)
#     img_ = cp.array(
#         [[0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0]]
#     )
#     assert_equal(img, img_)
#
#
# def test_bezier_curved_weight_neq_1():
#     img = cp.zeros((23, 10), 'uint8')
#     r0, c0 = 1, 1
#     r1, c1 = 11, 11
#     r2, c2 = 21, 1
#     rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 2)
#     img[rr, cc] = 1
#     assert_equal(img[r0, c0], 1)
#     assert_equal(img[r2, c2], 1)
#     img_ = cp.array(
#         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#     )
#     assert_equal(img, img_)
#
#
# def test_bezier_curve_shape():
#     img = cp.zeros((15, 20), 'uint8')
#     r0, c0 = 1, 5
#     r1, c1 = 6, 11
#     r2, c2 = 1, 14
#     rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 2, shape=(15, 20))
#     img[rr, cc] = 1
#     shift = 5
#     img_ = cp.zeros((15 + 2 * shift, 20), 'uint8')
#     r0, c0 = 1 + shift, 5
#     r1, c1 = 6 + shift, 11
#     r2, c2 = 1 + shift, 14
#     rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 2, shape=None)
#     img_[rr, cc] = 1
#     assert_array_equal(img, img_[shift:-shift, :])
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_polygon_perimeter():
#     expected = cp.array(
#         [[1, 1, 1, 1],
#          [1, 0, 0, 1],
#          [1, 1, 1, 1]]
#     )
#     out = cp.zeros_like(expected)
#
#     rr, cc = polygon_perimeter([0, 2, 2, 0],
#                                [0, 0, 3, 3])
#
#     out[rr, cc] = 1
#     assert_array_equal(out, expected)
#
#     out = cp.zeros_like(expected)
#     rr, cc = polygon_perimeter([-1, -1, 3,  3],
#                                [-1,  4, 4, -1],
#                                shape=out.shape, clip=True)
#     out[rr, cc] = 1
#     assert_array_equal(out, expected)
#
#     with pytest.raises(ValueError):
#         polygon_perimeter([0], [1], clip=True)
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_polygon_perimeter_outside_image():
#     rr, cc = polygon_perimeter([-1, -1, 3,  3],
#                                [-1,  4, 4, -1], shape=(3, 4))
#     assert_equal(len(rr), 0)
#     assert_equal(len(cc), 0)
#
#
# def test_rectangle_end():
#     expected = cp.array([[0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 0, 0, 0, 0]], dtype=cp.uint8)
#     start = (0, 1)
#     end = (3, 3)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(start, end=end, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     # Swap start and end
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(end=start, start=end, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     # Bottom left and top right
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(start=(3, 1), end=(0, 3), shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(end=(3, 1), start=(0, 3), shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#
# def test_rectangle_float_input():
#     expected = cp.array([[0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 0, 0, 0, 0]], dtype=cp.uint8)
#     start = (0.2, 0.8)
#     end = (3.1, 2.9)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(start, end=end, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     # Swap start and end
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(end=start, start=end, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     # Bottom left and top right
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(start=(3.1, 0.8), end=(0.2, 2.9), shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(end=(3.1, 0.8), start=(0.2, 2.9), shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#
# def test_rectangle_extent():
#     expected = cp.array([[0, 0, 0, 0, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 0, 0, 0, 0]], dtype=cp.uint8)
#     start = (1, 1)
#     extent = (3, 3)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle(start, extent=extent, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     img = cp.zeros((5, 5, 3), dtype=cp.uint8)
#     rr, cc = rectangle(start, extent=extent, shape=img.shape)
#     img[rr, cc, 0] = 1
#     expected_2 = cp.zeros_like(img)
#     expected_2[..., 0] = expected
#     assert_array_equal(img, expected_2)
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_rectangle_extent_negative():
#     # These two tests should be done together.
#     expected = cp.array([[0, 0, 0, 0, 0, 0],
#                          [0, 0, 1, 1, 1, 1],
#                          [0, 0, 1, 2, 2, 1],
#                          [0, 0, 1, 1, 1, 1],
#                          [0, 0, 0, 0, 0, 0]], dtype=cp.uint8)
#
#     start = (3, 5)
#     extent = (-1, -2)
#     img = cp.zeros(expected.shape, dtype=cp.uint8)
#     rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape)
#     img[rr, cc] = 1
#
#     rr, cc = rectangle(start, extent=extent, shape=img.shape)
#     img[rr, cc] = 2
#     assert_array_equal(img, expected)
#
#     # Ensure that rr and cc have no overlap
#     img = cp.zeros(expected.shape, dtype=cp.uint8)
#     rr, cc = rectangle(start, extent=extent, shape=img.shape)
#     img[rr, cc] = 2
#
#     rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_rectangle_perimiter():
#     expected = cp.array([[0, 0, 0, 0, 0, 0],
#                          [0, 0, 1, 1, 1, 1],
#                          [0, 0, 1, 0, 0, 1],
#                          [0, 0, 1, 1, 1, 1],
#                          [0, 0, 0, 0, 0, 0]], dtype=cp.uint8)
#     start = (2, 3)
#     end = (2, 4)
#     img = cp.zeros(expected.shape, dtype=cp.uint8)
#     # Test that the default parameter is indeed end
#     rr, cc = rectangle_perimeter(start, end, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     # Swap start and end
#     img = cp.zeros(expected.shape, dtype=cp.uint8)
#     rr, cc = rectangle_perimeter(end=start, start=end, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     img = cp.zeros(expected.shape, dtype=cp.uint8)
#     start = (2, 3)
#     extent = (1, 2)
#     rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_rectangle_perimiter_clip_bottom_right():
#     # clip=False
#     expected = cp.array([[0, 0, 0, 0, 0],
#                          [0, 1, 1, 1, 1],
#                          [0, 1, 0, 0, 0],
#                          [0, 1, 0, 0, 0],
#                          [0, 1, 0, 0, 0]], dtype=cp.uint8)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     start = (2, 2)
#     extent = (10, 10)
#     rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape,
#                                  clip=False)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     # clip=True
#     expected = cp.array([[0, 0, 0, 0, 0],
#                          [0, 1, 1, 1, 1],
#                          [0, 1, 0, 0, 1],
#                          [0, 1, 0, 0, 1],
#                          [0, 1, 1, 1, 1]], dtype=cp.uint8)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape,
#                                  clip=True)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_rectangle_perimiter_clip_top_left():
#     # clip=False
#     expected = cp.array([[0, 0, 0, 1, 0],
#                          [0, 0, 0, 1, 0],
#                          [0, 0, 0, 1, 0],
#                          [1, 1, 1, 1, 0],
#                          [0, 0, 0, 0, 0]], dtype=cp.uint8)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     start = (-5, -5)
#     end = (2, 2)
#     rr, cc = rectangle_perimeter(start, end=end, shape=img.shape,
#                                  clip=False)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     # clip=True
#     expected = cp.array([[1, 1, 1, 1, 0],
#                          [1, 0, 0, 1, 0],
#                          [1, 0, 0, 1, 0],
#                          [1, 1, 1, 1, 0],
#                          [0, 0, 0, 0, 0]], dtype=cp.uint8)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle_perimeter(start, end=end, shape=img.shape,
#                                  clip=True)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_rectangle_perimiter_clip_top_right():
#     expected = cp.array([[0, 1, 1, 1, 1],
#                          [0, 1, 0, 0, 1],
#                          [0, 1, 0, 0, 1],
#                          [0, 1, 1, 1, 1],
#                          [0, 0, 0, 0, 0]], dtype=cp.uint8)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     start = (-10, 2)
#     end = (2, 10)
#     rr, cc = rectangle_perimeter(start, end=end, shape=img.shape,
#                                  clip=True)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     expected = cp.array([[0, 1, 0, 0, 0],
#                          [0, 1, 0, 0, 0],
#                          [0, 1, 0, 0, 0],
#                          [0, 1, 1, 1, 1],
#                          [0, 0, 0, 0, 0]], dtype=cp.uint8)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle_perimeter(start, end=end, shape=img.shape,
#                                  clip=False)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#
# @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
# def test_rectangle_perimiter_clip_bottom_left():
#     expected = cp.array([[0, 0, 0, 0, 0],
#                          [1, 1, 1, 0, 0],
#                          [1, 0, 1, 0, 0],
#                          [1, 0, 1, 0, 0],
#                          [1, 1, 1, 0, 0]], dtype=cp.uint8)
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     start = (2, -3)
#     end = (10, 1)
#     rr, cc = rectangle_perimeter(start, end=end, shape=img.shape,
#                                  clip=True)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
#
#     expected = cp.array([[0, 0, 0, 0, 0],
#                          [1, 1, 1, 0, 0],
#                          [0, 0, 1, 0, 0],
#                          [0, 0, 1, 0, 0],
#                          [0, 0, 1, 0, 0]], dtype=cp.uint8)
#
#     img = cp.zeros((5, 5), dtype=cp.uint8)
#     rr, cc = rectangle_perimeter(start, end=end, shape=img.shape,
#                                  clip=False)
#     img[rr, cc] = 1
#     assert_array_equal(img, expected)
