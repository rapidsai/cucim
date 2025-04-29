import numpy as np
from numpy.testing import assert_equal

from cucim.skimage.util import regular_grid


def test_regular_grid_full():
    ar = np.zeros((2, 2))
    g = regular_grid(ar.shape, 25)
    assert_equal(g, [slice(None, None, None), slice(None, None, None)])
    ar[g] = 1
    assert_equal(ar.size, ar.sum())


def test_regular_grid_2d_8():
    ar = np.zeros((20, 40))
    g = regular_grid(ar.shape, 8)
    assert_equal(g, [slice(5.0, None, 10.0), slice(5.0, None, 10.0)])
    ar[g] = 1
    assert_equal(ar.sum(), 8)


def test_regular_grid_2d_32():
    ar = np.zeros((20, 40))
    g = regular_grid(ar.shape, 32)
    assert_equal(g, [slice(2.0, None, 5.0), slice(2.0, None, 5.0)])
    ar[g] = 1
    assert_equal(ar.sum(), 32)


def test_regular_grid_3d_8():
    ar = np.zeros((3, 20, 40))
    g = regular_grid(ar.shape, 8)
    assert_equal(
        g,
        [slice(1.0, None, 3.0), slice(5.0, None, 10.0), slice(5.0, None, 10.0)],
    )
    ar[g] = 1
    assert_equal(ar.sum(), 8)


def test_regular_grid_3d_100_channel_axis_2():
    ar = np.zeros((512, 512, 3))
    g = regular_grid(ar.shape, 100)
    assert_equal(
        g, [slice(25, None, 51), slice(25, None, 51), slice(1, None, 3)]
    )
    ar[g] = 1
    assert_equal(ar.sum(), 100)


def test_regular_grid_3d_100_channel_axis_1():
    ar = np.zeros((512, 3, 512))
    g = regular_grid(ar.shape, 100)
    assert_equal(
        g, [slice(25, None, 51), slice(1, None, 3), slice(25, None, 51)]
    )
    ar[g] = 1
    assert_equal(ar.sum(), 100)


def test_regular_grid_3d_100_channel_axis_1_v2():
    ar = np.zeros((512, 3, 256))
    g = regular_grid(ar.shape, 100)
    assert_equal(
        g, [slice(18, None, 36), slice(1, None, 3), slice(18, None, 36)]
    )
    ar[g] = 1
    # equals 98, not 100 as requested
    # (this is also true for scikit-image as of 0.25.2)
    assert ar.sum() >= 98
    assert ar.sum() <= 100
