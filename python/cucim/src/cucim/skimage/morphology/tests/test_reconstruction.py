"""
These tests are originally part of CellProfiler, code licensed under both GPL
and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""
import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal
from skimage.morphology import reconstruction as reconstruction_cpu

from cucim.skimage.morphology import reconstruction


def test_zeros():
    """Test reconstruction with image and mask of zeros"""
    assert_array_almost_equal(
        reconstruction(cp.zeros((5, 7)), cp.zeros((5, 7))), 0)


def test_image_equals_mask():
    """Test reconstruction where the image and mask are the same"""
    assert_array_almost_equal(
        reconstruction(cp.ones((7, 5)), cp.ones((7, 5))), 1)


def test_image_less_than_mask():
    """Test reconstruction where the image is uniform and less than mask"""
    image = cp.ones((5, 5))
    mask = cp.ones((5, 5)) * 2
    assert_array_almost_equal(reconstruction(image, mask), 1)


def test_one_image_peak():
    """Test reconstruction with one peak pixel"""
    image = cp.ones((5, 5))
    image[2, 2] = 2
    mask = cp.ones((5, 5)) * 3
    assert_array_almost_equal(reconstruction(image, mask), 2)


def test_two_image_peaks():
    """Test reconstruction with two peak pixels isolated by the mask"""
    # fmt: off
    image = cp.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 3, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

    mask = cp.array([[4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4]])

    expected = cp.array([[2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3]])
    # fmt: on
    assert_array_almost_equal(reconstruction(image, mask), expected)


def test_zero_image_one_mask():
    """Test reconstruction with an image of all zeros and a mask that's not"""
    result = reconstruction(cp.zeros((10, 10)), cp.ones((10, 10)))
    assert_array_almost_equal(result, 0)


def test_fill_hole():
    """Test reconstruction by erosion, which should fill holes in mask."""
    seed = cp.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0])
    mask = cp.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    result = reconstruction(seed, mask, method="erosion")
    assert_array_almost_equal(
        result, cp.array([0, 3, 6, 4, 4, 4, 4, 4, 2, 0])
    )


def test_invalid_seed():
    seed = cp.ones((5, 5))
    mask = cp.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed * 2, mask, method='dilation')
    with pytest.raises(ValueError):
        reconstruction(seed * 0.5, mask, method='erosion')


def test_invalid_selem():
    seed = cp.ones((5, 5))
    mask = cp.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, selem=np.ones((4, 4)))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, selem=np.ones((3, 4)))
    reconstruction(seed, mask, selem=np.ones((3, 3)))


def test_invalid_method():
    seed = cp.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0])
    mask = cp.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    with pytest.raises(ValueError):
        reconstruction(seed, mask, method="foo")


def test_invalid_offset_not_none():
    """Test reconstruction with invalid not None offset parameter"""
    # fmt: off
    image = cp.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 3, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

    mask = cp.array([[4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4]])
    # fmt: on
    with pytest.raises(ValueError):
        reconstruction(image, mask, method='dilation',
                       selem=cp.ones((3, 3)), offset=cp.array([3, 0]))


def test_offset_not_none():
    """Test reconstruction with valid offset parameter"""
    seed = cp.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    mask = cp.array([0, 8, 6, 8, 8, 8, 8, 4, 4, 0])
    expected = cp.array([0, 3, 6, 6, 6, 6, 6, 4, 4, 0])

    assert_array_almost_equal(
        reconstruction(seed, mask, method='dilation',
                       selem=cp.ones(3), offset=cp.array([0])), expected)


def test_reconstruction_float_inputs():
    """Verifies fix for: https://github.com/rapidsai/cuci/issues/36

    Run the 2D example from the reconstruction docstring and compare the output
    to scikit-image.
    """

    y, x = np.mgrid[:20:0.5, :20:0.5]
    bumps = np.sin(x) + np.sin(y)
    h = 0.3
    seed = bumps - h
    background_cpu = reconstruction_cpu(seed, bumps)
    background = reconstruction(cp.asarray(seed), cp.asarray(bumps))
    cp.testing.assert_allclose(background, background_cpu)
