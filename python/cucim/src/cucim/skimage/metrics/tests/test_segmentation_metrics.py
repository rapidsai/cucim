import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from numpy.testing import assert_almost_equal, assert_equal

from cucim.skimage.metrics import (adapted_rand_error, contingency_table,
                                   variation_of_information)


def test_contingency_table():
    im_true = cp.array([1, 2, 3, 4])
    im_test = cp.array([1, 1, 8, 8])

    # fmt: off
    table1 = cp.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0.25],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0.25]])
    # fmt: on

    sparse_table2 = contingency_table(im_true, im_test, normalize=True)
    table2 = sparse_table2.toarray()
    assert_array_equal(table1, table2)


def test_vi():
    im_true = cp.array([1, 2, 3, 4])
    im_test = cp.array([1, 1, 8, 8])
    assert_equal(float(cp.sum(variation_of_information(im_true, im_test))), 1)


def test_vi_ignore_labels():
    im1 = cp.array([[1, 0],
                    [2, 3]], dtype='uint8')
    im2 = cp.array([[1, 1],
                    [1, 0]], dtype='uint8')

    false_splits, false_merges = variation_of_information(im1, im2,
                                                          ignore_labels=[0])
    assert (false_splits, false_merges) == (0, 2 / 3)


def test_are():
    im_true = cp.array([[2, 1], [1, 2]])
    im_test = cp.array([[1, 2], [3, 1]])
    assert_almost_equal(
        tuple(map(float, adapted_rand_error(im_true, im_test))),
        (0.3333333, 0.5, 1.0)
    )
    assert_almost_equal(
        tuple(map(float, adapted_rand_error(im_true, im_test, alpha=0))),
        (0, 0.5, 1.0)
    )
    assert_almost_equal(
        tuple(map(float, adapted_rand_error(im_true, im_test, alpha=1))),
        (0.5, 0.5, 1.0)
    )

    with pytest.raises(ValueError):
        adapted_rand_error(im_true, im_test, alpha=1.01)
    with pytest.raises(ValueError):
        adapted_rand_error(im_true, im_test, alpha=-0.01)
