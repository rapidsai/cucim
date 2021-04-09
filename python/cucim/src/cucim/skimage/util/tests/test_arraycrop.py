import cupy as cp
from cupy.testing import assert_array_equal

from cucim.skimage.util import crop


def test_multi_crop():
    arr = cp.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2), (2, 1)))
    assert_array_equal(out[0], [7, 8])
    assert_array_equal(out[-1], [32, 33])
    assert out.shape == (6, 2)


def test_pair_crop():
    arr = cp.arange(45).reshape(9, 5)
    out = crop(arr, (1, 2))
    assert_array_equal(out[0], [6, 7])
    assert_array_equal(out[-1], [31, 32])
    assert out.shape == (6, 2)


def test_pair_tuple_crop():
    arr = cp.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2),))
    assert_array_equal(out[0], [6, 7])
    assert_array_equal(out[-1], [31, 32])
    assert out.shape == (6, 2)


def test_int_crop():
    arr = cp.arange(45).reshape(9, 5)
    out = crop(arr, 1)
    assert_array_equal(out[0], [6, 7, 8])
    assert_array_equal(out[-1], [36, 37, 38])
    assert out.shape == (7, 3)


def test_int_tuple_crop():
    arr = cp.arange(45).reshape(9, 5)
    out = crop(arr, (1,))
    assert_array_equal(out[0], [6, 7, 8])
    assert_array_equal(out[-1], [36, 37, 38])
    assert out.shape == (7, 3)


def test_copy_crop():
    arr = cp.arange(45).reshape(9, 5)
    out0 = crop(arr, 1, copy=True)
    assert out0.flags.c_contiguous
    out0[0, 0] = 100
    assert not cp.any(arr == 100)
    assert not cp.may_share_memory(arr, out0)

    out1 = crop(arr, 1)
    out1[0, 0] = 100
    assert arr[1, 1] == 100
    assert cp.may_share_memory(arr, out1)


def test_zero_crop():
    arr = cp.arange(45).reshape(9, 5)
    out = crop(arr, 0)
    assert out.shape == (9, 5)
