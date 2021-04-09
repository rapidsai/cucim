import cupy as cp
import pytest

from cucim.skimage._shared.testing import expected_warnings
from cucim.skimage.util.shape import view_as_blocks, view_as_windows


def test_view_as_blocks_block_not_a_tuple():
    A = cp.arange(10)
    with pytest.raises(TypeError):
        view_as_blocks(A, [5])


def test_view_as_blocks_negative_shape():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_blocks(A, (-2,))


def test_view_as_blocks_block_too_large():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_blocks(A, (11,))


def test_view_as_blocks_wrong_block_dimension():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_blocks(A, (2, 2))


def test_view_as_blocks_1D_array_wrong_block_shape():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_blocks(A, (3,))


def test_view_as_blocks_1D_array():
    A = cp.arange(10)
    B = view_as_blocks(A, (5,))
    # fmt: off
    cp.testing.assert_array_equal(
        B, cp.array([[0, 1, 2, 3, 4],
                     [5, 6, 7, 8, 9]])
    )
    # fmt: on


def test_view_as_blocks_2D_array():
    A = cp.arange(4 * 4).reshape(4, 4)
    B = view_as_blocks(A, (2, 2))
    # fmt: off
    cp.testing.assert_array_equal(
        B[0, 1], cp.array([[2, 3],
                           [6, 7]])
    )
    # fmt: on
    assert B[1, 0, 1, 1] == 13


def test_view_as_blocks_3D_array():
    A = cp.arange(4 * 4 * 6).reshape(4, 4, 6)
    B = view_as_blocks(A, (1, 2, 2))
    assert B.shape == (4, 2, 3, 1, 2, 2)
    # fmt: off
    cp.testing.assert_array_equal(
        B[2:, 0, 2], cp.array([[[[52, 53],
                                 [58, 59]]],
                               [[[76, 77],
                                 [82, 83]]]])
    )
    # fmt: on


def test_view_as_windows_input_not_array():
    A = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        view_as_windows(A, (2,))


def test_view_as_windows_wrong_window_dimension():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_windows(A, (2, 2))


def test_view_as_windows_negative_window_length():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_windows(A, (-1,))


def test_view_as_windows_window_too_large():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_windows(A, (11,))


def test_view_as_windows_step_below_one():
    A = cp.arange(10)
    with pytest.raises(ValueError):
        view_as_windows(A, (11,), step=0.9)


def test_view_as_windows_1D():
    A = cp.arange(10)
    window_shape = (3,)
    B = view_as_windows(A, window_shape)
    # fmt: off
    cp.testing.assert_array_equal(
        B, cp.array([[0, 1, 2],
                     [1, 2, 3],
                     [2, 3, 4],
                     [3, 4, 5],
                     [4, 5, 6],
                     [5, 6, 7],
                     [6, 7, 8],
                     [7, 8, 9]])
    )
    # fmt: on


def test_view_as_windows_2D():
    A = cp.arange(5 * 4).reshape(5, 4)
    window_shape = (4, 3)
    B = view_as_windows(A, window_shape)
    assert B.shape == (2, 2, 4, 3)
    # fmt: off
    cp.testing.assert_array_equal(
        B, cp.array([[[[0,  1,  2],
                       [4,  5,  6],
                       [8,  9, 10],
                       [12, 13, 14]],
                      [[1,  2,  3],
                       [5,  6,  7],
                       [9, 10, 11],
                       [13, 14, 15]]],
                     [[[4,  5,  6],
                       [8,  9, 10],
                       [12, 13, 14],
                       [16, 17, 18]],
                      [[5,  6,  7],
                       [9, 10, 11],
                       [13, 14, 15],
                       [17, 18, 19]]]]))
    # fmt: on


def test_view_as_windows_with_skip():
    A = cp.arange(20).reshape((5, 4))
    B = view_as_windows(A, 2, step=2)
    # fmt: off
    cp.testing.assert_array_equal(
        B, [[[[0, 1],
              [4, 5]],
             [[2, 3],
              [6, 7]]],
            [[[8, 9],
              [12, 13]],
             [[10, 11],
              [14, 15]]]]
    )
    # fmt: on
    C = view_as_windows(A, 2, step=4)
    assert C.shape == (1, 1, 2, 2)


def test_views_non_contiguous():
    A = cp.arange(16).reshape((4, 4))
    A = A[::2, :]

    with expected_warnings(["Cannot provide views"]):
        res_b = view_as_blocks(A, (2, 2))
    res_w = view_as_windows(A, (2, 2))
    print(res_b)
    print(res_w)
    # fmt: off
    expected_b = [[[[0,  1],
                    [8,  9]],
                   [[2,  3],
                    [10, 11]]]]

    expected_w = [[[[ 0,  1],
                    [ 8,  9]],
                   [[ 1,  2],
                    [ 9, 10]],
                   [[ 2,  3],
                    [10, 11]]]]
    # fmt: on
    cp.testing.assert_array_equal(res_b, expected_b)
    cp.testing.assert_array_equal(res_w, expected_w)


def test_view_as_windows_step_tuple():
    A = cp.arange(24).reshape((6, 4))
    B = view_as_windows(A, (3, 2), step=3)
    assert B.shape == (2, 1, 3, 2)
    assert B.size != A.size

    C = view_as_windows(A, (3, 2), step=(3, 2))
    assert C.shape == (2, 2, 3, 2)
    assert C.size == A.size

    # fmt: off
    cp.testing.assert_array_equal(
        C, [[[[0,  1],
              [4,  5],
              [8,  9]],
             [[2,  3],
              [6,  7],
              [10, 11]]],
            [[[12, 13],
              [16, 17],
              [20, 21]],
             [[14, 15],
              [18, 19],
              [22, 23]]]]
    )
    # fmt: on
