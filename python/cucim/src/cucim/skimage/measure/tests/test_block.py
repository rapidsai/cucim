import cupy as cp
import pytest
from cupy.testing import assert_array_equal

from cucim.skimage.measure import block_reduce


def test_block_reduce_sum():
    image1 = cp.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3))
    # fmt: off
    expected1 = cp.array([[24,  42],   # noqa
                          [96, 114]])
    # fmt: on
    assert_array_equal(expected1, out1)

    image2 = cp.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (3, 3))
    # fmt: off
    expected2 = cp.array([[ 81, 108,  87],   # noqa
                          [174, 192, 138]])
    # fmt: on
    assert_array_equal(expected2, out2)


def test_block_reduce_mean():
    image1 = cp.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=cp.mean)
    # fmt: off
    expected1 = cp.array([[ 4.,  7.],   # noqa
                          [16., 19.]])
    # fmt: on
    assert_array_equal(expected1, out1)

    image2 = cp.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=cp.mean)
    # fmt: off
    expected2 = cp.array([[14. , 10.8],
                          [ 8.5,  5.7]])  # noqa
    # fmt: on
    assert_array_equal(expected2, out2)


def test_block_reduce_median():
    image1 = cp.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=cp.median)
    # fmt: off
    expected1 = cp.array([[ 4.,  7.],   # noqa
                          [16., 19.]])
    # fmt: on
    assert_array_equal(expected1, out1)

    image2 = cp.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=cp.median)
    # fmt: off
    expected2 = cp.array([[14., 6.5],   # noqa
                          [ 0., 0. ]])  # noqa
    # fmt: on
    assert_array_equal(expected2, out2)

    image3 = cp.array([[1, 5, 5, 5], [5, 5, 5, 1000]])
    out3 = block_reduce(image3, (2, 4), func=cp.median)
    assert_array_equal(5, out3)


def test_block_reduce_min():
    image1 = cp.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=cp.min)
    # fmt: off
    expected1 = cp.array([[ 0, 3],    # noqa
                          [12, 15]])
    # fmt: on
    assert_array_equal(expected1, out1)

    image2 = cp.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=cp.min)
    # fmt: off
    expected2 = cp.array([[0, 0],
                          [0, 0]])
    # fmt: on
    assert_array_equal(expected2, out2)


def test_block_reduce_max():
    image1 = cp.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=cp.max)
    # fmt: off
    expected1 = cp.array([[ 8, 11],   # noqa
                          [20, 23]])
    # fmt: on
    assert_array_equal(expected1, out1)

    image2 = cp.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=cp.max)
    # fmt: off
    expected2 = cp.array([[28, 31],
                          [36, 39]])
    # fmt: on
    assert_array_equal(expected2, out2)


def test_invalid_block_size():
    image = cp.arange(4 * 6).reshape(4, 6)

    with pytest.raises(ValueError):
        block_reduce(image, [1, 2, 3])
    with pytest.raises(ValueError):
        block_reduce(image, [1, 0.5])


def test_default_block_size():
    image = cp.arange(4 * 6).reshape(4, 6)
    out = block_reduce(image, func=cp.min)
    expected = cp.array([[0, 2, 4],
                         [12, 14, 16]])
    assert_array_equal(expected, out)


def test_scalar_block_size():
    image = cp.arange(6 * 6).reshape(6, 6)
    out = block_reduce(image, 3, func=cp.min)
    expected1 = cp.array([[0, 3],
                         [18, 21]])
    assert_array_equal(expected1, out)
    expected2 = block_reduce(image, (3, 3), func=cp.min)
    assert_array_equal(expected2, out)


@pytest.mark.skip(reason="cupy.mean doesn't support setting dtype=cupy.uint8")
def test_func_kwargs_same_dtype():
    # fmt: off
    image = cp.array([[97, 123, 173, 227],
                     [217, 241, 221, 214],
                     [211,  11, 170,  53],                   # noqa
                     [214, 205, 101,  57]], dtype=cp.uint8)  # noqa
    # fmt: on

    out = block_reduce(
        image, (2, 2), func=cp.mean, func_kwargs={"dtype": cp.uint8}
    )
    expected = cp.array([[41, 16], [32, 31]], dtype=cp.uint8)

    assert_array_equal(out, expected)
    assert out.dtype == expected.dtype


def test_func_kwargs_different_dtype():
    # fmt: off
    image = cp.array([[0.45745366, 0.67479345, 0.20949775, 0.3147348],
                      [0.7209286, 0.88915504, 0.66153409, 0.07919526],
                      [0.04640037, 0.54008495, 0.34664343, 0.56152301],
                      [0.58085003, 0.80144708, 0.87844473, 0.29811511]],
                     dtype=cp.float64)
    # fmt: on

    out = block_reduce(image, (2, 2), func=cp.mean,
                       func_kwargs={'dtype': cp.float16})
    expected = cp.array([[0.6855, 0.3164], [0.4922, 0.521]], dtype=cp.float16)

    # Note: had to set decimal=3 for float16 to pass here when using CuPy
    cp.testing.assert_array_almost_equal(out, expected, decimal=3)
    assert out.dtype == expected.dtype
