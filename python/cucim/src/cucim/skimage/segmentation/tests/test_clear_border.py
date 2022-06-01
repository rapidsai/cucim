import cupy as cp
from cupy.testing import assert_array_equal

from cucim.skimage.segmentation import clear_border


def test_clear_border():
    image = cp.array(
        [[0, 0, 0, 0, 0, 0, 0, 1, 0],
         [1, 1, 0, 0, 1, 0, 0, 1, 0],
         [1, 1, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    # test default case
    result = clear_border(image.copy())
    ref = image.copy()
    ref[1:3, 0:2] = 0
    ref[0:2, -2] = 0
    assert_array_equal(result, ref)

    # test buffer
    result = clear_border(image.copy(), 1)
    assert_array_equal(result, cp.zeros(result.shape))

    # test background value
    result = clear_border(image.copy(), buffer_size=1, bgval=2)
    assert_array_equal(result, cp.full_like(image, 2))

    # test mask
    mask = cp.array([[0, 0, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
    result = clear_border(image.copy(), mask=mask)
    ref = image.copy()
    ref[1:3, 0:2] = 0
    assert_array_equal(result, ref)


def test_clear_border_3d():
    image = cp.array(
        [[[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [1, 0, 0, 0]],
         [[0, 0, 0, 0],
          [0, 1, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 0]],
         [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]]
    )
    # test default case
    result = clear_border(image.copy())
    ref = image.copy()
    ref[0, 3, 0] = 0
    assert_array_equal(result, ref)

    # test buffer
    result = clear_border(image.copy(), 1)
    assert_array_equal(result, cp.zeros(result.shape))

    # test background value
    result = clear_border(image.copy(), buffer_size=1, bgval=2)
    assert_array_equal(result, cp.full_like(image, 2))

    # test floating-point background value
    image_f32 = image.astype(cp.float32)
    result = clear_border(image_f32, buffer_size=1, bgval=2.5)
    assert_array_equal(result, cp.full_like(image_f32, 2.5))


def test_clear_border_non_binary():
    image = cp.array([[1, 2, 3, 1, 2],
                      [3, 3, 5, 4, 2],
                      [3, 4, 5, 4, 2],
                      [3, 3, 2, 1, 2]])

    result = clear_border(image)
    expected = cp.array([[0, 0, 0, 0, 0],
                         [0, 0, 5, 4, 0],
                         [0, 4, 5, 4, 0],
                         [0, 0, 0, 0, 0]])

    assert_array_equal(result, expected)
    assert not cp.all(image == result)


def test_clear_border_non_binary_3d():
    image3d = cp.array(
        [[[1, 2, 3, 1, 2],
          [3, 3, 3, 4, 2],
          [3, 4, 3, 4, 2],
          [3, 3, 2, 1, 2]],
         [[1, 2, 3, 1, 2],
          [3, 3, 5, 4, 2],
          [3, 4, 5, 4, 2],
          [3, 3, 2, 1, 2]],
         [[1, 2, 3, 1, 2],
          [3, 3, 3, 4, 2],
          [3, 4, 3, 4, 2],
          [3, 3, 2, 1, 2]]]
    )

    result = clear_border(image3d)
    expected = cp.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 5, 0, 0],
          [0, 0, 5, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]]
    )

    assert_array_equal(result, expected)
    assert not cp.all(image3d == result)


def test_clear_border_non_binary_inplace():
    image = cp.array([[1, 2, 3, 1, 2],
                      [3, 3, 5, 4, 2],
                      [3, 4, 5, 4, 2],
                      [3, 3, 2, 1, 2]])
    result = clear_border(image, out=image)
    expected = cp.array([[0, 0, 0, 0, 0],
                         [0, 0, 5, 4, 0],
                         [0, 4, 5, 4, 0],
                         [0, 0, 0, 0, 0]])

    assert_array_equal(result, expected)
    assert_array_equal(image, result)


def test_clear_border_non_binary_inplace_3d():
    image3d = cp.array(
        [[[1, 2, 3, 1, 2],
          [3, 3, 3, 4, 2],
          [3, 4, 3, 4, 2],
          [3, 3, 2, 1, 2]],
         [[1, 2, 3, 1, 2],
          [3, 3, 5, 4, 2],
          [3, 4, 5, 4, 2],
          [3, 3, 2, 1, 2]],
         [[1, 2, 3, 1, 2],
          [3, 3, 3, 4, 2],
          [3, 4, 3, 4, 2],
          [3, 3, 2, 1, 2]]]
    )

    result = clear_border(image3d, out=image3d)
    expected = cp.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 5, 0, 0],
          [0, 0, 5, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]]
    )

    assert_array_equal(result, expected)
    assert_array_equal(image3d, result)


def test_clear_border_non_binary_out():
    image = cp.array([[1, 2, 3, 1, 2],
                      [3, 3, 5, 4, 2],
                      [3, 4, 5, 4, 2],
                      [3, 3, 2, 1, 2]])
    out = cp.empty_like(image)
    result = clear_border(image, out=out)
    expected = cp.array([[0, 0, 0, 0, 0],
                         [0, 0, 5, 4, 0],
                         [0, 4, 5, 4, 0],
                         [0, 0, 0, 0, 0]])

    assert_array_equal(result, expected)
    assert_array_equal(out, result)


def test_clear_border_non_binary_out_3d():
    image3d = cp.array(
        [[[1, 2, 3, 1, 2],
          [3, 3, 3, 4, 2],
          [3, 4, 3, 4, 2],
          [3, 3, 2, 1, 2]],
         [[1, 2, 3, 1, 2],
          [3, 3, 5, 4, 2],
          [3, 4, 5, 4, 2],
          [3, 3, 2, 1, 2]],
         [[1, 2, 3, 1, 2],
          [3, 3, 3, 4, 2],
          [3, 4, 3, 4, 2],
          [3, 3, 2, 1, 2]]]
    )
    out = cp.empty_like(image3d)

    result = clear_border(image3d, out=out)
    expected = cp.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 5, 0, 0],
          [0, 0, 5, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]]
    )

    assert_array_equal(result, expected)
    assert_array_equal(out, result)
