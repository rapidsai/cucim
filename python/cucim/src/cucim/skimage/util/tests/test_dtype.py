import itertools

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal

from cucim.skimage import (img_as_float, img_as_float32, img_as_float64,
                           img_as_int, img_as_ubyte, img_as_uint)
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.util.dtype import _convert, convert

dtype_range = {cp.uint8: (0, 255),
               cp.uint16: (0, 65535),
               cp.int8: (-128, 127),
               cp.int16: (-32768, 32767),
               cp.float32: (-1.0, 1.0),
               cp.float64: (-1.0, 1.0)}


img_funcs = (img_as_int, img_as_float64, img_as_float32,
             img_as_uint, img_as_ubyte)
dtypes_for_img_funcs = (cp.int16, cp.float64, cp.float32, cp.uint16, cp.ubyte)
img_funcs_and_types = zip(img_funcs, dtypes_for_img_funcs)


def _verify_range(msg, x, vmin, vmax, dtype):
    assert x[0] == vmin
    assert x[-1] == vmax
    assert x.dtype == dtype


@pytest.mark.parametrize(
    "dtype, f_and_dt", itertools.product(dtype_range, img_funcs_and_types)
)
def test_range(dtype, f_and_dt):
    imin, imax = dtype_range[dtype]
    x = cp.linspace(imin, imax, 10).astype(dtype)

    f, dt = f_and_dt

    y = f(x)

    omin, omax = dtype_range[dt]

    if imin == 0 or omin == 0:
        omin = 0
        imin = 0

    _verify_range("From %s to %s" % (cp.dtype(dtype), cp.dtype(dt)),
                  y, omin, omax, np.dtype(dt))


# Add non-standard data types that are allowed by the `_convert` function.
dtype_range_extra = dtype_range.copy()
dtype_range_extra.update({cp.int32: (-2147483648, 2147483647),
                          cp.uint32: (0, 4294967295)})

dtype_pairs = [(cp.uint8, cp.uint32),
               (cp.int8, cp.uint32),
               (cp.int8, cp.int32),
               (cp.int32, cp.int8),
               (cp.float64, cp.float32),
               (cp.int32, cp.float32)]


@pytest.mark.parametrize("dtype_in, dt", dtype_pairs)
def test_range_extra_dtypes(dtype_in, dt):
    """Test code paths that are not skipped by `test_range`"""

    imin, imax = dtype_range_extra[dtype_in]
    x = cp.linspace(imin, imax, 10).astype(dtype_in)

    y = _convert(x, dt)

    omin, omax = dtype_range_extra[dt]
    _verify_range("From %s to %s" % (cp.dtype(dtype_in), cp.dtype(dt)),
                  y, omin, omax, cp.dtype(dt))


def test_downcast():
    x = cp.arange(10).astype(cp.uint64)
    with expected_warnings(['Downcasting']):
        y = img_as_int(x)
    assert cp.allclose(y, x.astype(cp.int16))
    assert y.dtype == cp.int16, y.dtype


def test_float_out_of_range():
    too_high = cp.array([2], dtype=cp.float32)
    with pytest.raises(ValueError):
        img_as_int(too_high)
    too_low = cp.array([-2], dtype=cp.float32)
    with pytest.raises(ValueError):
        img_as_int(too_low)


def test_float_float_all_ranges():
    arr_in = cp.array([[-10.0, 10.0, 1e20]], dtype=cp.float32)
    cp.testing.assert_array_equal(img_as_float(arr_in), arr_in)


def test_copy():
    x = cp.array([1], dtype=cp.float64)
    y = img_as_float(x)
    z = img_as_float(x, force_copy=True)

    assert y is x
    assert z is not x


def test_bool():
    img_ = cp.zeros((10, 10), bool)
    img8 = cp.zeros((10, 10), cp.bool8)
    img_[1, 1] = True
    img8[1, 1] = True
    for (func, dt) in [(img_as_int, cp.int16),
                       (img_as_float, cp.float64),
                       (img_as_uint, cp.uint16),
                       (img_as_ubyte, cp.ubyte)]:
        converted_ = func(img_)
        assert cp.sum(converted_) == dtype_range[dt][1]
        converted8 = func(img8)
        assert cp.sum(converted8) == dtype_range[dt][1]


def test_clobber():
    # The `img_as_*` functions should never modify input arrays.
    for func_input_type in img_funcs:
        for func_output_type in img_funcs:
            img = cp.random.rand(5, 5)

            img_in = func_input_type(img)
            img_in_before = img_in.copy()
            func_output_type(img_in)

            assert_array_equal(img_in, img_in_before)


def test_signed_scaling_float32():
    x = cp.array([-128, 127], dtype=cp.int8)
    y = img_as_float32(x)
    assert y.max().get() == 1


def test_float32_passthrough():
    x = cp.array([-1, 1], dtype=cp.float32)
    y = img_as_float(x)
    assert y.dtype == x.dtype


float_dtype_list = [float, float, cp.double, cp.single, cp.float32,
                    cp.float64, 'float32', 'float64']


def test_float_conversion_dtype():
    """Test any convertion from a float dtype to an other."""
    x = cp.array([-1, 1])

    # Test all combinations of dtypes convertions
    dtype_combin = np.array(np.meshgrid(float_dtype_list,
                                        float_dtype_list)).T.reshape(-1, 2)

    for dtype_in, dtype_out in dtype_combin:
        x = x.astype(dtype_in)
        y = _convert(x, dtype_out)
        assert y.dtype == cp.dtype(dtype_out)


def test_float_conversion_dtype_warns():
    """Test that convert issues a warning when called"""
    x = np.array([-1, 1])

    # Test all combinations of dtypes convertions
    dtype_combin = np.array(np.meshgrid(float_dtype_list,
                                        float_dtype_list)).T.reshape(-1, 2)

    for dtype_in, dtype_out in dtype_combin:
        x = x.astype(dtype_in)
        with expected_warnings(["The use of this function is discouraged"]):
            y = convert(x, dtype_out)
        assert y.dtype == cp.dtype(dtype_out)


def test_subclass_conversion():
    """Check subclass conversion behavior"""
    x = cp.array([-1, 1])

    for dtype in float_dtype_list:
        x = x.astype(dtype)
        y = _convert(x, cp.floating)
        assert y.dtype == x.dtype
