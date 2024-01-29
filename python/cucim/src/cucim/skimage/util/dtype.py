import math
import warnings
from warnings import warn

import cupy as cp
import numpy as np

from .._shared.utils import _supported_float_type

__all__ = [
    "img_as_float32",
    "img_as_float64",
    "img_as_float",
    "img_as_int",
    "img_as_uint",
    "img_as_ubyte",
    "img_as_bool",
    "dtype_limits",
]

# For integers Numpy uses `_integer_types` basis internally, and builds a leaky
# `cupy.XintYY` abstraction on top of it. This leads to situations when, for
# example, there are two cupy.Xint64 dtypes with the same attributes but
# different object references. In order to avoid any potential issues,
# we use the basis dtypes here. For more information, see:
# - https://github.com/scikit-image/scikit-image/issues/3043
# For convenience, for these dtypes we indicate also the possible bit depths
# (some of them are platform specific). For the details, see:
# http://www.unix.org/whitepapers/64bit.html
_integer_types = (
    cp.byte,
    cp.ubyte,  # 8 bits
    cp.short,
    cp.ushort,  # 16 bits
    cp.intc,
    cp.uintc,  # 16 or 32 or 64 bits
    int,
    cp.int_,
    cp.uint,  # 32 or 64 bits
    cp.longlong,
    cp.ulonglong,
)  # 64 bits
_integer_ranges = {
    t: (cp.iinfo(t).min, cp.iinfo(t).max) for t in _integer_types
}
dtype_range = {
    bool: (False, True),
    cp.bool_: (False, True),
    float: (-1, 1),
    cp.float16: (-1, 1),
    cp.float32: (-1, 1),
    cp.float64: (-1, 1),
}
dtype_range.update(_integer_ranges)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # cp.bool8 is a deprecated alias of cp.bool_
    if hasattr(cp, "bool8"):
        dtype_range[cp.bool8] = (False, True)

_supported_types = list(dtype_range.keys())


def dtype_limits(image, clip_negative=False):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.

    Returns
    -------
    imin, imax : tuple
        Lower and upper intensity limits.
    """
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


def _dtype_itemsize(itemsize, *dtypes):
    """Return first of `dtypes` with itemsize greater than `itemsize`

    Parameters
    ----------
    itemsize: int
        The data type object element size.

    Other Parameters
    ----------------
    *dtypes:
        Any Object accepted by `cp.dtype` to be converted to a data
        type object

    Returns
    -------
    dtype: data type object
        First of `dtypes` with itemsize greater than `itemsize`.

    """
    return next(dt for dt in dtypes if cp.dtype(dt).itemsize >= itemsize)


def _dtype_bits(kind, bits, itemsize=1):
    """Return dtype of `kind` that can store a `bits` wide unsigned int

    Parameters:
    kind: str
        Data type kind.
    bits: int
        Desired number of bits.
    itemsize: int
        The data type object element size.

    Returns
    -------
    dtype: data type object
        Data type of `kind` that can store a `bits` wide unsigned int

    """

    s = next(
        i
        for i in (itemsize,) + (2, 4, 8)
        if bits < (i * 8) or (bits == (i * 8) and kind == "u")
    )

    return cp.dtype(kind + str(s))


def _scale(a, n, m, copy=True):
    """Scale an array of unsigned/positive integers from `n` to `m` bits.

    Numbers can be represented exactly only if `m` is a multiple of `n`.

    Parameters
    ----------
    a : ndarray
        Input image array.
    n : int
        Number of bits currently used to encode the values in `a`.
    m : int
        Desired number of bits to encode the values in `out`.
    copy : bool, optional
        If True, allocates and returns new array. Otherwise, modifies
        `a` in place.

    Returns
    -------
    out : array
        Output image array. Has the same kind as `a`.
    """
    kind = a.dtype.kind
    if n > m and a.max() < 2**m:
        mnew = math.ceil(m / 2) * 2
        if mnew > m:
            dtype = f"int{mnew}"
        else:
            dtype = f"uint{mnew}"
        n = math.ceil(n / 2) * 2
        warn(
            f"Downcasting {a.dtype} to {dtype} without scaling because max "
            f"value {a.max()} fits in {dtype}",
            stacklevel=3,
        )
        return a.astype(_dtype_bits(kind, m))
    elif n == m:
        return a.copy() if copy else a
    elif n > m:
        # downscale with precision loss
        if copy:
            b = cp.empty(a.shape, _dtype_bits(kind, m))
            cp.floor_divide(
                a, 2 ** (n - m), out=b, dtype=a.dtype, casting="unsafe"
            )
            return b
        else:
            a //= 2 ** (n - m)
            return a
    elif m % n == 0:
        # exact upscale to a multiple of `n` bits
        if copy:
            b = cp.empty(a.shape, _dtype_bits(kind, m))
            cp.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
            return b
        else:
            a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
            a *= (2**m - 1) // (2**n - 1)
            return a
    else:
        # upscale to a multiple of `n` bits,
        # then downscale with precision loss
        o = (m // n + 1) * n
        if copy:
            b = cp.empty(a.shape, _dtype_bits(kind, o))
            cp.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
            b //= 2 ** (o - m)
            return b
        else:
            a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
            a *= (2**o - 1) // (2**n - 1)
            a //= 2 ** (o - m)
            return a


def _convert(image, dtype, force_copy=False, uniform=False):
    """
    Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).

    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.

    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.

    .. versionchanged :: 0.15
        ``_convert`` no longer warns about possible precision or sign
        information loss. See discussions on these warnings at:
        https://github.com/scikit-image/scikit-image/issues/2602
        https://github.com/scikit-image/scikit-image/issues/543#issuecomment-208202228
        https://github.com/scikit-image/scikit-image/pull/3575

    References
    ----------
    .. [1] DirectX data conversion rules.
           https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.

    """
    dtypeobj_in = image.dtype
    if dtype is cp.floating:
        dtypeobj_out = cp.dtype("float64")
    else:
        dtypeobj_out = cp.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    # Below, we do an `issubdtype` check.  Its purpose is to find out
    # whether we can get away without doing any image conversion.  This happens
    # when:
    #
    # - the output and input dtypes are the same or
    # - when the output is specified as a type, and the input dtype
    #   is a subclass of that type (e.g. `cp.floating` will allow
    #   `float32` and `float64` arrays through)

    if cp.issubdtype(dtype_in, np.core.numerictypes.obj2sctype(dtype)):
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError(
            f"Cannot convert from {dtypeobj_in} to " f"{dtypeobj_out}."
        )

    if kind_in in "ui":
        imin_in = cp.iinfo(dtype_in).min
        imax_in = cp.iinfo(dtype_in).max
    if kind_out in "ui":
        imin_out = cp.iinfo(dtype_out).min
        imax_out = cp.iinfo(dtype_out).max

    # any -> binary
    if kind_out == "b":
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == "b":
        result = image.astype(dtype_out)
        if kind_out != "f":
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == "f":
        if kind_out == "f":
            # float -> float
            return image.astype(dtype_out)

        if cp.min(image) < -1.0 or cp.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(
            itemsize_out, dtype_in, cp.float32, cp.float64
        )

        if not uniform:
            if kind_out == "u":
                image_out = cp.multiply(image, imax_out, dtype=computation_type)
            else:
                image_out = cp.multiply(
                    image, (imax_out - imin_out) / 2, dtype=computation_type
                )
                image_out -= 1.0 / 2.0
            cp.rint(image_out, out=image_out)
            cp.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == "u":
            image_out = cp.multiply(image, imax_out + 1, dtype=computation_type)
            cp.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = cp.multiply(
                image, (imax_out - imin_out + 1.0) / 2.0, dtype=computation_type
            )
            cp.floor(image_out, out=image_out)
            cp.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == "f":
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(
            itemsize_in, dtype_out, cp.float32, cp.float64
        )

        if kind_in == "u":
            # using cp.divide or cp.multiply doesn't copy the data
            # until the computation time
            image = cp.multiply(image, 1.0 / imax_in, dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     cp.maximum(image, -1.0, out=image)
        elif kind_in == "i":
            # From DirectX conversions:
            # The most negative value maps to -1.0f
            # Every other value is converted to a float (call it c)
            # and then result = c * (1.0f / (2⁽ⁿ⁻¹⁾-1)).

            image = cp.multiply(image, 1.0 / imax_in, dtype=computation_type)
            cp.maximum(image, -1.0, out=image)
        else:
            image = cp.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)

        return image.astype(dtype_out, copy=False)

    # unsigned int -> signed/unsigned int
    if kind_in == "u":
        if kind_out == "i":
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == "u":
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = cp.empty(image.shape, dtype_out)
        cp.maximum(image, 0, out=result, dtype=image.dtype, casting="unsafe")
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits("i", itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)


def convert(image, dtype, force_copy=False, uniform=False):
    warn(
        "The use of this function is discouraged as its behavior may change "
        "dramatically in scikit-image 1.0. This function will be removed"
        "in scikit-image 1.0.",
        FutureWarning,
        stacklevel=2,
    )
    return _convert(
        image=image, dtype=dtype, force_copy=force_copy, uniform=uniform
    )


if _convert.__doc__ is not None:
    convert.__doc__ = (
        _convert.__doc__
        + """

    Warns
    -----
    FutureWarning:
        .. versionadded:: 0.17

        The use of this function is discouraged as its behavior may change
        dramatically in scikit-image 1.0. This function will be removed
        in scikit-image 1.0.
    """
    )


def img_as_float32(image, force_copy=False):
    """Convert an image to single-precision (32-bit) floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float32
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return _convert(image, cp.float32, force_copy)


def img_as_float64(image, force_copy=False):
    """Convert an image to double-precision (64-bit) floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float64
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return _convert(image, cp.float64, force_copy)


def img_as_float(image, force_copy=False):
    """Convert an image to floating point format.

    This function is similar to `img_as_float64`, but will not convert
    lower-precision floating point arrays to `float64`.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    # casts float16, float32 and 8 or 16-bit integer types to float32
    float_dtype = _supported_float_type(image.dtype)
    return _convert(image, float_dtype, force_copy)


def img_as_uint(image, force_copy=False):
    """Convert an image to 16-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 65535.

    """
    return _convert(image, cp.uint16, force_copy)


def img_as_int(image, force_copy=False):
    """Convert an image to 16-bit signed integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of int16
        Output image.

    Notes
    -----
    The values are scaled between -32768 and 32767.
    If the input data-type is positive-only (e.g., uint8), then
    the output image will still only have positive values.

    """
    return _convert(image, cp.int16, force_copy)


def img_as_ubyte(image, force_copy=False):
    """Convert an image to 8-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of ubyte (uint8)
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 255.

    """
    return _convert(image, cp.uint8, force_copy)


def img_as_bool(image, force_copy=False):
    """Convert an image to boolean format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of bool (`bool_`)
        Output image.

    Notes
    -----
    The upper half of the input dtype's positive range is True, and the lower
    half is False. All negative values (if present) are False.

    """
    return _convert(image, bool, force_copy)
