import cupy as cp
import numpy as np
from kvikio.nvcomp import LZ4Manager
from numcodecs import registry
from numcodecs.abc import Codec


def ensure_ndarray(buf):
    if isinstance(buf, cp.ndarray):
        arr = buf
    elif hasattr(buf, "__cuda_array_interface__"):
        arr = cp.asarray(buf, copy=False)
    elif hasattr(buf, "__array_interface__"):
        arr = cp.asarray(np.asarray(buf))
    else:
        raise ValueError("expected a cupy.ndarray")
    return arr


def ensure_contiguous_ndarray(buf, max_buffer_size=None, flatten=True):
    """Convenience function to coerce `buf` to ndarray-like array.
    Also ensures that the returned value exports fully contiguous memory,
    and supports the new-style buffer interface. If the optional max_buffer_size
    is provided, raise a ValueError if the number of bytes consumed by the
    returned array exceeds this value.

    Parameters
    ----------
    buf : ndarray-like, array-like, or bytes-like
        A numpy array like object such as numpy.ndarray, cupy.ndarray, or
        any object exporting a buffer interface.
    max_buffer_size : int
        If specified, the largest allowable value of arr.nbytes, where arr
        is the returned array.
    flatten : bool
        If True, the array are flatten.

    Returns
    -------
    arr : cupy.ndarray
        A cupy.ndarray, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is
    guaranteed to return a view on memory exported by `buf`.
    """
    arr = ensure_ndarray(buf)

    # check for object arrays, these are just memory pointers, actual memory
    # holding item data is scattered elsewhere
    if arr.dtype == object:
        raise TypeError("object arrays are not supported")

    # check for datetime or timedelta ndarray, the buffer interface doesn't
    # support those
    if arr.dtype.kind in "Mm":
        arr = arr.view(np.int64)

    # check memory is contiguous, if so flatten
    if arr.flags.c_contiguous or arr.flags.f_contiguous:
        if flatten:
            # can flatten without copy
            arr = arr.reshape(-1, order="A")
    else:
        raise ValueError("an array with contiguous memory is required")

    if max_buffer_size is not None and arr.nbytes > max_buffer_size:
        msg = "Codec does not support buffers of > {} bytes".format(
            max_buffer_size
        )
        raise ValueError(msg)

    return arr


def ndarray_copy(src, dst):
    """Copy the contents of the array from `src` to `dst`."""

    if dst is None:
        # no-op
        return src

    # ensure ndarray like
    src = ensure_ndarray(src)
    dst = ensure_ndarray(dst)

    # flatten source array
    src = src.reshape(-1, order="A")

    # ensure same data type
    if dst.dtype != object:
        src = src.view(dst.dtype)

    # reshape source to match destination
    if src.shape != dst.shape:
        if dst.flags.f_contiguous:
            order = "F"
        else:
            order = "C"
        src = src.reshape(dst.shape, order=order)

    # copy via numpy
    cp.copyto(dst, src)

    return dst


class LZ4NVCOMP(Codec):
    """Codec providing compression using LZ4 on the GPU via nvCOMP.

    Parameters
    ----------
    acceleration : int
        Acceleration level. The larger the acceleration value, the faster the
        algorithm, but also the lesser the compression.

    See Also
    --------
    numcodecs.zstd.Zstd, numcodecs.blosc.Blosc

    """

    codec_id = "lz4nvcomp"
    max_buffer_size = 0x7E000000

    def __init__(
        self, compressor=None
    ):  # , acceleration=1  (nvcomp lz4 doesn't take an acceleration argument)
        # self.acceleration = acceleration
        self._compressor = None

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        if (
            self._compressor is None
        ):  # or self._compressor.data_type != buf.dtype:
            self._compressor = LZ4Manager(data_type=buf.dtype)
        return self._compressor.compress(buf)  # , self.acceleration)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        if (
            self._compressor is None
        ):  # or self._compressor.data_type != buf.dtype:
            self._compressor = LZ4Manager(data_type=buf.dtype)
        decompressed = self._compressor.decompress(buf)
        return ndarray_copy(decompressed, out)

    def __repr__(self):
        r = "%s" % type(self).__name__
        return r


registry.register_codec(LZ4NVCOMP)
