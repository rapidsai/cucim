import math
from functools import reduce
from operator import mul

import cupy

# TODO: when minimum numpy dependency is 1.25 use:
# np..exceptions.AxisError instead of AxisError
# and remove this try-except
try:
    from numpy import AxisError
except ImportError:
    from numpy.exceptions import AxisError

try:
    # try importing Cython-based private axis handling functions from CuPy
    if hasattr(cupy, "_core"):
        # CuPy 10 renames core->_core
        from cupy._core.internal import _normalize_axis_index  # NOQA
        from cupy._core.internal import _normalize_axis_indices  # NOQA
    else:
        from cupy.core.internal import _normalize_axis_index  # NOQA
        from cupy.core.internal import _normalize_axis_indices  # NOQA

except ImportError:
    # Fallback to local Python implementations

    def _normalize_axis_index(axis, ndim):  # NOQA
        """
        Normalizes an axis index, ``axis``, such that is a valid positive
        index into the shape of array with ``ndim`` dimensions. Raises a
        ValueError with an appropriate message if this is not possible.
        Args:
            axis (int):
                The un-normalized index of the axis. Can be negative
            ndim (int):
                The number of dimensions of the array that ``axis`` should
                be normalized against
        Returns:
            int:
                The normalized axis index, such that
                `0 <= normalized_axis < ndim`
        """
        if axis < 0:
            axis += ndim
        if not (0 <= axis < ndim):
            raise AxisError("axis out of bounds")
        return axis

    def _normalize_axis_indices(axes, ndim):  # NOQA
        """Normalize axis indices.
        Args:
            axis (int, tuple of int or None):
                The un-normalized indices of the axis. Can be negative.
            ndim (int):
                The number of dimensions of the array that ``axis`` should
                be normalized against
        Returns:
            tuple of int:
                The tuple of normalized axis indices.
        """
        if axes is None:
            axes = tuple(range(ndim))
        elif not isinstance(axes, tuple):
            axes = (axes,)

        res = []
        for axis in axes:
            axis = _normalize_axis_index(axis, ndim)
            if axis in res:
                raise ValueError("Duplicate value in 'axis'")
            res.append(axis)

        return tuple(sorted(res))


if hasattr(math, "prod"):
    prod = math.prod
else:

    def prod(iterable, *, start=1):
        return reduce(mul, iterable, start)
