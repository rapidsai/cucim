"""
Simplified version of cupy.gradient

This version doesn't support non-unit spacing or 2nd order edges.

Importantly, this version does not promote all integer dtypes to float64, but
instead will promote 8 and 16-bit integer types to float32.
"""
import cupy

from cucim.skimage._shared.utils import _supported_float_type


def gradient(f, axis=None, output_as_array=False):
    """Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    Args:
        f (cupy.ndarray): An N-dimensional array containing samples of a scalar
            function.
        axis (None or int or tuple of ints, optional): The gradient is
            calculated only along the given axis or axes. The default
            (axis = None) is to calculate the gradient for all the axes of the
            input array. axis may be negative, in which case it counts from the
            last to the first axis.
        output_as_array

    Returns:
        gradient (cupy.ndarray or list of cupy.ndarray): A set of ndarrays
        (or a single ndarray if there is only one dimension) corresponding
        to the derivatives of f with respect to each dimension. Each
        derivative has the same shape as f.

    """
    ndim = f.ndim  # number of dimensions
    if axis is None:
        axes = tuple(range(ndim))
    else:
        if cupy.isscalar(axis):
            axis = (axis,)
        for ax in axis:
            if ax < -ndim or ax > ndim + 1:
                raise ValueError(f"invalid axis: {ax}")
        axes = tuple(ax + ndim if ax < 0 else ax for ax in axis)
    len_axes = len(axes)

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice3 = [slice(None)] * ndim
    slice4 = [slice(None)] * ndim

    otype = f.dtype
    if cupy.issubdtype(otype, cupy.inexact):
        pass
    else:
        # All other types convert to floating point.
        float_dtype = _supported_float_type(otype)
        if cupy.issubdtype(otype, cupy.integer):
            f = f.astype(float_dtype)
        otype = float_dtype

    if output_as_array:
        out = cupy.empty((ndim,) + f.shape, dtype=otype)
        outvals = out
    else:
        outvals = []

    for axis in axes:
        if f.shape[axis] < 2:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least 2 elements are required."
            )
        # result allocation
        if not output_as_array:
            out = cupy.empty_like(f, dtype=otype)

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        out_sl = (axis,) + tuple(slice1) if output_as_array else tuple(slice1)
        out[out_sl] = (f[tuple(slice4)] - f[tuple(slice2)]) / 2.0

        # Numerical differentiation: 1st order edges
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
        out_sl = (axis,) + tuple(slice1) if output_as_array else tuple(slice1)
        out[out_sl] = f[tuple(slice2)] - f[tuple(slice3)]

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
        out_sl = (axis,) + tuple(slice1) if output_as_array else tuple(slice1)
        out[out_sl] = f[tuple(slice2)] - f[tuple(slice3)]
        if not output_as_array:
            outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals
