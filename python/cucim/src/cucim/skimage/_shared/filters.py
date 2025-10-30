# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Filters used across multiple skimage submodules.

These are defined here to avoid circular imports.

The unit tests remain under skimage/filters/tests/
"""
from collections.abc import Iterable

import cupy as cp

import cucim.skimage._vendored.ndimage as ndi

from .._shared.utils import _supported_float_type, convert_to_float


def gaussian(
    image,
    sigma=1,
    mode="nearest",
    cval=0,
    preserve_range=False,
    truncate=4.0,
    *,
    channel_axis=None,
    out=None,
):
    """Multi-dimensional Gaussian filter.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or color) to filter.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of ``img_as_float``.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    truncate : float, optional
        Truncate the filter at this many standard deviations.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 24.02
          `channel_axis` was added in 24.02`
    out : ndarray, optional
        If given, the filtered image will be stored in this array. It must have
        a floating point data type.

        .. versionadded:: 24.06
            `out` was added in 24.06`

    Returns
    -------
    filtered_image : ndarray
        the filtered array

    Notes
    -----
    This function is a wrapper around :func:`scipy.ndi.gaussian_filter`.

    Integer arrays are converted to float.

    ``out`` should be of floating point data type since `gaussian` converts the
    input `image` to float. If `out` is not provided, another array
    will be allocated and returned as the result.

    The multi-dimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    Examples
    --------

    >>> import cupy as cp
    >>> from cucim import skimage as ski
    >>> a = cp.zeros((3, 3))
    >>> a[1, 1] = 1
    >>> a
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    >>> ski.filters.gaussian(a, sigma=0.4)  # mild smoothing
    array([[0.00163116, 0.03712502, 0.00163116],
           [0.03712502, 0.84496158, 0.03712502],
           [0.00163116, 0.03712502, 0.00163116]])
    >>> ski.filters.gaussian(a, sigma=1)  # more smoothing
    array([[0.05855018, 0.09653293, 0.05855018],
           [0.09653293, 0.15915589, 0.09653293],
           [0.05855018, 0.09653293, 0.05855018]])
    >>> # Several modes are possible for handling boundaries
    >>> ski.filters.gaussian(a, sigma=1, mode='reflect')
    array([[0.08767308, 0.12075024, 0.08767308],
           [0.12075024, 0.16630671, 0.12075024],
           [0.08767308, 0.12075024, 0.08767308]])
    >>> # For RGB images, each is filtered separately
    >>> from skimage.data import astronaut
    >>> image = cp.array(astronaut())
    >>> filtered_img = ski.filters.gaussian(image, sigma=1, channel_axis=-1)
    """

    # CuPy Backend: refactor to avoid overhead of cp.any(cp.asarray(sigma))
    sigma_msg = "Sigma values less than zero are not valid"
    if not isinstance(sigma, Iterable):
        if sigma < 0:
            raise ValueError(sigma_msg)
    elif any(s < 0 for s in sigma):
        raise ValueError(sigma_msg)

    if channel_axis is not None:
        axes = tuple(
            ax for ax in range(image.ndim) if ax != channel_axis % image.ndim
        )
    else:
        axes = tuple(range(image.ndim))

    image = convert_to_float(image, preserve_range)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if out is None:
        out = cp.empty_like(image)
    elif not cp.issubdtype(out.dtype, cp.floating):
        raise ValueError(
            f"dtype of `out` must be floating point; got {out.dtype!r}."
        )
    # ndi.gaussian_filter cannot do in-place filtering, must make a copy
    out_copied = False
    if out is image:
        out_img = image.copy()
        out_copied = True
    else:
        out_img = out
    ndi.gaussian_filter(
        image,
        sigma,
        output=out_img,
        mode=mode,
        cval=cval,
        truncate=truncate,
        axes=axes,
    )
    if out_img is not None and out_copied:
        image[...] = out_img

    return out
