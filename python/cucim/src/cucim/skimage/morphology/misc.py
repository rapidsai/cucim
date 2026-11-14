# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Miscellaneous morphology functions."""

import functools

import cupy as cp

import cucim.skimage._vendored.ndimage as ndi

from .._shared.utils import warn

# Our function names don't exactly correspond to ndimages.
# This dictionary translates from our names to scipy's.
funcs = ("erosion", "dilation", "opening", "closing")
skimage2ndimage = {x: "grey_" + x for x in funcs}

# These function names are the same in ndimage.
funcs = (
    "black_tophat",
    "white_tophat",
)
skimage2ndimage.update({x: x for x in funcs})


def default_footprint(func):
    """Decorator to add a default footprint to morphology functions.

    Parameters
    ----------
    func : function
        A morphology function such as erosion, dilation, opening, closing,
        white_tophat, or black_tophat.

    Returns
    -------
    func_out : function
        The function, using a default footprint of same dimension
        as the input image with connectivity 1.

    """

    @functools.wraps(func)
    def func_out(image, footprint=None, *args, **kwargs):
        if footprint is None:
            footprint = ndi.generate_binary_structure(image.ndim, 1)
        return func(image, footprint=footprint, *args, **kwargs)

    return func_out


def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or cp.issubdtype(ar.dtype, cp.integer)):
        raise TypeError(
            f"Only bool or integer image types are supported. Got {ar.dtype}."
        )


def remove_small_objects(ar, *, max_size=63, connectivity=1, out=None):
    """Remove objects smaller than or equal to the specified size.

    Expects `ar` to be an array with labeled objects, and removes objects
    smaller than or equal to `max_size`. If `ar` is bool, the image is first
    labeled. This leads to potentially different behavior for bool vs. 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    max_size : int, optional (default: 63)
        Remove objects whose contiguous area (or volume, in N-D) contains this
        number of pixels or fewer.

        .. versionchanged:: 26.12
            The deprecated `min_size` parameter was removed. Use
            ``max_size=min_size - 1`` to preserve the old threshold behavior.
            All parameters after `ar` are now keyword-only, so a positional
            `min_size` value raises instead of silently changing meaning.
            The default is now 63, matching the former ``min_size=64``.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    See Also
    --------
    cucim.skimage.morphology.remove_small_holes

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import morphology
    >>> a = cp.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, max_size=5)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, max_size=6, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, max_size=5, out=a)
    >>> d is a
    True

    """
    # Raising type error if not int or bool
    _check_dtype_supported(ar)

    if out is None:
        out = ar.copy()
    else:
        out[:] = ar

    if max_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = cp.zeros_like(ar, dtype=cp.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = cp.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    if len(component_sizes) == 2 and out.dtype != bool:
        warn(
            "Only one label was provided to `remove_small_objects`. "
            "Did you mean to use a boolean array?"
        )

    too_small = component_sizes <= max_size

    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def remove_small_holes(ar, *, max_size=63, connectivity=1, out=None):
    """Remove contiguous holes smaller than or equal to the specified size.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest.
    max_size : int, optional (default: 63)
        Remove holes whose contiguous area (or volume, in N-D) contains this
        number of pixels or fewer.

        .. versionchanged:: 26.12
            The deprecated `area_threshold` parameter was removed. Use
            ``max_size=area_threshold - 1`` to preserve the old threshold
            behavior. All parameters after `ar` are now keyword-only, so a
            positional `area_threshold` value raises instead of silently
            changing meaning. The default is now 63, matching the former
            ``area_threshold=64``.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    out : ndarray
        Array of the same shape as `ar` and bool dtype, into which the
        output is placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small holes within connected components removed.

    See Also
    --------
    cucim.skimage.morphology.remove_small_objects

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import morphology
    >>> a = cp.array([[1, 1, 1, 1, 1, 0],
    ...               [1, 1, 1, 0, 1, 0],
    ...               [1, 0, 0, 1, 1, 0],
    ...               [1, 1, 1, 1, 1, 0]], bool)
    >>> b = morphology.remove_small_holes(a, max_size=1)
    >>> b
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> c = morphology.remove_small_holes(a, max_size=1, connectivity=2)
    >>> c
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True, False,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> d = morphology.remove_small_holes(a, max_size=1, out=a)
    >>> d is a
    True

    Notes
    -----
    If the array type is int, it is assumed that it contains already-labeled
    objects. The labels are not kept in the output image (this function always
    outputs a bool image). It is suggested that labeling is completed after
    using this function.

    """
    _check_dtype_supported(ar)

    # Creates warning if image is an integer image
    if ar.dtype != bool:
        warn(
            "Any labeled images will be returned as a boolean array. "
            "Did you mean to use a boolean array?",
            UserWarning,
        )

    if out is not None:
        if out.dtype != bool:
            raise TypeError("out dtype must be bool")
    else:
        out = ar.astype(bool, copy=True)

    # Creating the inverse of ar
    cp.logical_not(ar, out=out)

    # removing small objects from the inverse of ar
    out = remove_small_objects(
        out,
        max_size=max_size,
        connectivity=connectivity,
        out=out,
    )

    cp.logical_not(out, out=out)

    return out
