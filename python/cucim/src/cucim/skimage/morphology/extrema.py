# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""
Local extrema detection functions.

These functions identify local maxima and minima in n-dimensional arrays,
with proper handling of plateaus (connected regions of equal value).
"""

import warnings

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi
from cucim.skimage.util import invert

__all__ = ["local_maxima", "local_minima"]


def _resolve_footprint(footprint, connectivity, ndim):
    """
    Resolve footprint from footprint or connectivity parameter.

    Parameters
    ----------
    footprint : ndarray or None
        Explicit footprint array.
    connectivity : int or None
        Connectivity value for generating footprint.
    ndim : int
        Number of dimensions.

    Returns
    -------
    footprint : cupy.ndarray
        Boolean footprint array with shape (3,) * ndim.

    Raises
    ------
    ValueError
        If footprint has wrong number of dimensions or wrong size.
    """
    if footprint is None:
        if connectivity is None:
            # Default: full connectivity
            connectivity = ndim
        # Clamp connectivity to valid range
        connectivity = min(connectivity, ndim)
        # Generate binary structure: returns (3,) * ndim shaped array
        footprint = ndi.generate_binary_structure(ndim, connectivity)
    else:
        # Convert footprint to cupy boolean array
        footprint = cp.asarray(footprint, dtype=bool)

        # Validate footprint dimensions
        if footprint.ndim != ndim:
            raise ValueError(
                f"footprint must have the same number of dimensions as image, "
                f"got footprint.ndim={footprint.ndim} but image.ndim={ndim}"
            )

        # Validate footprint size (must be 3 in each dimension)
        for i, s in enumerate(footprint.shape):
            if s != 3:
                raise ValueError(
                    f"Each dimension size of footprint must be 3, "
                    f"got footprint.shape[{i}]={s}"
                )

    return footprint


def local_maxima(
    image, footprint=None, connectivity=None, indices=False, allow_borders=True
):
    """
    Find local maxima of n-dimensional array.

    This function finds local maxima in an image, where a local maximum is
    defined as a connected set of pixels (a plateau) with equal gray level
    that is strictly greater than ALL pixels in its external neighborhood.

    Parameters
    ----------
    image : ndarray
        Input array to find local maxima in.
    footprint : ndarray, optional
        Boolean array defining the neighborhood. If None, a full connectivity
        footprint is generated based on the ``connectivity`` parameter.
        ``footprint`` and ``connectivity`` are mutually exclusive.
    connectivity : int, optional
        The connectivity defining the neighborhood of a pixel. Used to generate
        a footprint if ``footprint`` is None. A connectivity of 1 means pixels
        sharing an edge (4-connectivity in 2D, 6-connectivity in 3D). A
        connectivity equal to ``image.ndim`` means pixels sharing a corner
        (8-connectivity in 2D, 26-connectivity in 3D).
        Default is ``image.ndim`` (full connectivity).
    indices : bool, optional
        If True, return a tuple of arrays containing the coordinates of local
        maxima. If False (default), return a boolean mask with the same shape
        as ``image``.
    allow_borders : bool, optional
        If True (default), local maxima at the image border are detected.
        If False, pixels at the image border are excluded from consideration.

    Returns
    -------
    maxima : ndarray or tuple of ndarrays
        If ``indices=False``, returns a boolean array of the same shape as
        ``image`` where True indicates pixels that are part of a local maximum.
        If ``indices=True``, returns a tuple of 1-D arrays containing the
        coordinates of local maxima (as returned by ``cp.nonzero``).

    See Also
    --------
    local_minima : Find local minima.
    skimage.feature.peak_local_max : Find peaks with additional filtering.

    Notes
    -----
    This function identifies strict local maxima: a connected plateau is
    only considered a local maximum if ALL pixels in its external neighborhood
    have strictly lower values (not equal). When a plateau is identified as
    a local maximum, ALL pixels belonging to that plateau are marked as True.

    The algorithm works as follows:
    1. Find candidate pixels using a hollow (center-excluded) maximum filter
    2. Label connected regions of candidates
    3. Validate each labeled region has uniform intensity
    4. Check that each valid plateau's external boundary has strictly lower
       values

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.morphology import local_maxima
    >>> image = cp.array([[0, 0, 0, 0, 0],
    ...                   [0, 1, 1, 1, 0],
    ...                   [0, 1, 2, 1, 0],
    ...                   [0, 1, 1, 1, 0],
    ...                   [0, 0, 0, 0, 0]], dtype=cp.float32)
    >>> local_maxima(image)
    array([[False, False, False, False, False],
           [False, False, False, False, False],
           [False, False,  True, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False]])

    For a plateau maximum, all plateau pixels are marked True:

    >>> image = cp.array([[0, 0, 0],
    ...                   [0, 5, 5],
    ...                   [0, 5, 5]], dtype=cp.float32)
    >>> local_maxima(image)
    array([[False, False, False],
           [False,  True,  True],
           [False,  True,  True]])
    """
    image = cp.asarray(image)

    if image.size == 0:
        if indices:
            return tuple(cp.array([], dtype=cp.intp) for _ in range(image.ndim))
        return cp.zeros(image.shape, dtype=bool)

    # Validate mutually exclusive parameters
    if footprint is not None and connectivity is not None:
        raise ValueError(
            "footprint and connectivity are mutually exclusive, "
            "only provide one."
        )

    # Call the internal implementation
    return _local_maxima(
        image,
        footprint=footprint,
        connectivity=connectivity,
        indices=indices,
        allow_borders=allow_borders,
    )


def local_minima(
    image, footprint=None, connectivity=None, indices=False, allow_borders=True
):
    """
    Find local minima of n-dimensional array.

    This function finds local minima in an image, where a local minimum is
    defined as a connected set of pixels (a plateau) with equal gray level
    that is strictly less than ALL pixels in its external neighborhood.

    Parameters
    ----------
    image : ndarray
        Input array to find local minima in.
    footprint : ndarray, optional
        Boolean array defining the neighborhood. If None, a full connectivity
        footprint is generated based on the ``connectivity`` parameter.
        ``footprint`` and ``connectivity`` are mutually exclusive.
    connectivity : int, optional
        The connectivity defining the neighborhood of a pixel. Used to generate
        a footprint if ``footprint`` is None. A connectivity of 1 means pixels
        sharing an edge (4-connectivity in 2D, 6-connectivity in 3D). A
        connectivity equal to ``image.ndim`` means pixels sharing a corner
        (8-connectivity in 2D, 26-connectivity in 3D).
        Default is ``image.ndim`` (full connectivity).
    indices : bool, optional
        If True, return a tuple of arrays containing the coordinates of local
        minima. If False (default), return a boolean mask with the same shape
        as ``image``.
    allow_borders : bool, optional
        If True (default), local minima at the image border are detected.
        If False, pixels at the image border are excluded from consideration.

    Returns
    -------
    minima : ndarray or tuple of ndarrays
        If ``indices=False``, returns a boolean array of the same shape as
        ``image`` where True indicates pixels that are part of a local minimum.
        If ``indices=True``, returns a tuple of 1-D arrays containing the
        coordinates of local minima (as returned by ``cp.nonzero``).

    See Also
    --------
    local_maxima : Find local maxima.

    Notes
    -----
    This function identifies strict local minima: a connected plateau is
    only considered a local minimum if ALL pixels in its external neighborhood
    have strictly higher values (not equal). When a plateau is identified as
    a local minimum, ALL pixels belonging to that plateau are marked as True.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.morphology import local_minima
    >>> image = cp.array([[5, 5, 5, 5, 5],
    ...                   [5, 4, 4, 4, 5],
    ...                   [5, 4, 3, 4, 5],
    ...                   [5, 4, 4, 4, 5],
    ...                   [5, 5, 5, 5, 5]], dtype=cp.float32)
    >>> local_minima(image)
    array([[False, False, False, False, False],
           [False, False, False, False, False],
           [False, False,  True, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False]])
    """
    return local_maxima(
        invert(image, signed_float=True),
        footprint=footprint,
        connectivity=connectivity,
        indices=indices,
        allow_borders=allow_borders,
    )


def _local_maxima(image, footprint, connectivity, indices, allow_borders):
    """
    Internal implementation of local_maxima.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (already validated as cupy array).
    footprint : cupy.ndarray or None
        Footprint defining the neighborhood.
    connectivity : int or None
        Connectivity for generating footprint.
    indices : bool
        Whether to return indices instead of mask.
    allow_borders : bool
        Whether to allow maxima at borders.

    Returns
    -------
    result : cupy.ndarray or tuple of cupy.ndarray
        Boolean mask or tuple of coordinate arrays.
    """
    # Validate dtype - float16 is not supported
    if image.dtype == np.float16:
        raise TypeError(
            f"dtype {image.dtype} which is not supported, try using "
            "float32 or float64 instead"
        )

    ndim = image.ndim
    original_shape = image.shape

    # Resolve footprint from footprint or connectivity parameter
    footprint = _resolve_footprint(footprint, connectivity, ndim)

    # Handle small arrays (any dimension < 3)
    # When allow_borders=False, we cannot apply a 3x3 footprint properly
    if not allow_borders and any(s < 3 for s in image.shape):
        warnings.warn(
            "maxima can't exist with any dimension smaller 3 "
            "if borders are excluded",
            UserWarning,
            stacklevel=4,
        )
        if indices:
            return tuple(cp.array([], dtype=cp.intp) for _ in range(ndim))
        return cp.zeros(original_shape, dtype=bool)

    # Pad image if allow_borders=True
    # This allows border pixels to be compared against "virtual" neighbors
    # with the minimum value, making them potential maxima
    if allow_borders:
        # Get minimum value for padding
        if cp.issubdtype(image.dtype, cp.floating):
            pad_value = float(image.min())
        else:
            pad_value = int(image.min())

        # Pad with 1 pixel on each side
        image = cp.pad(
            image, pad_width=1, mode="constant", constant_values=pad_value
        )

    # Step 3: Find candidates using maximum filter
    # Apply standard maximum filter (more efficient than hollow footprint
    # since rectangular footprints use separable 1D filters internally)
    # image_max[i,j] = max value in neighborhood of (i,j) including center
    image_max = ndi.maximum_filter(image, footprint=footprint)

    # Candidates are pixels where value equals the local maximum
    # i.e., pixels that are >= all neighbors in their neighborhood
    # These could be single-pixel maxima or part of a plateau
    candidates = image == image_max

    # Step 4: Label candidate regions and validate
    # Connected candidates form potential plateaus
    labels, num_labels = ndi.label(candidates, structure=footprint)

    if num_labels == 0:
        # No candidates found
        result = cp.zeros(image.shape, dtype=bool)
    else:
        # Step 5: Validate plateau boundaries using two checks:
        #
        # Check 1: No equal-value non-candidate neighbors
        # If a candidate pixel has a neighbor with the same value that is NOT
        # a candidate, that neighbor must see something higher. This means the
        # plateau's extended neighborhood includes higher values → invalid.
        #
        # We compute max of non-candidate neighbor values. For candidates,
        # we use a very low value so they don't contribute.
        if cp.issubdtype(image.dtype, cp.floating):
            low_val = float(image.min()) - 1.0
        else:
            # For integers, use float to allow going below min
            low_val = float(image.min()) - 1.0

        non_candidate_image = cp.where(
            candidates, low_val, image.astype(cp.float64)
        )
        max_non_candidate_neighbor = ndi.maximum_filter(
            non_candidate_image, footprint=footprint
        )
        # For each candidate, check if any non-candidate neighbor has value
        # >= candidate value. If so, this candidate (and its region) is invalid.
        no_bad_neighbor = image.astype(cp.float64) > max_non_candidate_neighbor

        # Check 2: Region must have at least one strictly lower neighbor
        # This ensures the region is a STRICT maximum (not a constant plateau).
        # For constant images, no pixel has a strictly lower neighbor.
        min_neighbor = ndi.minimum_filter(image, footprint=footprint)
        has_lower_neighbor = image > min_neighbor

        # Find labels where ANY pixel fails check 1 (has bad neighbor)
        # These labels are invalid
        bad_labels_mask = candidates & ~no_bad_neighbor
        if bad_labels_mask.any():
            bad_labels = cp.unique(labels[bad_labels_mask])
            bad_labels = bad_labels[bad_labels > 0]
        else:
            bad_labels = cp.array([], dtype=labels.dtype)

        # Find labels where AT LEAST ONE pixel passes check 2 (has lower)
        # These labels could be valid (if they also pass check 1)
        good_labels_mask = candidates & has_lower_neighbor
        if good_labels_mask.any():
            labels_with_lower = cp.unique(labels[good_labels_mask])
        else:
            labels_with_lower = cp.array([], dtype=labels.dtype)

        # Build validity lookup:
        # Valid = has at least one lower neighbor AND no bad neighbors
        valid_labels_lookup = cp.zeros(num_labels + 1, dtype=bool)
        valid_labels_lookup[labels_with_lower] = True  # start with these
        valid_labels_lookup[bad_labels] = False  # remove invalid
        valid_labels_lookup[0] = False  # background is never valid

        # When allow_borders=False, invalidate labels that touch the border
        # (not just border pixels, but entire connected regions touching border)
        if not allow_borders:
            # Lazy import to avoid circular import with segmentation module
            from cucim.skimage.segmentation._clear_border import (
                _get_border_labels,
            )

            # Find labels that touch the border and mark them invalid
            border_labels = _get_border_labels(labels)
            border_labels = border_labels[border_labels > 0]
            valid_labels_lookup[border_labels] = False

        result = valid_labels_lookup[labels]

    # Remove padding from result if we added it
    if allow_borders:
        # Extract the center region (remove padding)
        slices = tuple(slice(1, -1) for _ in range(ndim))
        result = result[slices]

    if indices:
        return cp.nonzero(result)
    return result
