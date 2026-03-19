# SPDX-FileCopyrightText: Copyright (c) 2003-2009 Massachusetts Institute of Technology
# SPDX-FileCopyrightText: Copyright (c) 2009-2011 Broad Institute
# SPDX-FileCopyrightText: 2003 Lee Kamentsky
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND (GPL-2.0-only OR BSD-3-Clause)

"""
This morphological reconstruction routine was adapted from CellProfiler, code
licensed under both GPL and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""

import cupy as cp
import numpy as np
import skimage
from packaging.version import Version

import cucim.skimage._vendored.ndimage as ndi

old_reconstruction_pyx = Version(skimage.__version__) < Version("0.20.0")


def reconstruction(
    seed,
    mask,
    method="dilation",
    footprint=None,
    offset=None,
    *,
    reconstruct_on_cpu=False,
):
    """Perform a morphological reconstruction of an image.

    Morphological reconstruction by dilation is similar to basic morphological
    dilation: high-intensity values will replace nearby low-intensity values.
    The basic dilation operator, however, uses a footprint to
    determine how far a value in the input image can spread. In contrast,
    reconstruction uses two images: a "seed" image, which specifies the values
    that spread, and a "mask" image, which gives the maximum allowed value at
    each pixel. The mask image, like the footprint, limits the spread
    of high-intensity values. Reconstruction by erosion is simply the inverse:
    low-intensity values spread from the seed image and are limited by the mask
    image, which represents the minimum allowed value.

    Alternatively, you can think of reconstruction as a way to isolate the
    connected regions of an image. For dilation, reconstruction connects
    regions marked by local maxima in the seed image: neighboring pixels
    less-than-or-equal-to those seeds are connected to the seeded region.
    Local maxima with values larger than the seed image will get truncated to
    the seed value.

    Parameters
    ----------
    seed : ndarray
        The seed image (a.k.a. marker image), which specifies the values that
        are dilated or eroded.
    mask : ndarray
        The maximum (dilation) / minimum (erosion) allowed value at each pixel.
    method : {'dilation'|'erosion'}, optional
        Perform reconstruction by dilation or erosion. In dilation (or
        erosion), the seed image is dilated (or eroded) until limited by the
        mask image. For dilation, each seed value must be less than or equal
        to the corresponding mask value; for erosion, the reverse is true.
        Default is 'dilation'.
    footprint : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the n-D square of radius equal to 1 (i.e. a 3x3 square
        for 2D images, a 3x3x3 cube for 3D images, etc.)
    offset : ndarray, optional
        The coordinates of the center of the footprint.
        Default is located on the geometrical center of the footprint, in that
        case footprint dimensions must be odd.
    reconstruct_on_cpu : bool, optional
        If False (default), use an iterative GPU-native algorithm that stays
        entirely on the device. If True, fall back to scikit-image's Cython
        reconstruction_loop on the CPU (requires host-device transfers).

    Returns
    -------
    reconstructed : ndarray
       The result of morphological reconstruction. Note that scikit-image always
       returns a floating-point image. cuCIM returns the same dtype as the input
       except in the case of erosion, where it will have promoted any unsigned
       integer dtype to a signed type (using the dtype returned by
       ``cp.promote_types(seed.dtype, cp.int8)``).

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.morphology import reconstruction

    First, we create a sinusoidal mask image with peaks at middle and ends.

    >>> x = cp.linspace(0, 4 * np.pi)
    >>> y_mask = cp.cos(x)

    Then, we create a seed image initialized to the minimum mask value (for
    reconstruction by dilation, min-intensity values don't spread) and add
    "seeds" to the left and right peak, but at a fraction of peak value (1).

    >>> y_seed = y_mask.min() * cp.ones_like(x)
    >>> y_seed[0] = 0.5
    >>> y_seed[-1] = 0
    >>> y_rec = reconstruction(y_seed, y_mask)

    The reconstructed image (or curve, in this case) is exactly the same as the
    mask image, except that the peaks are truncated to 0.5 and 0. The middle
    peak disappears completely: Since there were no seed values in this peak
    region, its reconstructed value is truncated to the surrounding value (-1).

    As a more practical example, we try to extract the bright features of an
    image by subtracting a background image created by reconstruction.

    >>> y, x = cp.mgrid[:20:0.5, :20:0.5]
    >>> bumps = cp.sin(x) + cp.sin(y)

    To create the background image, set the mask image to the original image,
    and the seed image to the original image with an intensity offset, `h`.

    >>> h = 0.3
    >>> seed = bumps - h
    >>> background = reconstruction(seed, bumps)

    The resulting reconstructed image looks exactly like the original image,
    but with the peaks of the bumps cut off. Subtracting this reconstructed
    image from the original image leaves just the peaks of the bumps

    >>> hdome = bumps - background

    This operation is known as the h-dome of the image and leaves features
    of height `h` in the subtracted image.

    Notes
    -----
    Two algorithm variants are available, selected by the
    ``reconstruct_on_cpu`` parameter:

    When ``reconstruct_on_cpu=True``, the CPU-based algorithm from [1]_ is used. This
    processes pixels in rank order using a linked-list traversal via
    scikit-image's Cython ``reconstruction_loop``. Data is transferred to
    the host for the inner loop and back to the device afterward.

    When ``reconstruct_on_cpu=False`` (default), an iterative parallel algorithm runs
    entirely on the GPU. Each iteration applies a conditional dilation (or
    erosion) via ``maximum_filter`` (or ``minimum_filter``) and clamps by the
    mask, repeating until convergence. This is highly efficient when the seed
    is close to the mask (e.g., in ``h_maxima`` / ``h_minima``), typically
    converging in few iterations with each iteration fully parallelized.
    However, in pathological cases where a single seed value must propagate
    across a long path (e.g., a single-pixel seed with a uniform mask),
    convergence can require O(N) iterations and the CPU path may be faster.

    An alternative GPU algorithm based on irregular wavefront propagation [4]_
    would handle such pathological cases efficiently by tiling the image into
    shared-memory-sized blocks and only re-processing tiles whose boundaries
    change. This approach is not currently implemented in cuCIM.

    Applications for grayscale reconstruction are discussed in [2]_ and [3]_.

    References
    ----------
    .. [1] Robinson, "Efficient morphological reconstruction: a downhill
           filter", Pattern Recognition Letters 25 (2004) 1759-1767.
    .. [2] Vincent, L., "Morphological Grayscale Reconstruction in Image
           Analysis: Applications and Efficient Algorithms", IEEE Transactions
           on Image Processing (1993)
    .. [3] Soille, P., "Morphological Image Analysis: Principles and
           Applications", Chapter 6, 2nd edition (2003), ISBN 3540429883.
    .. [4] Teodoro, G., Banerjee, T., Kurc, T.M., Sussman, A., Pan, T., and
           Saltz, J.H., "Efficient irregular wavefront propagation algorithms
           on hybrid CPU-GPU machines", Parallel Computing, 39(4-5),
           pp. 189-211, 2013.
    """
    seed = cp.asarray(seed)
    mask = cp.asarray(mask)

    if seed.shape != mask.shape:
        raise ValueError(
            f"seed and mask must have the same shape, "
            f"got {seed.shape} and {mask.shape}"
        )

    if method not in ("dilation", "erosion"):
        raise ValueError(
            "Reconstruction method can be one of 'erosion' "
            f"or 'dilation'. Got '{method}'."
        )

    if method == "dilation" and cp.any(seed > mask):  # synchronize!
        raise ValueError(
            "Intensity of seed image must be less than that "
            "of the mask image for reconstruction by dilation."
        )
    elif method == "erosion" and cp.any(seed < mask):  # synchronize!
        raise ValueError(
            "Intensity of seed image must be greater than that "
            "of the mask image for reconstruction by erosion."
        )

    if footprint is None:
        if reconstruct_on_cpu:
            footprint = np.ones([3] * seed.ndim, dtype=bool)
        else:
            footprint = cp.ones([3] * seed.ndim, dtype=bool)
    else:
        if reconstruct_on_cpu and isinstance(footprint, cp.ndarray):
            footprint = cp.asnumpy(footprint)
        footprint = footprint.astype(bool, copy=True)

    if offset is None:
        if not all([d % 2 == 1 for d in footprint.shape]):
            raise ValueError("Footprint dimensions must all be odd")
        offset = np.array([d // 2 for d in footprint.shape])
    else:
        if isinstance(offset, cp.ndarray):
            offset = cp.asnumpy(offset)
        if offset.ndim != footprint.ndim:
            raise ValueError("Offset and footprint ndims must be equal.")
        if not all([(0 <= o < d) for o, d in zip(offset, footprint.shape)]):
            raise ValueError("Offset must be included inside footprint")

    if reconstruct_on_cpu:
        return _reconstruction_cpu(seed, mask, method, footprint, offset)
    else:
        return _reconstruction_gpu(seed, mask, method, footprint, offset)


def _reconstruction_gpu(seed, mask, method, footprint, offset):
    """Iterative GPU-native morphological reconstruction.

    Repeatedly applies conditional dilation (or erosion) until convergence.
    Each iteration is a bulk GPU operation via maximum_filter/minimum_filter.
    """
    # Work in a dtype that can represent the reconstruction
    work_dtype = np.promote_types(seed.dtype, mask.dtype)
    if method == "erosion" and np.issubdtype(work_dtype, np.unsignedinteger):
        work_dtype = np.promote_types(work_dtype, np.int8)

    current = seed.astype(work_dtype, copy=True)
    mask_work = mask.astype(work_dtype, copy=False)

    # Convert reconstruction's `offset` to ndimage's `origin` parameter.
    # The CPU reconstruction defines offset as the footprint anchor: pixel i
    # propagates TO neighbors at positions defined by (footprint_pos - offset).
    # In the iterative GPU approach, maximum_filter has pixel i RECEIVE from
    # its window. To match the CPU's propagation direction, we reverse the
    # relationship: origin = shape // 2 - offset (per dimension).
    # For the default centered case (offset = shape // 2), origin = 0.
    origin = tuple(int(d // 2 - o) for d, o in zip(footprint.shape, offset))

    # Adaptive batch size: run several iterations between convergence checks
    # to amortize the cost of the device synchronization in .any().
    check_every = 4
    max_check_every = 64
    max_iter = max(seed.size, 1000)  # safety limit

    filter_kwargs = dict(footprint=footprint, origin=origin)

    is_dilation = method == "dilation"
    n_iter = 0
    while n_iter < max_iter:
        for _ in range(check_every):
            if is_dilation:
                dilated = ndi.maximum_filter(current, **filter_kwargs)
                cp.minimum(dilated, mask_work, out=current)
            else:
                eroded = ndi.minimum_filter(current, **filter_kwargs)
                cp.maximum(eroded, mask_work, out=current)
            n_iter += 1

        # Check convergence: would one more iteration change anything?
        if is_dilation:
            candidate = ndi.maximum_filter(current, **filter_kwargs)
            cp.minimum(candidate, mask_work, out=candidate)
        else:
            candidate = ndi.minimum_filter(current, **filter_kwargs)
            cp.maximum(candidate, mask_work, out=candidate)

        if not (candidate != current).any():
            break

        current = candidate
        n_iter += 1

        # Ramp up batch size if convergence is slow
        check_every = min(check_every * 2, max_check_every)

    return current


def _reconstruction_cpu(seed, mask, method, footprint, offset):
    """CPU-based morphological reconstruction using scikit-image's Cython loop.

    Data is transferred to host, processed single-threaded via
    scikit-image's reconstruction_loop, then transferred back.
    """
    from ..filters._rank_order import rank_order

    try:
        from skimage.morphology._grayreconstruct import reconstruction_loop
    except ImportError:
        try:
            from skimage.morphology._greyreconstruct import reconstruction_loop
        except ImportError:
            raise ImportError("reconstruction requires scikit-image")

    # Cross out the center of the footprint
    footprint[tuple(slice(d, d + 1) for d in offset)] = False

    # Make padding for edges of reconstructed image so we can ignore boundaries
    dims = (2,) + tuple(
        s1 + s2 - 1 for s1, s2 in zip(seed.shape, footprint.shape)
    )
    inside_slices = tuple(slice(o, o + s) for o, s in zip(offset, seed.shape))
    # Set padded region to minimum image intensity and mask along first axis so
    # we can interleave image and mask pixels when sorting.
    if method == "dilation":
        pad_value = cp.min(seed).item()
    else:
        pad_value = cp.max(seed).item()

    # CuPy Backend: modified to allow images_dtype based on input dtype
    #               instead of float64
    images_dtype = np.promote_types(seed.dtype, mask.dtype)
    # For erosion, we need to negate the array, so ensure we use a signed type
    # that can represent negative values without wraparound
    if method == "erosion" and cp.issubdtype(images_dtype, cp.unsignedinteger):
        # Promote unsigned types to signed types with sufficient range
        images_dtype = np.promote_types(images_dtype, cp.int8)

    images = cp.full(dims, pad_value, dtype=images_dtype)
    images[(0, *inside_slices)] = seed
    images[(1, *inside_slices)] = mask
    isize = images.size
    if old_reconstruction_pyx:
        # scikit-image < 0.20 Cython code only supports int32_t
        signed_int_dtype = np.int32
        unsigned_int_dtype = np.uint32
    else:
        # determine whether image is large enough to require 64-bit integers
        # use -isize so we get a signed dtype rather than an unsigned one
        signed_int_dtype = np.result_type(np.min_scalar_type(-isize), np.int32)
        # the corresponding unsigned type has same char, but uppercase
        unsigned_int_dtype = np.dtype(signed_int_dtype.char.upper())

    # Create a list of strides across the array to get the neighbors within
    # a flattened array
    value_stride = np.array(images.strides[1:]) // images.dtype.itemsize
    image_stride = images.strides[0] // images.dtype.itemsize
    footprint_mgrid = np.mgrid[
        [slice(-o, d - o) for d, o in zip(footprint.shape, offset)]
    ]
    footprint_offsets = footprint_mgrid[:, footprint].transpose()
    nb_strides = np.array(
        [
            np.sum(value_stride * footprint_offset)
            for footprint_offset in footprint_offsets
        ],
        signed_int_dtype,
    )

    # CuPy Backend: changed flatten to ravel to avoid copy
    images = images.ravel()

    # Erosion goes smallest to largest; dilation goes largest to smallest.
    index_sorted = cp.argsort(images).astype(signed_int_dtype, copy=False)
    if method == "dilation":
        index_sorted = index_sorted[::-1]

    # Make a linked list of pixels sorted by value. -1 is the list terminator.
    index_sorted = cp.asnumpy(index_sorted)
    prev = np.full(isize, -1, signed_int_dtype)
    next = np.full(isize, -1, signed_int_dtype)
    prev[index_sorted[1:]] = index_sorted[:-1]
    next[index_sorted[:-1]] = index_sorted[1:]

    # Cython inner-loop compares the rank of pixel values.
    if method == "dilation":
        value_rank, value_map = rank_order(images)
    elif method == "erosion":
        value_rank, value_map = rank_order(-images)
        value_map = -value_map

    start = index_sorted[0]
    value_rank = cp.asnumpy(value_rank.astype(unsigned_int_dtype, copy=False))
    reconstruction_loop(value_rank, prev, next, nb_strides, start, image_stride)

    # Reshape reconstructed image to original image shape and remove padding.
    value_rank = cp.asarray(value_rank[:image_stride])

    rec_img = value_map[value_rank]
    rec_img = rec_img.reshape(dims[1:])
    return rec_img[inside_slices]
