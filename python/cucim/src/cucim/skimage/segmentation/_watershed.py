# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded watershed segmentation by parallel path-cost relaxation.

The algorithm combines the marker-controlled flooding watershed, the
lexicographic watershed path cost of Lotufo and Falcão, cellular-automaton
style GPU relaxation and block-asynchronous shared-memory tiling with a few
choices specific to this implementation. See the Notes and References of
`watershed` for how each part relates to the literature, and
`_watershed_ca.py` and `_watershed_ca_block_async.py` for the kernels.
"""

import operator
import warnings

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi

from ..morphology.extrema import local_minima
from ..util._regular_grid import regular_seeds
from ._watershed_ca import (
    _DTYPE_TO_CTYPE,
    _FLAT_INDEX_CODE,
    _KERNEL_PREAMBLE,
    _generate_neighbor_loop,
    _init_keys,
    _labels_from_keys,
    _relax_keys_global,
    _watershed_compact,
)
from ._watershed_ca_block_async import _relax_keys_block_async

#########################################
# Watershed line post-processing kernel #
#########################################


@cp.memoize(for_each_device=True)
def _get_watershed_line_kernel(ndim, connectivity=1, label_ctype="int32_t"):
    """Get post-processing kernel to create 1-pixel watershed lines.

    After the relaxation has converged and all pixels have final labels,
    this kernel identifies boundary pixels. To produce 1-pixel-wide lines
    (rather than 2-pixel), only the pixel that was reached "later" in the
    flooding (larger key) is set to 0. For equal keys, the pixel with the
    higher label value is zeroed as a consistent tie-breaker. Marker pixels
    are copied through unchanged, so (as in scikit-image) adjacent marker
    regions may not have a separating line.

    Uses a 1D grid for all dimensionalities.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    connectivity : int
        Neighborhood connectivity.
    label_ctype : str
        C type string for label arrays (e.g. "int32_t", "int8_t").
    """
    L = label_ctype
    dim_params, size_expr, coord_code, neighbor_loop = _generate_neighbor_loop(
        ndim,
        connectivity,
        f"""{L} nlabel = labels_in[nidx];
            if (nlabel != 0 && nlabel != my_label) {{
                unsigned long long nkey = keys[nidx];
                if (nkey < my_key || (nkey == my_key && nlabel < my_label)) {{
                    is_boundary = 1;
                }}
            }}""",
    )

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_line(
    const {L}* __restrict__ labels_in,
    const unsigned long long* __restrict__ keys,
    const {L}* __restrict__ markers,
    {L}* __restrict__ labels_out,
    {dim_params}
) {{
    {_FLAT_INDEX_CODE.format(size_expr=size_expr)}

    {L} my_label = labels_in[idx];
    if (markers[idx] != 0 || my_label == 0) {{
        // Markers are fixed inputs and must never become watershed lines.
        labels_out[idx] = my_label;
        return;
    }}

    {coord_code}

    unsigned long long my_key = keys[idx];
    int is_boundary = 0;
    {neighbor_loop}

    labels_out[idx] = is_boundary ? 0 : my_label;
}}
"""
    )
    return cp.RawKernel(kernel_code, "watershed_line")


def _validate_inputs(image, markers, mask, connectivity):
    """Validate and prepare inputs for watershed algorithm.

    See `watershed` for parameter descriptions.
    """
    # Check image first to get ndim
    if not isinstance(image, cp.ndarray):
        raise TypeError(
            f"image must be a cupy.ndarray, got {type(image).__name__}"
        )

    ndim = image.ndim

    # The kernels use 32-bit flat indices.
    if image.size >= 2**31:
        raise ValueError(
            "watershed supports images with fewer than 2**31 elements, got "
            f"{image.size}"
        )

    # Check connectivity - must be an integer
    if not isinstance(connectivity, (int, np.integer, cp.integer)):
        raise TypeError(
            f"connectivity must be an integer, got {type(connectivity)}"
        )
    connectivity = int(connectivity)

    # Check connectivity value based on dimensionality
    if connectivity not in range(1, ndim + 1):
        raise ValueError(
            f"connectivity must be in [1, {ndim}] for {ndim}D images, "
            f"got {connectivity}"
        )

    # Convert image to float32 for processing
    if image.dtype != cp.float32:
        image = image.astype(cp.float32)

    # Validate mask before markers (needed for marker generation)
    if mask is not None:
        if not isinstance(mask, cp.ndarray):
            raise TypeError(
                f"mask must be a cupy.ndarray, got {type(mask).__name__}"
            )

        if mask.shape != image.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match image shape {image.shape}"
            )

        # Any non-zero value is inside the mask, as in scikit-image.
        mask = mask.astype(bool, copy=False)

    # Handle markers
    if markers is None:
        # Auto-detect markers from local minima (matching scikit-image)
        markers_bool = local_minima(image, connectivity=connectivity)
        if mask is not None:
            markers_bool &= mask
        footprint = ndi.generate_binary_structure(ndim, connectivity)
        markers = ndi.label(markers_bool, structure=footprint)[0]
    elif not isinstance(markers, (cp.ndarray, list, tuple)):
        # Assume int: generate that many regularly-spaced markers.
        # Scale by the fraction of masked pixels (like scikit-image) so the
        # requested count refers to markers *within* the mask. The masked
        # pixel count is only needed here, so the reduction (and its
        # device->host sync) is deferred to this branch.
        n_markers = int(markers)
        if mask is not None:
            n_pixels = int(cp.count_nonzero(mask))
            if n_pixels == 0:
                markers = cp.zeros(image.shape, dtype=cp.int32)
            else:
                n_markers = int(n_markers / (n_pixels / image.size))
                markers = regular_seeds(image.shape, n_markers)
                markers = cp.where(mask, markers, 0)
        else:
            markers = regular_seeds(image.shape, n_markers)
    else:
        if not isinstance(markers, cp.ndarray):
            raise TypeError(
                f"markers must be a cupy.ndarray or int, "
                f"got {type(markers).__name__}"
            )

        if markers.shape != image.shape:
            raise ValueError(
                f"markers shape {markers.shape} must match "
                f"image shape {image.shape}"
            )

        if mask is not None:
            markers = cp.where(mask, markers, 0)

    return image, markers, mask, connectivity


def watershed(
    image,
    markers=None,
    connectivity=1,
    offset=None,
    mask=None,
    compactness=0,
    watershed_line=False,
    *,
    use_block_async=None,
    inner_iterations=None,
    convergence_check_interval=None,
):
    """Watershed segmentation using cellular-automaton-style relaxation.

    This function implements a GPU-accelerated watershed transform using
    CA-inspired parallel relaxation of path costs. It follows the
    scikit-image watershed API rather than reproducing one paper verbatim,
    supporting arbitrary markers, masks, n-dimensional connectivity, watershed
    lines and compact watershed.

    Parameters
    ----------
    image : cupy.ndarray, shape (M, N[, ...])
        Input image (typically a gradient magnitude or distance transform).
        Lower values have higher priority for watershed expansion.
        n-dimensional images with fewer than ``2**31`` elements are
        supported. The image is processed in single precision (float32).
    markers : int, cupy.ndarray of int, or None, optional
        The desired number of basins, or an array marking the basins with
        the values to be assigned in the label matrix. Zero means not a
        marker. If None (default), markers are determined as the local
        minima of the image, using
        :func:`cucim.skimage.morphology.local_minima` followed by
        :func:`cupyx.scipy.ndimage.label` (with the same given
        `connectivity`). If an int, that many regularly-spaced markers
        are generated using :func:`cucim.skimage.util.regular_seeds`.
        If an array, non-zero values represent different regions to grow
        from. Negative markers are supported (e.g., -1 for background).
        Unlike scikit-image, Python ``list``/``tuple`` markers are not
        accepted; pass a :class:`cupy.ndarray` or an int.
    connectivity : int, optional
        Neighborhood connectivity as integer:
        - For 2D images:
          - 1: 4-connectivity (von Neumann neighborhood - orthogonal neighbors)
          - 2: 8-connectivity (Moore neighborhood - includes diagonals)
        - For 3D images:
          - 1: 6-connectivity (face neighbors)
          - 2: 18-connectivity (face + edge neighbors)
          - 3: 26-connectivity (face + edge + corner neighbors)
        Default is 1. Unlike scikit-image, only an integer is accepted; a
        connectivity given as an ndarray footprint raises ``TypeError``.
    offset : array_like of shape image.ndim, optional
        The coordinates of the center of the connectivity footprint. This
        parameter is accepted for compatibility with the scikit-image API,
        but only the default (``None``, i.e. a footprint centered on each
        pixel) is supported. Passing any other value raises
        :class:`NotImplementedError`.
    mask : cupy.ndarray of bool, shape (M, N[, ...]), optional
        If provided, only pixels/voxels where mask is True (non-zero) will be
        segmented. Useful for restricting watershed to regions of interest.
        Default is None (the whole image is segmented).
    compactness : float, optional
        Use compact watershed with given compactness parameter.
        Higher values give more regularly-shaped basins. When compactness
        is positive, the priority is computed as:
            priority = (
                image[pixel] + compactness * euclidean_distance(pixel, marker)
            )
        Typical values range from 0.001 to 1.0. Default is 0 (standard
        watershed based purely on image values).
    watershed_line : bool, optional
        If True, a one-pixel wide line separates the regions obtained by
        the watershed algorithm. The line has the label 0. This is
        implemented as a post-processing step after the relaxation
        converges (see Notes). Default is False.
    use_block_async : bool or None, optional
        Selects the kernel used for standard watershed (``compactness=0``).
        If True, a block-asynchronous kernel relaxes shared-memory tiles
        (plus halo) for many local iterations per launch; it is available
        for images with at most three dimensions of size greater than one
        (dimensions of size one are dropped and a 1D image is processed as
        a single row). If False, a global-memory kernel performing one
        relaxation step per launch is used. If None (default), the
        block-asynchronous kernel is used whenever it is available and the
        global kernel otherwise. Both kernels converge to the same unique
        solution, so this setting only affects performance. Passing True
        when it does not apply (``compactness > 0``, or more than three
        dimensions of size greater than one) warns and falls back to the
        global kernel.
    inner_iterations : int or None, optional
        Maximum number of shared-memory iterations each block performs per
        launch of the block-asynchronous kernel. A block stops early once
        its tile stops changing. If None (default), uses 64 for 2D, 32 for
        3D, and 512 for single-row images (such as 1D images), which use
        1 x 512 tiles. Only affects performance, and has no effect when the
        block-asynchronous kernel is not used.
    convergence_check_interval : int or None, optional
        Number of relaxation kernel launches to perform between successive
        convergence checks. Each check reads a device flag back to the host,
        which forces a synchronization and serializes the CPU and GPU.
        Running several launches back-to-back before checking reduces these
        host-device syncs by roughly this factor, at the cost of up to
        ``convergence_check_interval - 1`` extra launches after convergence.
        Because the relaxation is idempotent at its fixed point, those extra
        launches are no-ops and never change the result. If None (default),
        a value tuned to the kernel is used: 8 for the global kernels and 4
        for the block-asynchronous kernel, whose launches past convergence
        are cheap because every tile is then skipped. Must be >= 1 if given.

    Returns
    -------
    labels : cupy.ndarray of int, shape (M, N[, ...])
        Labeled array of the same shape as ``image``. Each basin is labeled
        with the value of its marker (negative marker values are preserved),
        and the output dtype matches the marker dtype.

    Raises
    ------
    TypeError
        If an input has an unsupported type or `markers` does not have a
        signed or unsigned integer dtype.
    ValueError
        If input arrays have incompatible shapes or invalid parameters.
    NotImplementedError
        If a non-default ``offset`` is given (only ``offset=None`` is
        supported).

    Notes
    -----
    This implementation combines ideas from several publications with a few
    choices of its own; it is not an implementation of any single published
    algorithm.

    - **Model.** Marker-controlled watershed segmentation [1]_ computed by
      flooding from the markers [2]_, with the semantics of scikit-image's
      priority-queue implementation: pixels are flooded in order of value,
      and ties are broken first-in-first-out.
    - **Path cost.** That flooding order is expressed as the lexicographic
      path cost of Lotufo and Falcão [3]_, written explicitly as
      ``C = [M, d]`` by Felkel et al. [4]_. Each pixel carries a key
      ``(L, D)``: ``L`` is the minimax path cost from a marker (the flooding
      level at which the pixel is reached) and ``D`` is the number of steps
      since the path last rose to that level. The labels are the root labels
      of the resulting optimum-path forest, as in the image foresting
      transform [5]_.
    - **Parallel relaxation.** Instead of using a priority queue, every
      pixel repeatedly takes the best key offered by its neighbors, in the
      spirit of the Ford-Bellman cellular-automaton watersheds of Kauffmann
      and Piché [6]_, [7]_. (Both relax a summed path cost, a weighted or
      topographic distance, rather than the minimax cost used here, with
      synchronous double-buffered updates.)
    - **Block-asynchronous tiling.** For images with at most three
      dimensions of size greater than one (dimensions of size one are
      dropped, and a 1D image is processed as a single row), a tile plus a
      one-pixel halo is relaxed in shared memory until it stops changing,
      and the kernel is relaunched until no tile changes, following the
      block-asynchronous cellular automaton of Quesada-Barriuso et al.
      (Section 4.2 of [8]_). Their watershed is unseeded (it floods from all
      regional minima) and updates each tile in synchronous steps with its
      halo held fixed. Because data then propagate at different speeds
      inside and across tiles, watershed lines on plateaus are misplaced,
      which their artifact-free variant (Section 4.3) corrects with a
      separately computed distance to the plateau's lower border. Here, keys are updated in
      place, so threads and tiles may read keys written by others in the
      same step, plateaus are ordered by ``D``, and no correction is needed
      because the result does not depend on the update order (see below).
    - **Label resolution.** After the keys converge, each pixel's parent is
      its first neighbor with the smallest key, and the parent chains are
      resolved to the markers by pointer jumping [9]_, as in the
      path-reducing GPU watershed of Yeghiazaryan and Voiculescu [10]_,
      which reduces steepest-descent label paths to direct pointers with
      ``label(p) = label(label(p))``, a bounded number of times between
      synchronizations. Here each thread walks its own chain for at most 16
      steps per launch, replacing every pointer it passes with its
      grandparent (path halving), so most images need one or two launches.
    - **Skipping quiet regions.** Pixels (global kernel) and tiles
      (block-asynchronous kernel) are skipped unless a neighbor improved its
      key in the previous or current launch, similar to the active-vertex
      masks of GPU shortest-path relaxation [11]_ and the pixel activation
      map of the GPU image foresting transform [12]_.
    - **Specific to this implementation.** ``D``, which a sequential
      FIFO queue keeps implicitly [4]_, is stored explicitly and packed
      together with an order-preserving encoding of ``L`` into one 64-bit
      integer, so keys are compared and updated as single integers and the
      relaxation has a unique fixed point that any update order reaches.
      Quiet tiles are detected with per-tile launch stamps.

    Because the key strictly increases along every step, the relaxation has
    a unique fixed point that does not depend on the order of updates. The
    block-asynchronous and global kernels, and repeated runs, therefore give
    identical labels even though updates are applied in place without
    synchronization, and ``use_block_async``, ``inner_iterations`` and
    ``convergence_check_interval`` only affect performance. Each pixel takes
    the label of its first neighbor (in a fixed neighbor order) with the
    smallest key. The result matches scikit-image except where neighbors are
    tied in both ``L`` and ``D`` (the tie zones of the optimum-path forest,
    for example between equidistant markers on a plateau), where the fixed
    neighbor order rather than the full priority-queue insertion order
    decides, and where rounding the image to float32 merges distinct values.

    The compact watershed extension follows the scikit-image approach [13]_,
    adding a distance penalty to encourage more regularly-shaped regions. This
    is useful for superpixel generation. Because the cost of reaching a pixel
    then depends on the marker it is reached from, the keys no longer have a
    unique fixed point, so compact watershed uses a double-buffered (Jacobi)
    relaxation of labels, keys and source markers instead, again skipping
    pixels whose neighborhood did not change. Candidates with equal priority
    are ranked by the offering neighbor's key. The result is deterministic,
    but because a pixel keeps the source marker it was first reached from
    with its final priority, it can differ more from scikit-image near region
    boundaries and can leave small groups of pixels disconnected from the
    rest of their region.

    Watershed lines are computed after convergence: a non-marker pixel
    becomes part of the line if a neighbor with a different label was reached
    earlier (smaller key), with ties broken by label value. scikit-image
    instead creates lines during flooding, so line pixels can differ.

    References
    ----------
    .. [1] Meyer, F., & Beucher, S. (1990). Morphological segmentation.
           Journal of Visual Communication and Image Representation, 1(1),
           pp. 21-46.
           :DOI:`10.1016/1047-3203(90)90014-M`
    .. [2] Beucher, S., & Meyer, F. (1993). The morphological approach to
           segmentation: the watershed transformation. In E. R. Dougherty
           (Ed.), Mathematical Morphology in Image Processing, Marcel Dekker
           Inc., New York, pp. 433-481.
    .. [3] Lotufo, R., & Falcão, A. (2000). The ordered queue and the
           optimality of the watershed approaches. In Mathematical Morphology
           and its Applications to Image and Signal Processing (ISMM).
           Computational Imaging and Vision, vol 18. Springer, Boston, MA,
           pp. 341-350.
           :DOI:`10.1007/0-306-47025-X_37`
    .. [4] Felkel, P., Bruckschwaiger, M., & Wegenkittl, R. (2001).
           Implementation and complexity of the watershed-from-markers
           algorithm computed as a minimal cost forest. Computer Graphics
           Forum, 20(3), pp. 26-35.
           :DOI:`10.1111/1467-8659.00495`
    .. [5] Falcão, A. X., Stolfi, J., & Lotufo, R. A. (2004). The image
           foresting transform: theory, algorithms, and applications. IEEE
           Transactions on Pattern Analysis and Machine Intelligence, 26(1),
           pp. 19-29.
           :DOI:`10.1109/TPAMI.2004.1261076`
    .. [6] Kauffmann, C., & Piché, N. (2008). Cellular automaton for ultra-fast
           watershed transform on GPU, 2008 19th International Conference on
           Pattern Recognition (ICPR), Tampa, FL, USA, 2008, pp. 1-4.
           :DOI:`10.1109/ICPR.2008.4761628`
    .. [7] Kauffmann, C., & Piché, N. (2010). Seeded ND medical image
           segmentation by cellular automaton on GPU, International Journal of
           Computer Assisted Radiology and Surgery, Volume 5, Issue 3, 2010,
           pp. 251-262.
           :DOI:`10.1007/s11548-009-0392-0`
    .. [8] P. Quesada-Barriuso, D.B. Heras, F. Argüello, Efficient 2D and 3D
           watershed on graphics processing unit: block-asynchronous approaches
           based on cellular automata, Computers & Electrical Engineering,
           Volume 39, Issue 8, 2013, pp. 2638-2655, ISSN 0045-7906,
           :DOI:`10.1016/j.compeleceng.2013.04.020`
    .. [9] https://en.wikipedia.org/wiki/Pointer_jumping
    .. [10] Yeghiazaryan, V., & Voiculescu, I. (2018). Path reducing
            watershed for the GPU. In 2018 IEEE Winter Conference on
            Applications of Computer Vision (WACV), pp. 577-585.
            :DOI:`10.1109/WACV.2018.00069`
    .. [11] Harish, P., & Narayanan, P. J. (2007). Accelerating large graph
            algorithms on the GPU using CUDA. In High Performance Computing
            (HiPC 2007), Lecture Notes in Computer Science, vol. 4873,
            pp. 197-208.
            :DOI:`10.1007/978-3-540-77220-0_21`
    .. [12] Gonçalves, H., de Vasconcelos, G. Q., Rangel, P., Carvalho, M.,
            Archilha, N., & Spina, T. (2019). cudaIFT: 180x faster image
            foresting transform for waterpixel estimation using CUDA. In
            Proceedings of the 14th International Joint Conference on Computer
            Vision, Imaging and Computer Graphics Theory and Applications
            (VISAPP), pp. 395-404.
            :DOI:`10.5220/0007402703950404`
    .. [13] Neubert, P., & Protzel, P. (2014). Compact Watershed and
            Preemptive SLIC: On Improving Trade-offs of Superpixel
            Segmentation Algorithms, 2014 22nd International Conference on
            Pattern Recognition (ICPR), Stockholm, Sweden, 2014, pp. 996-1001
            :DOI:`10.1109/ICPR.2014.181`

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import segmentation
    >>> from cupyx.scipy import ndimage as ndi

    Create a simple test image with two peaks:

    >>> image = cp.zeros((10, 10), dtype=cp.float32)
    >>> image[2, 2] = 1
    >>> image[7, 7] = 1
    >>> image = ndi.gaussian_filter(image, sigma=1.0)

    Create markers at the peaks:

    >>> markers = cp.zeros((10, 10), dtype=cp.int32)
    >>> markers[2, 2] = 1
    >>> markers[7, 7] = 2

    Apply watershed:

    >>> labels = segmentation.watershed(-image, markers)
    >>> labels.shape
    (10, 10)
    >>> cp.unique(labels)
    array([1, 2], dtype=int32)

    Apply compact watershed for more regular regions:

    >>> labels_compact = segmentation.watershed(-image, markers,
    ...                                         compactness=0.1)
    """
    if offset is not None:
        # `offset` is accepted only to keep the positional argument order
        # consistent with scikit-image. The CA kernels use symmetric
        # neighborhoods centered on each pixel, so a shifted footprint
        # center is not supported.
        raise NotImplementedError(
            "The `offset` parameter is accepted for scikit-image API "
            "compatibility but only the default (offset=None) is supported."
        )

    if inner_iterations is not None:
        try:
            inner_iterations = operator.index(inner_iterations)
        except TypeError as e:
            raise TypeError(
                "inner_iterations must be an integer or None, got "
                f"{type(inner_iterations).__name__}"
            ) from e
        if inner_iterations < 1:
            raise ValueError(
                "inner_iterations must be a positive integer, got "
                f"{inner_iterations}"
            )

    if convergence_check_interval is not None:
        convergence_check_interval = int(convergence_check_interval)
        if convergence_check_interval < 1:
            raise ValueError(
                "convergence_check_interval must be a positive integer, got "
                f"{convergence_check_interval}"
            )

    # Determine output label dtype from markers
    if isinstance(markers, cp.ndarray):
        label_dtype = cp.dtype(markers.dtype)
        if label_dtype not in _DTYPE_TO_CTYPE:
            raise TypeError(
                "markers must have a signed or unsigned integer dtype, got "
                f"{label_dtype}"
            )
    else:
        label_dtype = cp.dtype(cp.int32)

    # Validate and prepare inputs
    image, markers, mask, connectivity = _validate_inputs(
        image, markers, mask, connectivity
    )

    # User-provided marker dtypes were validated above. Keep internally
    # generated or mask-adjusted markers consistent with the kernel type.
    if markers.dtype != label_dtype:
        markers = markers.astype(label_dtype)

    compact = compactness > 0
    ndim = image.ndim
    work_shape = image.shape
    work_connectivity = connectivity
    if not compact:
        # Standard watershed runs on an equivalent shape: dimensions of size
        # one have no in-bounds neighbors, so dropping them leaves every
        # neighborhood unchanged, and a 1D image is relaxed as a single row.
        # This lets such images use the block-asynchronous kernel, and is
        # applied for every kernel so that they break exact ties identically.
        work_shape = tuple(s for s in image.shape if s != 1)
        work_shape = (1,) * max(0, 2 - len(work_shape)) + work_shape
        work_connectivity = min(connectivity, len(work_shape))
    work_ndim = len(work_shape)

    # The block-asynchronous kernel is only implemented for standard
    # watershed with 2 or 3 effective dimensions. An explicit request that
    # cannot be honored warns and falls back to the global kernel.
    block_async_supported = not compact and work_ndim in (2, 3)
    if use_block_async is None:
        use_block_async = block_async_supported
    elif use_block_async and not block_async_supported:
        if compact:
            reason = "use_block_async is not supported with compactness > 0"
        else:
            reason = (
                "use_block_async is only supported for images with at most "
                f"3 dimensions of size > 1, got shape {image.shape}"
            )
        warnings.warn(
            f"{reason}; falling back to the global kernel.", stacklevel=2
        )
        use_block_async = False

    image_flat = cp.ascontiguousarray(image).ravel()
    markers_flat = cp.ascontiguousarray(markers).ravel()
    if mask is not None:
        mask_flat = cp.ascontiguousarray(mask).view(cp.uint8).ravel()
        has_mask = 1
    else:
        # Never read when has_mask is 0.
        mask_flat = cp.empty(1, dtype=cp.uint8)
        has_mask = 0

    # Every relaxation launch reaches at least one more step along each
    # optimal path, so image.size launches always suffice. The loops exit as
    # soon as they converge, so this ceiling has no effect in practice.
    max_iterations = int(image.size)

    if compact:
        labels, keys = _watershed_compact(
            image_flat,
            markers_flat,
            mask_flat,
            has_mask,
            connectivity,
            ndim,
            image.shape,
            max_iterations,
            compactness,
            convergence_check_interval=convergence_check_interval,
        )
    else:
        cost, keys = _init_keys(image_flat, markers_flat, mask_flat, has_mask)
        relax_kwargs = dict(
            ndim=work_ndim,
            connectivity=work_connectivity,
            image_shape=work_shape,
            max_iterations=max_iterations,
            convergence_check_interval=convergence_check_interval,
        )
        if use_block_async:
            _relax_keys_block_async(
                cost, keys, inner_iterations=inner_iterations, **relax_kwargs
            )
        else:
            _relax_keys_global(cost, keys, **relax_kwargs)
        labels = _labels_from_keys(
            cost, keys, markers_flat, work_ndim, work_connectivity, work_shape
        )

    # Post-processing: create watershed lines if requested
    if watershed_line:
        labels_out = cp.empty_like(labels)
        wl_kernel = _get_watershed_line_kernel(
            work_ndim,
            work_connectivity,
            label_ctype=_DTYPE_TO_CTYPE[label_dtype],
        )
        threads_per_block = 256
        blocks = (image.size + threads_per_block - 1) // threads_per_block
        wl_kernel(
            (blocks,),
            (threads_per_block,),
            (
                labels,
                keys,
                markers_flat,
                labels_out,
                *(int(s) for s in work_shape),
            ),
        )
        labels = labels_out

    return labels.reshape(image.shape)
