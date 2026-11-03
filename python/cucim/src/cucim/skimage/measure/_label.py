# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import functools

import cupy as cp
import scipy.ndimage as cpu_ndi

from ._label_kernels import _label

_INT32_LABEL_SIZE_LIMIT = 2**31 - 2


def _label_dtype_for_size(size):
    return cp.int64 if size >= _INT32_LABEL_SIZE_LIMIT else cp.int32


@functools.cache
def _get_structure(ndim, connectivity):
    if connectivity is None:
        # use the full connectivity by default
        connectivity = ndim
    if not 1 <= connectivity <= ndim:
        raise ValueError(f"Connectivity below 1 or above {ndim} is illegal.")
    return cpu_ndi.generate_binary_structure(ndim, connectivity)


def label(label_image, background=None, return_num=False, connectivity=None):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    label_image : ndarray of dtype int
        Image to label.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    cucim.skimage.measure.regionprops
    cucim.skimage.measure.regionprops_table

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864
    .. [3] J. Jaiganesh and M. Burtscher, "A high-performance
           connected components implementation for GPUs", Proceedings of the
           27th International Symposium on High-Performance Parallel and
           Distributed Computing (HPDC '18), 2018, pp. 92-104.
           https://doi.org/10.1145/3208040.3208041
    .. [4] Y. Komura, "A generalized GPU-based connected component
           labeling algorithm", 2016. :arxiv:`1603.08357`
    .. [5] S. Allegretti, F. Bolelli and C. Grana,
           "Optimized block-based algorithms to label connected components on
           GPUs", IEEE Transactions on Parallel and Distributed Systems 31(2),
           2020, pp. 423-438. https://doi.org/10.1109/TPDS.2019.2934683
    .. [6] F. Bolelli, S. Allegretti, L. Lumetti and C. Grana, "A
           state-of-the-art review with code about connected components
           labeling on GPUs", IEEE Transactions on Parallel and Distributed
           Systems 37(4), 2026 (published online 2024), pp. 885-904.
           https://doi.org/10.1109/TPDS.2024.3434357

    Notes
    -----
    The cucim implementation of this function uses 32-bit integers for the
    label array unless the input size requires 64-bit labels.

    This implementation started from the one in CuPy
    (``cupyx.scipy.ndimage.label``), for which no literature reference was
    specified. The general type of approach (a union-find on the GPU) was
    determined by inspecting that implementation. The references [3]_, [4]_
    and [5]_ describe this type of approach and the later changes listed here.

    The regions are found with a concurrent union-find on the GPU in which
    the root of each tree is the first pixel of its region (see [3]_ for the
    compare-and-swap union of roots and the path halving, and [4]_ and [5]_
    for linking each pixel to a preceding neighbor before the unions). The
    blocks described below are the approach of the block-based union-find
    (BUF) of [5]_. See [6]_ for a review and comparison of GPU connected
    component labeling algorithms.

    For the best performance, pass a boolean image and use full connectivity
    (the default, ``connectivity=None``, or ``connectivity=label_image.ndim``)
    in 2D or 3D. In that case each 2 x 2 (or 2 x 2 x 2) block of pixels is
    handled as a single node, which is typically 2 to 6 times faster than the
    general implementation (the gain is largest for dense images and in 3D).
    The general implementation is used for integer images (also if they only
    contain 0 and 1, so convert those with ``label_image.astype(bool)``), for a
    ``connectivity`` below ``label_image.ndim``, for 1D and for 4D or higher
    dimensional images, and when ``background`` is neither 0 nor, for a boolean
    image, 1 (``True``). The labels are the
    same for both implementations.

    Examples
    --------
    >>> import cupy as cp
    >>> x = cp.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=-1))
    [[1 2 2]
     [2 1 2]
     [2 2 1]]
    >>> x = cp.asarray([[1, 0, 0],
    ...                 [1, 1, 5],
    ...                 [0, 0, 0]])
    >>> print(label(x))
    [[1 0 0]
     [1 1 2]
     [0 0 0]]
    """
    ndim = label_image.ndim
    structure = _get_structure(ndim, connectivity)
    if background is None:
        background = 0
    elif label_image.dtype == cp.bool_ and background == 1:
        # the foreground of a boolean image is the False pixels (this keeps
        # the boolean image, which is labeled faster than an integer image)
        label_image = ~label_image
    elif background != 0:
        # offset so that background becomes 0 as expected by _label below
        label_image = label_image - background

    if label_image.dtype.kind not in "bui":
        # skimage always copies the input into a np.intp dtype array so do the
        # same here for non-integer dtypes.
        label_image = label_image.astype(cp.intp)

    labels = cp.empty(
        label_image.shape,
        order="C",
        dtype=_label_dtype_for_size(label_image.size),
    )
    num = _label(label_image, structure, labels, greyscale_mode=True)

    if return_num:
        return labels, num
    return labels
