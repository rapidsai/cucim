import warnings

import cupy as cp
import cucim.skimage._vendored.ndimage as ndi
import numpy as np

from cucim.core.operations.morphology import distance_transform_edt

from .._shared.utils import check_nD, deprecate_kwarg
from ._medial_axis_lookup import \
    cornerness_table as _medial_axis_cornerness_table
from ._medial_axis_lookup import lookup_table as _medial_axis_lookup_table

# --------- Skeletonization and thinning based on Guo and Hall 1989 ---------


_G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                      0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                      1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
                      0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)


_G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                       0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)


@deprecate_kwarg({"max_iter": "max_num_iter"}, removed_version="23.02.00",
                 deprecated_version="22.02.00")
def thin(image, max_num_iter=None):
    """
    Perform morphological thinning of a binary image.

    Parameters
    ----------
    image : binary (M, N) ndarray
        The image to be thinned.
    max_num_iter : int, number of iterations, optional
        Regardless of the value of this parameter, the thinned image
        is returned immediately if an iteration produces no change.
        If this parameter is specified it thus sets an upper bound on
        the number of iterations performed.

    Returns
    -------
    out : ndarray of bool
        Thinned image.

    See Also
    --------
    medial_axis

    Notes
    -----
    This algorithm [1]_ works by making multiple passes over the image,
    removing pixels matching a set of criteria designed to thin
    connected regions while preserving eight-connected components and
    2 x 2 squares [2]_. In each of the two sub-iterations the algorithm
    correlates the intermediate skeleton image with a neighborhood mask,
    then looks up each neighborhood in a lookup table indicating whether
    the central pixel should be deleted in that sub-iteration.

    References
    ----------
    .. [1] Z. Guo and R. W. Hall, "Parallel thinning with
           two-subiteration algorithms," Comm. ACM, vol. 32, no. 3,
           pp. 359-373, 1989. :DOI:`10.1145/62065.62074`
    .. [2] Lam, L., Seong-Whan Lee, and Ching Y. Suen, "Thinning
           Methodologies-A Comprehensive Survey," IEEE Transactions on
           Pattern Analysis and Machine Intelligence, Vol 14, No. 9,
           p. 879, 1992. :DOI:`10.1109/34.161346`

    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square[0, 1] =  1
    >>> square
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> skel = thin(square)
    >>> skel.astype(np.uint8)
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    # check that image is 2d
    check_nD(image, 2)

    # convert image to uint8 with values in {0, 1}
    skel = cp.asarray(image, dtype=bool).astype(cp.uint8)

    # neighborhood mask
    mask = cp.asarray([[ 8,  4,   2],  # noqa
                       [16,  0,   1],  # noqa
                       [32, 64, 128]], dtype=cp.uint8)

    G123_LUT = cp.asarray(_G123_LUT)
    G123P_LUT = cp.asarray(_G123P_LUT)

    # iterate until convergence, up to the iteration limit
    max_num_iter = max_num_iter or cp.inf
    num_iter = 0
    n_pts_old, n_pts_new = cp.inf, cp.sum(skel)
    while n_pts_old != n_pts_new and num_iter < max_num_iter:
        n_pts_old = n_pts_new

        # perform the two "subiterations" described in the paper
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode="constant")
            # take deletion decision from this subiteration's LUT
            D = cp.take(lut, N)
            # perform deletion
            skel[D] = 0

        n_pts_new = cp.sum(skel)  # count points after thinning
        num_iter += 1

    return skel.astype(bool)


# --------- Skeletonization by medial axis transform --------


def _get_tiebreaker(n, random_seed):
    # CuPy generator doesn't currently have the permutation method, so
    # fall back to cp.random.permutation instead.
    cp.random.seed(random_seed)
    if n < 2 << 31:
        dtype = np.int32
    else:
        dtype = np.intp
    tiebreaker = cp.random.permutation(cp.arange(n, dtype=dtype))
    return tiebreaker


def medial_axis(image, mask=None, return_distance=False, *, random_state=None):
    """Compute the medial axis transform of a binary image.

    Parameters
    ----------
    image : binary ndarray, shape (M, N)
        The image of the shape to be skeletonized.
    mask : binary ndarray, shape (M, N), optional
        If a mask is given, only those elements in `image` with a true
        value in `mask` are used for computing the medial axis.
    return_distance : bool, optional
        If true, the distance transform is returned as well as the skeleton.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is None the `numpy.random.Generator` singleton is
        used.
        If `random_state` is an int, a new ``Generator`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` instance then that
        instance is used.

        .. versionadded:: 0.19

    Returns
    -------
    out : ndarray of bools
        Medial axis transform of the image
    dist : ndarray of ints, optional
        Distance transform of the image (only returned if `return_distance`
        is True)

    See Also
    --------
    skeletonize

    Notes
    -----
    This algorithm computes the medial axis transform of an image
    as the ridges of its distance transform.

    The different steps of the algorithm are as follows
     * A lookup table is used, that assigns 0 or 1 to each configuration of
       the 3x3 binary square, whether the central pixel should be removed
       or kept. We want a point to be removed if it has more than one neighbor
       and if removing it does not change the number of connected components.

     * The distance transform to the background is computed, as well as
       the cornerness of the pixel.

     * The foreground (value of 1) points are ordered by
       the distance transform, then the cornerness.

     * A cython function is called to reduce the image to its skeleton. It
       processes pixels in the order determined at the previous step, and
       removes or maintains a pixel according to the lookup table. Because
       of the ordering, it is possible to process all pixels in only one
       pass.

    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> medial_axis(square).astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """
    try:
        from skimage.morphology._skeletonize_cy import _skeletonize_loop
    except ImportError as e:
        warnings.warn(
            "Could not find required private skimage Cython function:\n"
            "\tskimage.morphology._skeletonize_cy._skeletonize_loop\n"
        )
        raise e

    if mask is None:
        # masked_image is modified in-place later so make a copy of the input
        masked_image = image.astype(bool, copy=True)
    else:
        masked_image = image.astype(bool, copy=True)
        masked_image[~mask] = False

    # Load precomputed lookup table based on three conditions:
    # 1. Keep only positive pixels
    # AND
    # 2. Keep if removing the pixel results in a different connectivity
    # (if the number of connected components is different with and
    # without the central pixel)
    # OR
    # 3. Keep if # pixels in neighborhood is 2 or less
    # Note that this table is independent of the image
    table = _medial_axis_lookup_table

    # Build distance transform
    distance = distance_transform_edt(masked_image)
    if return_distance:
        store_distance = distance.copy()

    # Corners
    # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    # We use a cornerness_table lookup table where the score of a
    # configuration is the number of background (0-value) pixels in the
    # 3x3 neighborhood
    cornerness_table = cp.asarray(_medial_axis_cornerness_table)
    corner_score = _table_lookup(masked_image, cornerness_table)

    # Define arrays for inner loop
    distance = distance[masked_image]
    i, j = cp.where(masked_image)

    # Determine the order in which pixels are processed.
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    tiebreaker = _get_tiebreaker(n=distance.size, random_seed=random_state)
    order = cp.lexsort(
        cp.stack(
            (tiebreaker, corner_score[masked_image], distance),
            axis=0
        )
    )

    # Call _skeletonize_loop on the CPU. It requies a single pass over the
    # full array using a specific pixel order, so cannot be run multithreaded!
    order = cp.asnumpy(order.astype(cp.int32, copy=False))
    table = cp.asnumpy(table.astype(cp.uint8, copy=False))
    i = cp.asnumpy(i).astype(dtype=np.intp, copy=False)
    j = cp.asnumpy(j).astype(dtype=np.intp, copy=False)
    result = cp.asnumpy(masked_image)
    # Remove pixels not belonging to the medial axis
    _skeletonize_loop(result.view(np.uint8), i, j, order, table)
    result = cp.asarray(result.view(bool), dtype=bool)

    if mask is not None:
        result[~mask] = image[~mask]
    if return_distance:
        return result, store_distance
    else:
        return result


def _table_lookup(image, table):
    """
    Perform a morphological transform on an image, directed by its
    neighbors

    Parameters
    ----------
    image : ndarray
        A binary image
    table : ndarray
        A 512-element table giving the transform of each pixel given
        the values of that pixel and its 8-connected neighbors.

    Returns
    -------
    result : ndarray of same shape as `image`
        Transformed image

    Notes
    -----
    The pixels are numbered like this::

      0 1 2
      3 4 5
      6 7 8

    The index at a pixel is the sum of 2**<pixel-number> for pixels
    that evaluate to true.
    """
    #
    # We accumulate into the indexer to get the index into the table
    # at each point in the image
    #
    # max possible value of indexer is 512, so just use int16 dtype
    kernel = cp.array(
        [[256, 128, 64], [32, 16, 8], [4, 2, 1]],
        dtype=cp.int16
    )
    indexer = ndi.convolve(image, kernel, output=np.int16, mode="constant")
    image = table[indexer]
    return image
