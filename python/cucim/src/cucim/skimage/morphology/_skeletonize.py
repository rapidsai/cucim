import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np

from .._shared.utils import check_nD, deprecate_kwarg

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


@deprecate_kwarg({'max_iter': 'max_num_iter'}, removed_version="23.02.00",
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
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = cp.take(lut, N)
            # perform deletion
            skel[D] = 0

        n_pts_new = cp.sum(skel)  # count points after thinning
        num_iter += 1

    return skel.astype(bool)
