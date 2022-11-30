import itertools
import math

import cupy as cp
import numpy as np

_order0_or_1 = """
    mc[0] = m[0];
"""

_order2_2d = """
    /* Implementation of the commented code below with C-order raveled
     * indices into 3 x 3 matrices, m and mc.
     *
     * mc[0, 0] = m[0, 0];
     * cx = m[1, 0] / m[0, 0];
     * cy = m[0, 1] / m[0, 0];
     * mc[1, 1] = m[1, 1] - cx*m[0, 1];
     * mc[2, 0] = m[2, 0] - cx*m[1, 0];
     * mc[0, 2] = m[0, 2] - cy*m[0, 1];
     */
    mc[0] = m[0];
    F cx = m[3] / m[0];
    F cy = m[1] / m[0];
    mc[4] = m[4] - cx*m[1];
    mc[6] = m[6] - cx*m[3];
    mc[2] = m[2] - cy*m[1];
"""

_order3_2d = """
    /* Implementation of the commented code below with C-order raveled
     * indices into 4 x 4 matrices, m and mc.
     *
     * mc[0, 0] = m[0, 0];
     * cx = m[1, 0] / m[0, 0];
     * cy = m[0, 1] / m[0, 0];
     * mc[1, 1] = m[1, 1] - cx*m[0, 1];
     * mc[2, 0] = m[2, 0] - cx*m[1, 0];
     * mc[0, 2] = m[0, 2] - cy*m[0, 1];
     * mc[2, 1] = (m[2, 1] - 2*cx*m[1, 1] - cy*m[2, 0] + cx*cx*m[0, 1] + cy*cx*m[1, 0]);
     * mc[1, 2] = (m[1, 2] - 2*cy*m[1, 1] - cx*m[0, 2] + 2*cy*cx*m[0, 1]);
     * mc[3, 0] = m[3, 0] - 3*cx*m[2, 0] + 2*cx*cx*m[1, 0];
     * mc[0, 3] = m[0, 3] - 3*cy*m[0, 2] + 2*cy*cx*m[0, 1];
     */

    mc[0] = m[0];
    F cx = m[4] / m[0];
    F cy = m[1] / m[0];
    // 2nd order moments
    mc[5] = m[5] - cx*m[1];
    mc[8] = m[8] - cx*m[4];
    mc[2] = m[2] - cy*m[1];
    // 3rd order moments
    mc[9] = (m[9] - 2*cx*m[5] - cy*m[8] + cx*cx*m[1] + cy*cx*m[4]);
    mc[6] = (m[6] - 2*cy*m[5] - cx*m[2] + 2*cy*cx*m[1]);
    mc[12] = m[12] - 3*cx*m[8] + 2*cx*cx*m[4];
    mc[3] = m[3] - 3*cy*m[2] + 2*cy*cy*m[1];
"""  # noqa


# Note for 2D kernels using C-order raveled indices
_order2_3d = """
    /* Implementation of the commented code below with C-order raveled
     * indices into shape (3, 3, 3) matrices, m and mc.
     *
     * mc[0, 0, 0] = m[0, 0, 0];
     * cx = m[1, 0, 0] / m[0, 0, 0];
     * cy = m[0, 1, 0] / m[0, 0, 0];
     * cz = m[0, 0, 1] / m[0, 0, 0];
     * mc[0, 0, 2] = -cz*m[0, 0, 1] + m[0, 0, 2];
     * mc[0, 1, 1] = -cy*m[0, 0, 1] + m[0, 1, 1];
     * mc[0, 2, 0] = -cy*m[0, 1, 0] + m[0, 2, 0];
     * mc[1, 0, 1] = -cx*m[0, 0, 1] + m[1, 0, 1];
     * mc[1, 1, 0] = -cx*m[0, 1, 0] + m[1, 1, 0];
     * mc[2, 0, 0] = -cx*m[1, 0, 0] + m[2, 0, 0];
     */
    mc[0] = m[0];
    F cx = m[9] / m[0];
    F cy = m[3] / m[0];
    F cz = m[1] / m[0];
    // 2nd order moments
    mc[2] = -cz*m[1] + m[2];
    mc[4] = -cy*m[1] + m[4];
    mc[6] = -cy*m[3] + m[6];
    mc[10] = -cx*m[1] + m[10];
    mc[12] = -cx*m[3] + m[12];
    mc[18] = -cx*m[9] + m[18];
"""

_order3_3d = """
    /* Implementation of the commented code below with C-order raveled
     * indices into shape (4, 4, 4) matrices, m and mc.
     *
     * mc[0, 0, 0] = m[0, 0, 0];
     * cx = m[1, 0, 0] / m[0, 0, 0];
     * cy = m[0, 1, 0] / m[0, 0, 0];
     * cz = m[0, 0, 1] / m[0, 0, 0];
     * // 2nd order moments
     * mc[0, 0, 2] = -cz*m[0, 0, 1] + m[0, 0, 2];
     * mc[0, 1, 1] = -cy*m[0, 0, 1] + m[0, 1, 1];
     * mc[0, 2, 0] = -cy*m[0, 1, 0] + m[0, 2, 0];
     * mc[1, 0, 1] = -cx*m[0, 0, 1] + m[1, 0, 1];
     * mc[1, 1, 0] = -cx*m[0, 1, 0] + m[1, 1, 0];
     * mc[2, 0, 0] = -cx*m[1, 0, 0] + m[2, 0, 0];
     * // 3rd order moments
     * mc[0, 0, 3] = (2*cz*cz*m[0, 0, 1] - 3*cz*m[0, 0, 2] + m[0, 0, 3]);
     * mc[0, 1, 2] = (-cy*m[0, 0, 2] + 2*cz*(cy*m[0, 0, 1] - m[0, 1, 1]) + m[0, 1, 2]);
     * mc[0, 2, 1] = (cy*cy*m[0, 0, 1] - 2*cy*m[0, 1, 1] + cz*(cy*m[0, 1, 0] - m[0, 2, 0]) + m[0, 2, 1]);
     * mc[0, 3, 0] = (2*cy*cy*m[0, 1, 0] - 3*cy*m[0, 2, 0] + m[0, 3, 0]);
     * mc[1, 0, 2] = (-cx*m[0, 0, 2] + 2*cz*(cx*m[0, 0, 1] - m[1, 0, 1]) + m[1, 0, 2]);
     * mc[1, 1, 1] = (-cx*m[0, 1, 1] + cy*(cx*m[0, 0, 1] - m[1, 0, 1]) + cz*(cx*m[0, 1, 0] - m[1, 1, 0]) + m[1, 1, 1]);
     * mc[1, 2, 0] = (-cx*m[0, 2, 0] - 2*cy*(-cx*m[0, 1, 0] + m[1, 1, 0]) + m[1, 2, 0]);
     * mc[2, 0, 1] = (cx*cx*m[0, 0, 1] - 2*cx*m[1, 0, 1] + cz*(cx*m[1, 0, 0] - m[2, 0, 0]) + m[2, 0, 1]);
     * mc[2, 1, 0] = (cx*cx*m[0, 1, 0] - 2*cx*m[1, 1, 0] + cy*(cx*m[1, 0, 0] - m[2, 0, 0]) + m[2, 1, 0]);
     * mc[3, 0, 0] = (2*cx*cx*m[1, 0, 0] - 3*cx*m[2, 0, 0] + m[3, 0, 0]);
     */
    mc[0] = m[0];
    F cx = m[16] / m[0];
    F cy = m[4] / m[0];
    F cz = m[1] / m[0];
    // 2nd order moments
    mc[2] = -cz*m[1] + m[2];
    mc[5] = -cy*m[1] + m[5];
    mc[8] = -cy*m[4] + m[8];
    mc[17] = -cx*m[1] + m[17];
    mc[20] = -cx*m[4] + m[20];
    mc[32] = -cx*m[16] + m[32];
    // 3rd order moments
    mc[3] = (2*cz*cz*m[1] - 3*cz*m[2] + m[3]);
    mc[6] = (-cy*m[2] + 2*cz*(cy*m[1] - m[5]) + m[6]);
    mc[9] = (cy*cy*m[1] - 2*cy*m[5] + cz*(cy*m[4] - m[8]) + m[9]);
    mc[12] = (2*cy*cy*m[4] - 3*cy*m[8] + m[12]);
    mc[18] = (-cx*m[2] + 2*cz*(cx*m[1] - m[17]) + m[18]);
    mc[21] = (-cx*m[5] + cy*(cx*m[1] - m[17]) + cz*(cx*m[4] - m[20]) + m[21]);
    mc[24] = (-cx*m[8] - 2*cy*(-cx*m[4] + m[20]) + m[24]);
    mc[33] = (cx*cx*m[1] - 2*cx*m[17] + cz*(cx*m[16] - m[32]) + m[33]);
    mc[36] = (cx*cx*m[4] - 2*cx*m[20] + cy*(cx*m[16] - m[32]) + m[36]);
    mc[48] = (2*cx*cx*m[16] - 3*cx*m[32] + m[48]);
"""  # noqa


def _moments_raw_to_central_fast(moments_raw):
    """Analytical formulae for 2D and 3D central moments of order < 4.

    `moments_raw_to_central` will automatically call this function when
    ndim < 4 and order < 4.

    Parameters
    ----------
    moments_raw : ndarray
        The raw moments.

    Returns
    -------
    moments_central : ndarray
        The central moments.
    """
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    # convert to float64 during the computation for better accuracy
    moments_raw = moments_raw.astype(cp.float64, copy=False)
    moments_central = cp.zeros_like(moments_raw)
    if order >= 4 or ndim not in [2, 3]:
        raise ValueError(
            "This function only supports 2D or 3D moments of order < 4."
        )
    if ndim == 2:
        if order < 2:
            operation = _order0_or_1
        elif order == 2:
            operation = _order2_2d
        elif order == 3:
            operation = _order3_2d
    elif ndim == 3:
        if order < 2:
            operation = _order0_or_1
        elif order == 2:
            operation = _order2_3d
        elif order == 3:
            operation = _order3_3d

    kernel = cp.ElementwiseKernel(
        'raw F m',
        'raw F mc',
        operation=operation,
        name=f"order{order}_{ndim}d_kernel"
    )
    # run a single-threaded kernel, so we can avoid device->host->device copy
    kernel(moments_raw, moments_central, size=1)
    return moments_central


def moments_raw_to_central(moments_raw):
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    if ndim in [2, 3] and order < 4:
        # fast path with analytical GPU kernels
        # (avoids any host/device transfers)
        moments_central = _moments_raw_to_central_fast(moments_raw)
        return moments_central.astype(moments_raw.dtype, copy=False)

    # Fallback to general formula applied on the host
    m = cp.asnumpy(moments_raw)  # synchronize
    moments_central = np.zeros_like(moments_raw)
    # centers as computed in centroid above
    centers = tuple(m[tuple(np.eye(ndim, dtype=int))] / m[(0,) * ndim])

    if ndim == 2:
        # This is the general 2D formula from
        # https://en.wikipedia.org/wiki/Image_moment#Central_moments
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                for i in range(p + 1):
                    term1 = math.comb(p, i)
                    term1 *= (-centers[0]) ** (p - i)
                    for j in range(q + 1):
                        term2 = math.comb(q, j)
                        term2 *= (-centers[1]) ** (q - j)
                        moments_central[p, q] += term1 * term2 * m[i, j]
        return moments_central

    # The nested loops below are an n-dimensional extension of the 2D formula
    # given at https://en.wikipedia.org/wiki/Image_moment#Central_moments

    # iterate over all [0, order] (inclusive) on each axis
    for orders in itertools.product(*((range(order + 1),) * ndim)):
        # `orders` here is the index into the `moments_central` output array
        if sum(orders) > order:
            # skip any moment that is higher than the requested order
            continue
        # loop over terms from `m` contributing to `moments_central[orders]`
        for idxs in itertools.product(*[range(o + 1) for o in orders]):
            val = m[idxs]
            for i_order, c, idx in zip(orders, centers, idxs):
                val *= math.comb(i_order, idx)
                val *= (-c) ** (i_order - idx)
            moments_central[orders] += val

    return cp.asarray(moments_central)
