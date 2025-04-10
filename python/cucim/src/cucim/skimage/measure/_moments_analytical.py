import itertools
import math

import cupy as cp
import numpy as np

from ._regionprops_gpu_moments_kernels import (
    regionprops_moments_central,
)


def moments_raw_to_central(moments_raw):
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    if ndim in [2, 3] and order < 4:
        # fast path with analytical GPU kernels
        # (avoids any host/device transfers)

        # have to temporarily prepend a "labels" dimension to reuse
        # regionprops_moments_central
        moments_central = regionprops_moments_central(
            moments_raw[cp.newaxis, ...], ndim
        )[0]
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
