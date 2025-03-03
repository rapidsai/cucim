import math
import warnings

import cupy as cp
import numpy as np

_distance_on_cpu = True
try:
    from cupyx.scipy.spatial import distance

    try:
        test_points = cp.zeros((10, 2), dtype=cp.int32)
        distance.cdist(test_points, test_points)
        del test_points
        _distance_on_cpu = False
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        print(f"CuPy distance computation test failed with error: {e}")
except ImportError as e:
    print(f"cupyx.scipy.spatial.distance import failed with error: {e}")
    pass

if _distance_on_cpu:
    import scipy.spatial.distance as distance


__all__ = ["pdist_max_blockwise"]


def pdist_max_blockwise(
    coords,
    metric="sqeuclidean",
    *,
    coords_per_block=4000,
    compute_argmax=False,
    cdist_kwargs={},
):
    """Find maximum pointwise distance.

    Computes by processing blocks of coordinates to reduce overall memory
    requirement. The memory used at runtime will be proportional to
    ``coords_per_block**2``.

    A block size of >= 2000 is recommended to avoid poor GPU resource usage
    and to reduce kernel launch overhead.

    Parameters
    ----------
    coords : numpy.ndarray or cupy.ndarray of shape (num_points, ndim)
        The coordinates to process.
    metric : str, optional
        Can be any metric supported by `scipy.spatial.distance.cdist`. The
        default is the squared Euclidean distance (sqeuclidean).
    coords_per_block : bool, optional
        Internally, calls to cdist will be made with subsets of coords where
        the subset size is (coords_per_block, ndim).
    compute_argmax : bool, optional
        If True, the value of the coordinate indices corresponding to the
        maxima is returned as the second return Value. Otherwise that value
        will be ``None``.
    cdist_kwargs = dict, optional
        Can provide any additional kwargs to cdist (e.g. `p` for Minkowski
        norms).

    Returns
    -------
    max : float
        The maximal pairwise distance between the points in `coords`.
    argmax : 2-tuple of int or None
        The pair of indices into coords that corresponds to the maxima. If
        `compute_argmax` is False, this will be None.

    Notes
    -----
    The full `cdist` matrix has ``num_coords**2`` elements. Storing this is
    very wasteful if we  only want to find the maximum pointwise distance. This
    function processes smaller blocks of the overall cdist result, keeping
    track of the maximum value (and corrdsponding indices) across all blocks.

    A schematic of the block sizes processed for an array of 10,000 coordinates
    with a block size of 4000 points is shown below:

      ┌────────┬────────┬────────┐
      │  4000  │  4000  │  2000  │
      │  4000  │  4000  │  2000  │
      ├────────┼────────┼────────┤
      │   X    │  4000  │  2000  │
      │        │  4000  │  2000  │
      ├────────┼────────┼────────┤
      │   X    │   X    │  2000  │
      │        │        │  2000  │
      └────────┴────────┴────────┘

    where due to the symmetry of the pdist matrix, we don't compute the lower
    triangular blocks.

    Note that this function always uses 32-bit floating point precision for the
    distance calculations.
    """

    if _distance_on_cpu:
        warnings.warn(
            "cuVS >= 25.02 or pylibraft < 24.12 must be installed to use "
            "GPU-accelerated pairwise distance computations. Falling back "
            "to SciPy-based CPU implementation."
        )
        xp = np
        coords = cp.asnumpy(coords)
    else:
        xp = cp
        coords = cp.asarray(coords)

    if not isinstance(coords, (np.ndarray, cp.ndarray)):
        raise TypeError("coords must be a numpy or cupy array")

    if coords.ndim != 2:
        raise ValueError(
            f"coords must be a 2-dimensional array, got shape {coords.shape}"
        )

    num_coords, _ = coords.shape
    if num_coords == 0:
        raise RuntimeError("No coordinates to process")

    if "out" in cdist_kwargs:
        raise ValueError(
            "'out' cannot be provided via cdist_kwargs (reserved "
            "for internal use)"
        )

    blocks_per_dim = math.ceil(num_coords / coords_per_block)
    if coords.dtype not in [xp.float32, xp.float64]:
        coords = coords.astype(xp.float32, copy=False)
    if blocks_per_dim > 1:
        # reuse the same temporary storage array for most blocks
        # (last block in row and column may be smaller)
        temp = xp.zeros(
            (coords_per_block, coords_per_block), dtype=coords.dtype
        )
    if not coords.flags.c_contiguous:
        coords = xp.ascontiguousarray(coords)
    max_dist = 0
    for i in range(blocks_per_dim):
        for j in range(blocks_per_dim):
            if j < i:
                # skip symmetric regions
                continue
            sl_m = slice(
                i * coords_per_block,
                min((i + 1) * coords_per_block, num_coords),
            )
            sl_n = slice(
                j * coords_per_block,
                min((j + 1) * coords_per_block, num_coords),
            )
            coords_block1 = coords[sl_m, :]
            coords_block2 = coords[sl_n, :]
            if i < blocks_per_dim - 1 and j < blocks_per_dim - 1:
                distance.cdist(
                    coords_block1,
                    coords_block2,
                    metric=metric,
                    out=temp,
                    **cdist_kwargs,
                )
                current_output = temp
            else:
                # omit out= for the last block as size may be different
                out = distance.cdist(
                    coords_block1, coords_block2, metric=metric, **cdist_kwargs
                )
                current_output = out
            current_max = float(current_output.max())
            if compute_argmax:
                if current_max > max_dist:
                    loc_index = i, j
                    distances_max = current_output
                    max_dist = current_max
            else:
                max_dist = max(current_max, max_dist)
    if compute_argmax:
        # Adjust from intra-block coordinate indices into the global ones
        i, j = loc_index
        loc = np.unravel_index(int(distances_max.argmax()), distances_max.shape)
        loc = (
            int(loc[0]) + i * coords_per_block,
            int(loc[1]) + j * coords_per_block,
        )
    else:
        loc = None
    return max_dist, loc
