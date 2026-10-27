# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math

import cupy as cp


def regular_grid(ar_shape, n_points):
    """Find `n_points` regularly spaced along `ar_shape`.

    The returned points (as slices) should be as close to cubically-spaced as
    possible. Essentially, the points are spaced by the Nth root of the input
    array size, where N is the number of dimensions. However, if an array
    dimension cannot fit a full step size, it is "discarded", and the
    computation is done for only the remaining dimensions.

    Parameters
    ----------
    ar_shape : sequence of int
        The shape of the space embedding the grid. ``len(ar_shape)`` is the
        number of dimensions.
    n_points : int
        The (approximate) number of points to embed in the space.

    Returns
    -------
    slices : tuple of (slice, ...)
        A slice along each dimension of `ar_shape`, such that the intersection
        of all the slices give the coordinates of regularly spaced points.

    Examples
    --------
    >>> import cupy as cp
    >>> ar = cp.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 8)
    >>> g
    (slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    array(8.)
    >>> ar = cp.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 32)
    >>> g
    (slice(2, None, 5), slice(2, None, 5))
    >>> ar[g] = 1
    >>> ar.sum()
    array(32.)
    >>> ar = cp.zeros((3, 20, 40))
    >>> g = regular_grid(ar.shape, 8)
    >>> g
    (slice(1, None, 3), slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    array(8.)
    """
    ar_shape = tuple(int(s) for s in ar_shape)
    ndim = len(ar_shape)

    # argsort of argsort gives the rank (inverse permutation)
    sorted_order = sorted(range(ndim), key=lambda i: ar_shape[i])
    unsort_dim_idxs = [0] * ndim
    for rank, i in enumerate(sorted_order):
        unsort_dim_idxs[i] = rank
    sorted_dims = [ar_shape[i] for i in sorted_order]

    space_size = float(math.prod(ar_shape))
    if space_size <= n_points:
        return (slice(None),) * ndim

    initial_step = (space_size / n_points) ** (1.0 / ndim)
    stepsizes = [initial_step] * ndim

    if any(sd < ss for sd, ss in zip(sorted_dims, stepsizes)):
        for dim in range(ndim):
            stepsizes[dim] = sorted_dims[dim]
            space_size = float(math.prod(sorted_dims[dim + 1 :]))
            fill_val = (space_size / n_points) ** (1.0 / (ndim - dim - 1))
            for j in range(dim + 1, ndim):
                stepsizes[j] = fill_val
            if all(sd >= ss for sd, ss in zip(sorted_dims, stepsizes)):
                break

    starts = [int(s // 2) for s in stepsizes]
    stepsizes = [round(s) for s in stepsizes]
    slices = [
        slice(start, None, step) for start, step in zip(starts, stepsizes)
    ]
    return tuple(slices[i] for i in unsort_dim_idxs)


@cp.memoize(for_each_device=True)
def _get_regular_seeds_kernel(ndim):
    """Get a kernel that fills a regular seed image in a single launch.

    Each thread handles one pixel: if the pixel falls on the regular grid
    it gets a sequential label (1-based), otherwise it gets 0.
    """
    if ndim == 2:
        kernel_code = r"""
extern "C" __global__
void regular_seeds_2d(
    int* out,
    int height, int width,
    int start0, int step0,
    int start1, int step1,
    int n_per_row
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = height * width;
    if (idx >= size) return;

    int r = idx / width;
    int c = idx % width;

    // Check if this pixel is on the grid
    if (r >= start0 && c >= start1 &&
        (r - start0) % step0 == 0 &&
        (c - start1) % step1 == 0) {
        int gr = (r - start0) / step0;
        int gc = (c - start1) / step1;
        out[idx] = 1 + gr * n_per_row + gc;
    } else {
        out[idx] = 0;
    }
}
"""
        return cp.RawKernel(kernel_code, "regular_seeds_2d")
    elif ndim == 3:
        kernel_code = r"""
extern "C" __global__
void regular_seeds_3d(
    int* out,
    int depth, int height, int width,
    int start0, int step0,
    int start1, int step1,
    int start2, int step2,
    int n_per_plane, int n_per_row
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = depth * height * width;
    if (idx >= size) return;

    int d = idx / (height * width);
    int r = (idx / width) % height;
    int c = idx % width;

    if (d >= start0 && r >= start1 && c >= start2 &&
        (d - start0) % step0 == 0 &&
        (r - start1) % step1 == 0 &&
        (c - start2) % step2 == 0) {
        int gd = (d - start0) / step0;
        int gr = (r - start1) / step1;
        int gc = (c - start2) / step2;
        out[idx] = 1 + gd * n_per_plane + gr * n_per_row + gc;
    } else {
        out[idx] = 0;
    }
}
"""
        return cp.RawKernel(kernel_code, "regular_seeds_3d")
    else:
        raise NotImplementedError(
            f"regular_seeds kernel only supports 2D and 3D, got {ndim}D"
        )


def _ceildiv(a, b):
    return (a + b - 1) // b


def regular_seeds(ar_shape, n_points, dtype=cp.int32):
    """Return an image with ~`n_points` regularly-spaced nonzero pixels.

    Parameters
    ----------
    ar_shape : tuple of int
        The shape of the desired output image.
    n_points : int
        The desired number of nonzero points.
    dtype : dtype-like, optional
        The desired data type of the output.

    Returns
    -------
    seed_img : cupy.ndarray of int or bool
        The desired image.

    Examples
    --------
    >>> regular_seeds((5, 5), 4)
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 2, 0],
           [0, 0, 0, 0, 0],
           [0, 3, 0, 4, 0],
           [0, 0, 0, 0, 0]], dtype=int32)
    """
    ar_shape = tuple(int(s) for s in ar_shape)
    ndim = len(ar_shape)
    grid = regular_grid(ar_shape, n_points)

    # Extract start and step from each slice
    starts = []
    steps = []
    for s, dim_size in zip(grid, ar_shape):
        start = s.start if s.start is not None else 0
        step = s.step if s.step is not None else 1
        starts.append(int(start))
        steps.append(int(step))

    size = math.prod(ar_shape)

    if ndim in (2, 3):
        kernel = _get_regular_seeds_kernel(ndim)
        seed_img = cp.empty(size, dtype=cp.int32)
        threads = 256
        blocks = _ceildiv(size, threads)

        if ndim == 2:
            n_per_row = _ceildiv(ar_shape[1] - starts[1], steps[1])
            kernel(
                (blocks,),
                (threads,),
                (
                    seed_img,
                    ar_shape[0],
                    ar_shape[1],
                    starts[0],
                    steps[0],
                    starts[1],
                    steps[1],
                    n_per_row,
                ),
            )
        else:  # ndim == 3
            n_per_row = _ceildiv(ar_shape[2] - starts[2], steps[2])
            n_per_plane = (
                _ceildiv(ar_shape[1] - starts[1], steps[1]) * n_per_row
            )
            kernel(
                (blocks,),
                (threads,),
                (
                    seed_img,
                    ar_shape[0],
                    ar_shape[1],
                    ar_shape[2],
                    starts[0],
                    steps[0],
                    starts[1],
                    steps[1],
                    starts[2],
                    steps[2],
                    n_per_plane,
                    n_per_row,
                ),
            )

        seed_img = seed_img.reshape(ar_shape)
        if dtype != cp.int32:
            seed_img = seed_img.astype(dtype)
        return seed_img
    else:
        # Fallback for other dimensionalities
        seed_img = cp.zeros(ar_shape, dtype=dtype)
        seed_img[grid] = 1 + cp.reshape(
            cp.arange(seed_img[grid].size), seed_img[grid].shape
        )
        return seed_img
