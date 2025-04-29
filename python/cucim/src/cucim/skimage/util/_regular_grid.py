import math


def argsort(shape):
    """argsort for a sequence without the overhead of NumPy array creation"""
    return sorted(range(len(shape)), key=lambda i: shape[i])


def regular_grid(ar_shape, n_points):
    """Find `n_points` regularly spaced along `ar_shape`.

    The returned points (as slices) should be as close to cubically-spaced as
    possible. Essentially, the points are spaced by the Nth root of the input
    array size, where N is the number of dimensions. However, if an array
    dimension cannot fit a full step size, it is "discarded", and the
    computation is done for only the remaining dimensions.

    Parameters
    ----------
    ar_shape : array-like of ints
        The shape of the space embedding the grid. ``len(ar_shape)`` is the
        number of dimensions.
    n_points : int
        The (approximate) number of points to embed in the space.

    Returns
    -------
    slices : tuple of slice objects
        A slice along each dimension of `ar_shape`, such that the intersection
        of all the slices give the coordinates of regularly spaced points.

        .. versionchanged:: 0.14.1
            In scikit-image 0.14.1 and 0.15, the return type was changed from a
            list to a tuple to ensure `compatibility with Numpy 1.15`_ and
            higher. If your code requires the returned result to be a list, you
            may convert the output of this function to a list with:

            >>> result = list(regular_grid(ar_shape=(3, 20, 40), n_points=8))

            .. _compatibility with NumPy 1.15: https://github.com/numpy/numpy/blob/master/doc/release/1.15.0-notes.rst#deprecations

    Examples
    --------
    >>> ar = np.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 8)
    >>> g
    (slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    8.0
    >>> ar = np.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 32)
    >>> g
    (slice(2, None, 5), slice(2, None, 5))
    >>> ar[g] = 1
    >>> ar.sum()
    32.0
    >>> ar = np.zeros((3, 20, 40))
    >>> g = regular_grid(ar.shape, 8)
    >>> g
    (slice(1, None, 3), slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    8.0
    """
    # ar_shape = np.asanyarray(ar_shape)
    ndim = len(ar_shape)
    sort_dim_idx = argsort(ar_shape)
    unsort_dim_idxs = argsort(sort_dim_idx)

    sorted_dims = tuple(ar_shape[s] for s in sort_dim_idx)
    space_size = float(math.prod(ar_shape))
    if space_size <= n_points:
        return (slice(None),) * ndim
    step_size = (space_size / n_points) ** (1.0 / ndim)
    if any(s < step_size for s in sorted_dims):
        step_sizes = [
            step_size,
        ] * ndim
        for dim in range(ndim):
            step_sizes[dim] = sorted_dims[dim]
            space_size = float(math.prod(sorted_dims[dim + 1 :]))
            new_sz = (space_size / n_points) ** (1.0 / (ndim - dim - 1))
            for d2 in range(dim + 1, ndim):
                step_sizes[d2] = new_sz
            if not any(s < sz for s, sz in zip(sorted_dims, step_sizes)):
                break
        starts = tuple(int(s // 2) for s in step_sizes)
        step_sizes = tuple(round(s) for s in step_sizes)
    else:
        starts = (int(step_size) // 2,) * ndim
        step_sizes = (round(step_size),) * ndim
    slices = [
        slice(start, None, step) for start, step in zip(starts, step_sizes)
    ]
    slices = tuple(slices[i] for i in unsort_dim_idxs)
    return slices
