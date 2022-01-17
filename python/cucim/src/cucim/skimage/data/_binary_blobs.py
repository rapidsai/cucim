import cupy as cp


def binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2,
                 volume_fraction=0.5, seed=None):
    """
    Generate synthetic binary image with several rounded blob-like objects.

    Parameters
    ----------
    length : int, optional
        Linear size of output image.
    blob_size_fraction : float, optional
        Typical linear size of blob, as a fraction of ``length``, should be
        smaller than 1.
    n_dim : int, optional
        Number of dimensions of output image.
    volume_fraction : float, default 0.5
        Fraction of image pixels covered by the blobs (where the output is 1).
        Should be in [0, 1].
    seed : {None, int, `cupy.random.Generator`}, optional
        If `seed` is None the `cupy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Returns
    -------
    blobs : ndarray of bools
        Output binary image

    Notes
    -----
    Warning: CuPy does not give identical randomly generated numbers as NumPy,
    so using a specific seed here will not give an identical pattern to the
    scikit-image implementation.

    The behavior for a given random seed may also change across CuPy major
    versions.
    See: https://docs.cupy.dev/en/stable/reference/random.html

    Examples
    --------
    >>> from cucim.skimage import data
    >>> # tiny size (5, 5)
    >>> blobs = data.binary_blobs(length=5, blob_size_fraction=0.2, seed=1)
    >>> # larger size
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.1)
    >>> # Finer structures
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.05)
    >>> # Blobs cover a smaller volume fraction of the image
    >>> blobs = data.binary_blobs(length=256, volume_fraction=0.3)
    """
    # filters is quite an expensive import since it imports all of scipy.signal
    # We lazy import here
    from .._shared.filters import gaussian

    rs = cp.random.default_rng(seed)
    shape = tuple([length] * n_dim)
    mask = cp.zeros(shape)
    n_pts = max(int(1. / blob_size_fraction) ** n_dim, 1)
    points = (length * rs.random((n_dim, n_pts))).astype(int)
    mask[tuple(indices for indices in points)] = 1
    mask = gaussian(mask, sigma=0.25 * length * blob_size_fraction)
    threshold = cp.percentile(mask, 100 * (1 - volume_fraction))
    return cp.logical_not(mask < threshold)
