import numpy as np

from ._pba_2d import _pba_2d
from ._pba_3d import _pba_3d

# TODO: support sampling distances
#       support the distances and indices output arguments
#       support chamfer, chessboard and l1/manhattan distances too?


def distance_transform_edt(image, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None,
                           *, block_params=None, float64_distances=False):
    """Exact Euclidean distance transform.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its shortest distance to
    the background (any zero-valued element).

    In addition to the distance transform, the feature transform can be
    calculated. In this case the index of the closest background element to
    each foreground element is returned in a separate array.

    Parameters
    ----------
    image : array_like
        Input data to transform. Can be any type but will be converted into
        binary: 1 wherever image equates to True, 0 elsewhere.
    sampling : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the image rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to calculate the distance transform.
    return_indices : bool, optional
        Whether to calculate the feature transform.
    distances : float32 cupy.ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it. `return_distances` must be True. It must be the same
        shape as `image`.
    indices : int32 cupy.ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it. `return_indicies` must be True. Its shape must be
        `(image.ndim,) + image.shape`.

    Other Parameters
    ----------------
    block_params : 3-tuple of int
        The m1, m2, m3 algorithm parameters as described in [2]_. If None,
        suitable defaults will be chosen. Note: This parameter is specific to
        cuCIM and does not exist in SciPy.
    float64_distances : bool, optional
        If True, use double precision in the distance computation (to match
        SciPy behavior). Otherwise, single precision will be used for
        efficiency. Note: This parameter is specific to cuCIM and does not
        exist in SciPy.

    Returns
    -------
    distances : float64 ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True and `distances` is not supplied. It will
        have the same shape as `image`.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an image-shaped array for each
        dimension of the image. See example below. Returned only when
        `return_indices` is True and `indices` is not supplied.

    Notes
    -----
    The Euclidean distance transform gives values of the Euclidean distance::

                    n
      y_i = sqrt(sum (x[i]-b[i])**2)
                    i

    where b[i] is the background point (value 0) with the smallest Euclidean
    distance to input points x[i], and n is the number of dimensions.

    Note that the `indices` output may differ from the one given by
    `scipy.ndimage.distance_transform_edt` in the case of input pixels that are
    equidistant from multiple background points.

    The parallel banding algorithm implemented here was originally described in
    [1]_. The kernels used here correspond to the revised PBA+ implementation
    that is described on the author's website [2]_. The source code of the
    author's PBA+ implementation is available at [3]_.

    References
    ----------
    ..[1] Thanh-Tung Cao, Ke Tang, Anis Mohamed, and Tiow-Seng Tan. 2010.
        Parallel Banding Algorithm to compute exact distance transform with the
        GPU. In Proceedings of the 2010 ACM SIGGRAPH symposium on Interactive
        3D Graphics and Games (I3D ’10). Association for Computing Machinery,
        New York, NY, USA, 83–90.
        DOI:https://doi.org/10.1145/1730804.1730818
    .. [2] https://www.comp.nus.edu.sg/~tants/pba.html
    .. [3] https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.core.operations import morphology
    >>> a = cp.array(([0,1,1,1,1],
    ...               [0,0,1,1,1],
    ...               [0,1,1,1,1],
    ...               [0,1,1,1,0],
    ...               [0,1,1,0,0]))
    >>> morphology.distance_transform_edt(a)
    array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
           [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
           [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    With a sampling of 2 units along x, 1 along y:

    >>> morphology.distance_transform_edt(a, sampling=[2,1])
    array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
           [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
           [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    Asking for indices as well:

    >>> edt, inds = morphology.distance_transform_edt(a, return_indices=True)
    >>> inds
    array([[[0, 0, 1, 1, 3],
            [1, 1, 1, 1, 3],
            [2, 2, 1, 3, 3],
            [3, 3, 4, 4, 3],
            [4, 4, 4, 4, 4]],
           [[0, 0, 1, 1, 4],
            [0, 1, 1, 1, 4],
            [0, 0, 1, 4, 4],
            [0, 0, 3, 3, 4],
            [0, 0, 3, 3, 4]]])

    """
    if distances is not None:
        raise NotImplementedError(
            "preallocated distances image is not supported"
        )
    if indices is not None:
        raise NotImplementedError(
            "preallocated indices image is not supported"
        )
    scalar_sampling = None
    if sampling is not None:
        sampling = np.unique(np.atleast_1d(sampling))
        if len(sampling) == 1:
            scalar_sampling = float(sampling)
            sampling = None
        else:
            raise NotImplementedError(
                "non-uniform values in sampling is not currently supported"
            )

    if image.ndim == 3:
        pba_func = _pba_3d
    elif image.ndim == 2:
        pba_func = _pba_2d
    else:
        raise NotImplementedError(
            "Only 2D and 3D distance transforms are supported.")

    vals = pba_func(
        image,
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
        block_params=block_params
    )

    if return_distances and scalar_sampling is not None:
        vals = (vals[0] * scalar_sampling,) + vals[1:]

    if len(vals) == 1:
        vals = vals[0]

    return vals
