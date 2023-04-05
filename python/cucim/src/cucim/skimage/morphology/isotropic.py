"""
Binary morphological operations
"""
import cupy as cp

from cucim.core.operations.morphology import distance_transform_edt


def _check_output(out, shape):
    """Check shape and dtype of output array.

    Parameters
    ----------
    out : cp.ndarray or None
        The array to check
    shape : tuple of int
        The expected shape

    Returns
    -------
    out : cp.ndarray
        The original array (or boolean view of a uint8 array).
    """
    if out is None:
        return None
    if out.shape != shape:
        raise ValueError("out.shape must match image.shape")
    if not out.flags.c_contiguous:
        raise ValueError("out array must have C-contiguous memory layout")
    if out.dtype == bool:
        return out
    elif out.dtype == cp.uint8:
        # view uint8 as bool
        return out.view(bool)
    else:
        raise ValueError("provided out array should have boolean type")


def isotropic_erosion(image, radius, out=None, spacing=None):
    """Return binary morphological erosion of an image.

    This function returns the same result as
    :func:`skimage.morphology.binary_erosion` but performs faster for large
    circular structuring elements. This works by applying a threshold to the
    exact Euclidean distance map of the image [1]_, [2]_. The implementation is
    based on:
    func:`cucim.core.operations.morphology.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be eroded.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension
        (number of axes). If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    eroded : ndarray of bool
        The result of the morphological erosion taking values in
        ``[False, True]``.

    Notes
    -----
    Empirically, on an RTX A6000 GPU, it was observed that
    ``isotropic_erosion`` is faster than ``binary_erosion`` with
    ``decomposition=None`` at radius 12 in 2D and radius 3 in 3D. It becomes
    faster than ``binary_erosion`` with ``decomposition="sequence"`` at radius
    14 in 2D and radius 5 in 3D. In practice, the exact point at which these
    isotropic functions become faster than their binary counterparts will also
    be dependent on image shape and content.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """
    out = _check_output(out, image.shape)
    dist = distance_transform_edt(image, sampling=spacing)
    if out is not None:
        cp.greater(dist, radius, out=out)
    else:
        out = cp.greater(dist, radius)
    return out


def isotropic_dilation(image, radius, out=None, spacing=None):
    """Return binary morphological dilation of an image.

    This function returns the same result as
    :func:`skimage.morphology.binary_dilation` but performs faster for large
    circular structuring elements. This works by applying a threshold to the
    exact Euclidean distance map of the inverted image [1]_, [2]_. The
    implementation is based on:
    func:`cucim.core.operations.morphology.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be dilated.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension
        (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    dilated : ndarray of bool
        The result of the morphological dilation with values in
        ``[False, True]``.

    Notes
    -----
    Empirically, on an RTX A6000 GPU, it was observed that
    ``isotropic_dilation`` is faster than ``binary_dilation`` with
    ``decomposition=None`` at radius 12 in 2D and radius 3 in 3D. It becomes
    faster than ``binary_dilation`` with ``decomposition="sequence"`` at radius
    14 in 2D and radius 5 in 3D. In practice, the exact point at which these
    isotropic functions become faster than their binary counterparts will also
    be dependent on image shape and content.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """
    out = _check_output(out, image.shape)
    dist = distance_transform_edt(cp.logical_not(image), sampling=spacing)
    if out is not None:
        cp.less_equal(dist, radius, out=out)
    else:
        out = cp.less_equal(dist, radius)
    return out


def isotropic_opening(image, radius, out=None, spacing=None):
    """Return binary morphological opening of an image.

    This function returns the same result as
    :func:`skimage.morphology.binary_opening` but performs faster for large
    circular structuring elements. This works by thresholding the exact
    Euclidean distance map [1]_, [2]_. The implementation is based on:
    func:`cucim.core.operations.morphology.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius with which the regions should be opened.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension
        (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    opened : ndarray of bool
        The result of the morphological opening.

    Notes
    -----
    Empirically, on an RTX A6000 GPU, it was observed that
    ``isotropic_opening`` is faster than ``binary_opening`` with
    ``decomposition=None`` at radius 12 in 2D and radius 3 in 3D. It becomes
    faster than ``binary_erosion`` with ``decomposition="sequence"`` at radius
    14 in 2D and radius 5 in 3D. In practice, the exact point at which these
    isotropic functions become faster than their binary counterparts will also
    be dependent on image shape and content.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """
    out = _check_output(out, image.shape)
    eroded = isotropic_erosion(image, radius, spacing=spacing)
    return isotropic_dilation(eroded, radius, out=out, spacing=spacing)


def isotropic_closing(image, radius, out=None, spacing=None):
    """Return binary morphological closing of an image.

    This function returns the same result as binary
    :func:`skimage.morphology.binary_closing` but performs faster for large
    circular structuring elements. This works by thresholding the exact
    Euclidean distance map [1]_, [2]_. The implementation is based on:
    func:`cucim.core.operations.morphology.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius with which the regions should be closed.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension
        (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    closed : ndarray of bool
        The result of the morphological closing.

    Notes
    -----
    Empirically, on an RTX A6000 GPU, it was observed that
    ``isotropic_closing`` is faster than ``binary_closing`` with
    ``decomposition=None`` at radius 12 in 2D and radius 3 in 3D. It becomes
    faster than ``binary_erosion`` with ``decomposition="sequence"`` at radius
    14 in 2D and radius 5 in 3D. In practice, the exact point at which these
    isotropic functions become faster than their binary counterparts will also
    be dependent on image shape and content.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """
    out = _check_output(out, image.shape)
    dilated = isotropic_dilation(image, radius, spacing=spacing)
    return isotropic_erosion(dilated, radius, out=out, spacing=spacing)
