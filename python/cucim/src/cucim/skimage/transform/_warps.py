import math

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi

from .._shared.utils import (_to_ndimage_mode, _validate_interpolation_order,
                             channel_as_last_axis, convert_to_float,
                             safe_as_int, warn)
from .._vendored import pad
from ..measure import block_reduce
from ._geometric import (AffineTransform, ProjectiveTransform,
                         SimilarityTransform)

HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform,
)


def _preprocess_resize_output_shape(image, output_shape):
    """Validate resize output shape according to input image.

    Parameters
    ----------
    image: ndarray
        Image to be resized.
    output_shape: tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved.

    Returns
    -------
    image: ndarray
        The input image, but with additional singleton dimensions appended in
        the case where ``len(output_shape) > input.ndim``.
    output_shape: tuple
        The output image converted to tuple.

    Raises
    ------
    ValueError:
        If output_shape length is smaller than the image number of
        dimensions

    Notes
    -----
    The input image is reshaped if its number of dimensions is not
    equal to output_shape_length.

    """
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape += (1, ) * (output_ndim - image.ndim)
        image = cp.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1], )
    elif output_ndim < image.ndim:
        raise ValueError("output_shape length cannot be smaller than the "
                         "image number of dimensions")

    return image, output_shape


def resize(image, output_shape, order=None, mode='reflect', cval=0, clip=None,
           preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None):
    """Resize image to match a certain size.

    Performs interpolation to up-size or down-size N-dimensional images. Note
    that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For downsampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.

    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels a
        n-dimensional interpolation is applied.

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        If `order` > 1, this will be enabled by default, since higher order
        interpolation may produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to downsampling. It is crucial to filter when downsampling
        the image to avoid aliasing artifacts. If not specified, it is set to
        True when downsampling an image whose data type is not bool.
        It is also set to False when using nearest neighbor interpolation
        (``order`` == 0) with integer input data type.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering used when anti-aliasing.
        By default, this value is chosen as (s - 1) / 2 where s is the
        downsampling factor, where s > 1. For the up-size case, s < 1, no
        anti-aliasing is performed prior to rescaling.

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.transform import resize
    >>> image = cp.array(data.camera())
    >>> resize(image, (100, 100)).shape
    (100, 100)

    """

    image, output_shape = _preprocess_resize_output_shape(image, output_shape)
    input_shape = image.shape
    input_type = image.dtype

    if input_type == cp.float16:
        image = image.astype(cp.float32, copy=False)

    if anti_aliasing is None:
        anti_aliasing = (
            not input_type == bool and
            not (cp.issubdtype(input_type, cp.integer) and order == 0) and
            any(x < y for x, y in zip(output_shape, input_shape)))

    if input_type == bool and anti_aliasing:
        raise ValueError("anti_aliasing must be False for boolean images")

    factors = tuple(si / so for si, so in zip(input_shape, output_shape))
    order = _validate_interpolation_order(input_type, order)
    if order > 0:
        image = convert_to_float(image, preserve_range)

    if clip is None:
        clip = True if order > 1 else False

    # Translate modes used by np.pad to those used by scipy.ndimage
    ndi_mode = _to_ndimage_mode(mode)
    if anti_aliasing:
        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = tuple([max(0, (f - 1) / 2) for f in factors])
        else:
            if np.isscalar(anti_aliasing_sigma):
                anti_aliasing_sigma = (anti_aliasing_sigma,) * len(factors)
            elif len(anti_aliasing_sigma) != len(factors):
                raise ValueError("invalid anti_aliasing_sigma length")
            if any(sigma < 0 for sigma in anti_aliasing_sigma):
                raise ValueError("Anti-aliasing standard deviation must be "
                                 "greater than or equal to zero")
            elif any(((sigma > 0) & (factor <= 1))
                     for factor, sigma in zip(factors, anti_aliasing_sigma)):
                warn("Anti-aliasing standard deviation greater than zero but "
                     "not down-sampling along all axes")

        # TODO: CuPy itself should do this grid-constant->constant conversion
        #       make upstream PR for SciPy-compatible behavior
        _ndi_mode = {'grid-constant': 'constant', 'grid-wrap':'wrap'}.get(ndi_mode, ndi_mode)  # noqa
        # keep ndi.gaussian_filter rather than cucim.skimage.filters.gaussian
        # to avoid undesired dtype coercion
        filtered = ndi.gaussian_filter(image, anti_aliasing_sigma, cval=cval,
                                       mode=_ndi_mode)
    else:
        filtered = image

    zoom_factors = [1 / f for f in factors]
    out = ndi.zoom(filtered, zoom_factors, order=order, mode=ndi_mode,
                   cval=cval, grid_mode=True)
    _clip_warp_output(image, out, mode, cval, order, clip)
    return out


@channel_as_last_axis()
def rescale(image, scale, order=None, mode='reflect', cval=0, clip=None,
            preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None,
            *, channel_axis=None):
    """Scale image by a certain factor.

    Performs interpolation to up-scale or down-scale N-dimensional images.
    Note that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For down-sampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.

    Parameters
    ----------
    image : ndarray
        Input image.
    scale : {float, tuple of floats}
        Scale factors. Separate scale factors can be defined as
        `(rows, cols[, ...][, dim])`.

    Returns
    -------
    scaled : ndarray
        Scaled version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        If `order` > 1, this will be enabled by default, since higher order
        interpolation may produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to down-scaling. It is crucial to filter when down-sampling
        the image to avoid aliasing artifacts. If input image data
        type is bool, no anti-aliasing is applied.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
        By default, this value is chosen as (s - 1) / 2 where s is the
        down-scaling factor.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 22.02.00
           ``channel_axis`` was added in 22.02.00.

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.transform import rescale
    >>> image = cp.array(data.camera())
    >>> rescale(image, 0.1).shape
    (51, 51)
    >>> rescale(image, 0.5).shape
    (256, 256)

    """
    scale = np.atleast_1d(scale)
    multichannel = channel_axis is not None
    if len(scale) > 1:
        if ((not multichannel and len(scale) != image.ndim) or
                (multichannel and len(scale) != image.ndim - 1)):
            raise ValueError("Supply a single scale, or one value per spatial "
                             "axis")
        if multichannel:
            scale = np.concatenate((scale, [1]))
    orig_shape = np.asarray(image.shape)
    output_shape = np.maximum(np.round(scale * orig_shape), 1)
    if multichannel:  # don't scale channel dimension
        output_shape[-1] = orig_shape[-1]

    return resize(image, output_shape, order=order, mode=mode, cval=cval,
                  clip=clip, preserve_range=preserve_range,
                  anti_aliasing=anti_aliasing,
                  anti_aliasing_sigma=anti_aliasing_sigma)


def _ndimage_affine(image, matrix, output_shape, order, mode, cval, clip,
                    preserve_range):
    """Thin wrapper around scipy.ndimage.affine_transform

    Validates input and handles clipping of output in the same way as ``warp``.
    """
    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions",
                         image.shape)

    order = _validate_interpolation_order(image.dtype, order)

    if image.dtype.kind == "c":
        if not preserve_range:
            raise NotImplementedError("TODO")
    elif order > 0:
        image = convert_to_float(image, preserve_range)

    input_shape = image.shape

    if output_shape is None:
        output_shape = input_shape
    else:
        output_shape = safe_as_int(output_shape)

    # Pre-filtering not necessary for order 0, 1 interpolation
    prefilter = order > 1

    ndi_mode = _to_ndimage_mode(mode)
    warped = ndi.affine_transform(image, matrix, prefilter=prefilter,
                                  mode=ndi_mode, order=order, cval=cval,
                                  output_shape=tuple(output_shape))

    _clip_warp_output(image, warped, mode, cval, order, clip)
    return warped


def _ndimage_rotate(image, angle, resize, order, mode, cval, clip,
                    preserve_range):
    """Thin wrapper around scipy.ndimage.rotate

    Validates input and handles clipping of output in the same way as ``warp``.
    """
    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions",
                         image.shape)

    order = _validate_interpolation_order(image.dtype, order)

    if image.dtype.kind == "c":
        if not preserve_range:
            raise NotImplementedError("TODO")
    elif order > 0:
        image = convert_to_float(image, preserve_range)

    # Pre-filtering not necessary for order 0, 1 interpolation
    prefilter = order > 1

    ndi_mode = _to_ndimage_mode(mode)
    warped = ndi.rotate(image, angle, reshape=resize, prefilter=prefilter,
                        mode=ndi_mode, order=order, cval=cval)
    _clip_warp_output(image, warped, mode, cval, order, clip)
    return warped


def rotate(image, angle, resize=False, center=None, order=None,
           mode='constant', cval=0, clip=True, preserve_range=False):
    """Rotate image by a certain angle around its center.

    Parameters
    ----------
    image : ndarray
        Input image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    resize : bool, optional
        Determine whether the shape of the output image will be automatically
        calculated, so the complete rotated image exactly fits. Default is
        False.
    center : iterable of length 2
        The rotation center. If ``center=None``, the image is rotated around
        its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
        note that this parameter is (cols, rows), contrary to normal skimage
        ordering.

    Returns
    -------
    rotated : ndarray
        Rotated version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        If `order` > 1, this will be enabled by default, since higher order
        interpolation may produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    If ``image.ndim > 2``, the rotation occurs for the first two dimensions of
    the array. Unlike the scikit-image implementation, more than one additional
    axis may be present on the array.

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.transform import rotate
    >>> image = cp.array(data.camera())
    >>> rotate(image, 2).shape
    (512, 512)
    >>> rotate(image, 2, resize=True).shape
    (530, 530)
    >>> rotate(image, 90, resize=True).shape
    (512, 512)

    """
    rows, cols = image.shape[0], image.shape[1]
    if image.dtype == cp.float16:
        image = image.astype(cp.float32)
    img_center = np.array((cols, rows)) / 2. - 0.5
    if center is None:
        center = img_center
        centered = True
    else:
        center = np.asarray(center)
        centered = np.array_equal(center, img_center)

    if centered and not resize:
        # can use cupyx.scipy.ndimage.rotate
        return _ndimage_rotate(
            image, angle, resize, order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range
        )

    # rotation around center
    tform1 = SimilarityTransform(translation=center, xp=np)
    tform2 = SimilarityTransform(rotation=np.deg2rad(angle), xp=np)
    tform3 = SimilarityTransform(translation=-center, xp=np)
    tform = tform3 + tform2 + tform1

    output_shape = None
    if resize:
        # determine shape of output image
        # fmt: off
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        # fmt: on
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = (round(out_rows), round(out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation, xp=np)
        tform = tform4 + tform

    # Make sure the transform is exactly affine, to ensure fast warping.
    tform.params[2] = (0, 0, 1)

    # swap axes in the matrix to match cupyx.scipy.ndimage.affine_transform
    tform.params[:2, :2] = tform.params[:2, :2].T
    tform.params[:2, 2] = tform.params[1::-1, 2]

    if image.ndim == 2:
        affine_params = tform.params
    elif image.ndim > 2:
        # note: only the first two dimensions are the ones being rotated
        #       embed 2D affine into larger identity matrix.
        affine_params = np.eye(image.ndim + 1)
        affine_params[:3, :3] = tform.params
        # keep original shape on the excess dimensions
        output_shape = output_shape + image.shape[2:]

    # transfer the coordinate transform to the GPU
    affine_params = cp.asarray(affine_params)

    return _ndimage_affine(
        image, affine_params, output_shape=output_shape, order=order,
        mode=mode, cval=cval, clip=clip, preserve_range=preserve_range
    )


def downscale_local_mean(image, factors, cval=0, clip=True):
    """Down-sample N-dimensional image by local averaging.

    The image is padded with `cval` if it is not perfectly divisible by the
    integer factors.

    In contrast to interpolation in `skimage.transform.resize` and
    `skimage.transform.rescale` this function calculates the local mean of
    elements in each block of size `factors` in the input image.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    factors : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        integer factors.
    clip : bool, optional
        Unused, but kept here for API consistency with the other transforms
        in this module. (The local mean will never fall outside the range
        of values in the input image, assuming the provided `cval` also
        falls within that range.)

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
        For integer inputs, the output dtype will be ``float64``.
        See :func:`numpy.mean` for details.

    Examples
    --------
    >>> import cupy as cp
    >>> a = cp.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> downscale_local_mean(a, (2, 3))
    array([[3.5, 4. ],
           [5.5, 4.5]])

    """
    return block_reduce(image, factors, cp.mean, cval)


def _swirl_mapping(xy, center, rotation, strength, radius):
    x, y = xy.T
    x0, y0 = center
    xdiff = x - x0
    ydiff = y - y0
    rho = cp.sqrt(xdiff * xdiff + ydiff * ydiff)

    # Ensure that the transformation decays to approximately 1/1000-th
    # within the specified radius.
    radius = radius / 5 * math.log(2)

    theta = rotation + strength * cp.exp(-rho / radius)
    theta += cp.arctan2(ydiff, xdiff)

    xy[..., 0] = x0 + rho * cp.cos(theta)
    xy[..., 1] = y0 + rho * cp.sin(theta)

    return xy


def swirl(image, center=None, strength=1, radius=100, rotation=0,
          output_shape=None, order=None, mode='reflect', cval=0, clip=None,
          preserve_range=False):
    """Perform a swirl transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    center : (column, row) tuple or (2,) ndarray, optional
        Center coordinate of transformation.
    strength : float, optional
        The amount of swirling applied.
    radius : float, optional
        The extent of the swirl in pixels.  The effect dies out
        rapidly beyond `radius`.
    rotation : float, optional
        Additional rotation applied to the image.

    Returns
    -------
    swirled : ndarray
        Swirled version of the input.

    Other parameters
    ----------------
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode, with 'constant' used as the default. Modes match
        the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        If `order` > 1, this will be enabled by default, since higher order
        interpolation may produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    """
    if center is None:
        center = np.array(image.shape)[:2][::-1] / 2

    warp_args = {'center': center,
                 'rotation': rotation,
                 'strength': strength,
                 'radius': radius}

    return warp(image, _swirl_mapping, map_args=warp_args,
                output_shape=output_shape, order=order, mode=mode, cval=cval,
                clip=clip, preserve_range=preserve_range)


def _stackcopy(a, b):
    """Copy b into each color layer of a, such that::

      a[:,:,0] = a[:,:,1] = ... = b

    Parameters
    ----------
    a : (M, N) or (M, N, P) ndarray
        Target array.
    b : (M, N)
        Source array.

    Notes
    -----
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.

    """
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def warp_coords(coord_map, shape, dtype=cp.float64):
    """Build the source coordinates for the output of a 2-D image warp.

    Parameters
    ----------
    coord_map : callable like GeometricTransform.inverse
        Return input coordinates for given output coordinates.
        Coordinates are in the shape (P, 2), where P is the number
        of coordinates and each element is a ``(row, col)`` pair.
    shape : tuple
        Shape of output image ``(rows, cols[, bands])``.
    dtype : np.dtype or string
        dtype for return value (sane choices: float32 or float64).

    Returns
    -------
    coords : (ndim, rows, cols[, bands]) array of dtype `dtype`
            Coordinates for `scipy.ndimage.map_coordinates`, that will yield
            an image of shape (orows, ocols, bands) by drawing from source
            points according to the `coord_transform_fn`.

    Notes
    -----

    This is a lower-level routine that produces the source coordinates for 2-D
    images used by `warp()`.

    It is provided separately from `warp` to give additional flexibility to
    users who would like, for example, to re-use a particular coordinate
    mapping, to use specific dtypes at various points along the the
    image-warping process, or to implement different post-processing logic
    than `warp` performs after the call to `ndi.map_coordinates`.


    Examples
    --------
    Produce a coordinate map that shifts an image up and to the right:

    >>> import cupy as cp
    >>> from cucim.skimage.transform import warp_coords
    >>> from skimage import data
    >>> from cupyx.scipy.ndimage import map_coordinates
    >>>
    >>> def shift_up10_left20(xy):
    ...     return xy - cp.array([-20, 10])[None, :]
    >>>
    >>> image = cp.array(data.astronaut().astype(cp.float32))
    >>> coords = warp_coords(shift_up10_left20, image.shape)
    >>> warped_image = map_coordinates(image, coords)

    """
    shape = safe_as_int(shape)
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = cp.empty(coords_shape, dtype=dtype)

    # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
    tf_coords = cp.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    # Map each (row, col) pair to the source image according to
    # the user-provided mapping
    tf_coords = coord_map(tf_coords)

    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # Place the x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    if len(shape) == 3:
        coords[2, ...] = cp.arange(shape[2], dtype=coords.dtype)

    return coords


def _clip_warp_output(input_image, output_image, mode, cval, order, clip):
    """Clip output image to range of values of input image.

    Note that this function modifies the values of `output_image` in-place
    and it is only modified if ``clip=True`` (or clip is ``None`` and
    `order` > 1).

    Parameters
    ----------
    input_image : ndarray
        Input image.
    output_image : ndarray
        Output image, which is modified in-place.

    Other parameters
    ----------------
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : int
        The interpolation order used. If `clip` is ``None`` and `order` <= 1,
        no clipping will be applied.
    clip : bool or None
        Whether to clip the output to the range of values of the input image.
        If `order` > 1, this will be enabled by default, since higher order
        interpolation may produce values outside the given input range.

    """
    if (clip is None and order <= 1) or not clip:
        # no clipping by default for order = 0 or 1
        return
    if isinstance(input_image, tuple) and len(input_image) == 2:
        min_val, max_val = input_image
        # copy device scalars to host if necessary
        if isinstance(min_val, cp.ndarray):
            min_val = min_val.item()
        if isinstance(max_val, cp.ndarray):
            max_val = max_val.item()
    else:
        min_val = input_image.min().item()
        if np.isnan(min_val):
            # NaNs detected, use NaN-safe min/max
            min_func = cp.nanmin
            max_func = cp.nanmax
            min_val = min_func(input_image).item()
        else:
            min_func = cp.min
            max_func = cp.max
        max_val = max_func(input_image).item()

    # Check if cval has been used such that it expands the effective input
    # range
    preserve_cval = (
        mode == 'constant'
        and not min_val <= cval <= max_val
        and min_func(output_image) <= cval <= max_func(output_image)
    )

    # expand min/max range to account for cval
    if preserve_cval:
        # cast cval to the same dtype as the input image
        cval = input_image.dtype.type(cval)
        min_val = min(min_val, cval)
        max_val = max(max_val, cval)

    cp.clip(output_image, min_val, max_val, out=output_image)


def warp(image, inverse_map, map_args=None, output_shape=None, order=None,
         mode='constant', cval=0., clip=None, preserve_range=False):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    inverse_map : transformation object, callable ``cr = f(cr, **kwargs)``, or ndarray
        Inverse coordinate map, which transforms coordinates in the output
        images into their corresponding coordinates in the input image.

        There are a number of different options to define this map, depending
        on the dimensionality of the input image. A 2-D image can have 2
        dimensions for gray-scale images, or 3 dimensions with color
        information.

         - For 2-D images, you can directly pass a transformation object,
           e.g. `skimage.transform.SimilarityTransform`, or its inverse.
         - For 2-D images, you can pass a ``(3, 3)`` homogeneous
           transformation matrix, e.g.
           `skimage.transform.SimilarityTransform.params`.
         - For 2-D images, a function that transforms a ``(M, 2)`` array of
           ``(col, row)`` coordinates in the output image to their
           corresponding coordinates in the input image. Extra parameters to
           the function can be specified through `map_args`.
         - For N-D images, you can directly pass an array of coordinates.
           The first dimension specifies the coordinates in the input image,
           while the subsequent dimensions determine the position in the
           output image. E.g. in case of 2-D images, you need to pass an array
           of shape ``(2, rows, cols)``, where `rows` and `cols` determine the
           shape of the output image, and the first dimension contains the
           ``(row, col)`` coordinate in the input image.
           See `scipy.ndimage.map_coordinates` for further documentation.

        Note, that a ``(3, 3)`` matrix is interpreted as a homogeneous
        transformation matrix, so you cannot interpolate values from a 3-D
        input, if the output is of shape ``(3,)``.

        See example section for usage.
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.  Note that, even for multi-band images, only rows
        and columns need to be specified.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic

         Default is 0 if image.dtype is bool and 1 otherwise.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        If `order` > 1, this will be enabled by default, since higher order
        interpolation may produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    warped : double ndarray
        The warped input image.

    Notes
    -----
    - The input image is converted to a `double` image.
    - In case of a `SimilarityTransform`, `AffineTransform` and
      `ProjectiveTransform` and `order` in [0, 3] this function uses the
      underlying transformation matrix to warp the image with a much faster
      routine.

    Examples
    --------
    >>> from cucim.skimage.transform import warp
    >>> from skimage import data
    >>> image = cp.array(data.camera())

    The following image warps are all equal but differ substantially in
    execution time. The image is shifted to the bottom.

    Use a geometric transform to warp an image (fast):

    >>> from cucim.skimage.transform import SimilarityTransform
    >>> tform = SimilarityTransform(translation=(0, -10))
    >>> warped = warp(image, tform)

    Use a callable (slow):

    >>> def shift_down(xy):
    ...     xy[:, 1] -= 10
    ...     return xy
    >>> warped = warp(image, shift_down)

    Use a transformation matrix to warp an image (fast):

    >>> import cupy as cp
    >>> matrix = cp.asarray([[1, 0, 0], [0, 1, -10], [0, 0, 1]])
    >>> warped = warp(image, matrix)
    >>> from cucim.skimage.transform import ProjectiveTransform, warp
    >>> warped = warp(image, ProjectiveTransform(matrix=matrix))

    You can also use the inverse of a geometric transformation (fast):

    >>> warped = warp(image, tform.inverse)

    For N-D images you can pass a coordinate array, that specifies the
    coordinates in the input image for every element in the output image. E.g.
    if you want to rescale a 3-D cube, you can do:

    >>> cube_shape = (30, 30, 30)
    >>> cube = cp.random.rand(*cube_shape)

    Setup the coordinate array, that defines the scaling:

    >>> scale = 0.1
    >>> output_shape = tuple(int(scale * s) for s in cube_shape)
    >>> coords0, coords1, coords2 = cp.mgrid[:output_shape[0],
    ...                    :output_shape[1], :output_shape[2]]
    >>> coords = cp.asarray([coords0, coords1, coords2])

    Assume that the cube contains spatial data, where the first array element
    center is at coordinate (0.5, 0.5, 0.5) in real space, i.e. we have to
    account for this extra offset when scaling the image:

    >>> coords = (coords + 0.5) / scale - 0.5
    >>> warped = warp(cube, coords)

    """  # noqa
    if map_args is None:
        map_args = {}

    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions",
                         image.shape)
    order = _validate_interpolation_order(image.dtype, order)

    if image.dtype.kind == "c":
        if not preserve_range:
            raise NotImplementedError("TODO")
    elif order > 0:
        image = convert_to_float(image, preserve_range)
        if image.dtype == cp.float16:
            image = image.astype(cp.float32)

    input_shape = np.array(image.shape)

    if output_shape is None:
        output_shape = input_shape
    else:
        output_shape = safe_as_int(output_shape)

    if isinstance(inverse_map, cp.ndarray) and inverse_map.shape == (3, 3,):
        # inverse_map is a transformation matrix as numpy array,
        # this is only used for order >= 4.
        inverse_map = ProjectiveTransform(matrix=inverse_map)

    if isinstance(inverse_map, cp.ndarray):
        # inverse_map is directly given as coordinates
        coords = inverse_map
    else:
        # inverse_map is given as function, that transforms (N, 2)
        # destination coordinates to their corresponding source
        # coordinates. This is only supported for 2(+1)-D images.

        if image.ndim < 2 or image.ndim > 3:
            raise ValueError("Only 2-D images (grayscale or color) are "
                             "supported, when providing a callable "
                             "`inverse_map`.")

        def coord_map(*args):
            return inverse_map(*args, **map_args)

        if len(input_shape) == 3 and len(output_shape) == 2:
            # Input image is 2D and has color channel, but output_shape is
            # given for 2-D images. Automatically add the color channel
            # dimensionality.
            output_shape = (output_shape[0], output_shape[1],
                            input_shape[2])

        coords = warp_coords(coord_map, output_shape)

    # Pre-filtering not necessary for order 0, 1 interpolation
    prefilter = order > 1

    ndi_mode = _to_ndimage_mode(mode)
    warped = ndi.map_coordinates(image, coords, prefilter=prefilter,
                                 mode=ndi_mode, order=order, cval=cval)

    _clip_warp_output(image, warped, mode, cval, order, clip)
    return warped


def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from Cartesian to polar coordinates

    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * cp.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * cp.cos(angle)) + center[1]
    coords = cp.column_stack((cc, rr))
    return coords


def _log_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from Cartesian to polar coordinates

    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = width / math.log(radius)``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((cp.exp(output_coords[:, 0] / k_radius)) * cp.sin(angle)) + center[0]
    cc = ((cp.exp(output_coords[:, 0] / k_radius)) * cp.cos(angle)) + center[1]
    coords = cp.column_stack((cc, rr))
    return coords


@channel_as_last_axis()
def warp_polar(image, center=None, *, radius=None, output_shape=None,
               scaling='linear', channel_axis=None, **kwargs):
    """Remap image to polar or log-polar coordinates space.

    Parameters
    ----------
    image : ndarray
        Input image. Only 2-D arrays are accepted by default. 3-D arrays are
        accepted if a `channel_axis` is specified.
    center : tuple (row, col), optional
        Point in image that represents the center of the transformation (i.e.,
        the origin in cartesian space). Values can be of type `float`.
        If no value is given, the center is assumed to be the center point
        of the image.
    radius : float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape : tuple (row, col), optional
    scaling : {'linear', 'log'}, optional
        Specify whether the image warp is polar or log-polar. Defaults to
        'linear'.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 22.02.00
           ``channel_axis`` was added in 22.02.00.
    **kwargs : keyword arguments
        Passed to `transform.warp`.

    Returns
    -------
    warped : ndarray
        The polar or log-polar warped image.

    Examples
    --------
    Perform a basic polar warp on a grayscale image:

    >>> from skimage import data
    >>> from cucim.skimage.transform import warp_polar
    >>> image = cp.array(data.checkerboard())
    >>> warped = warp_polar(image)

    Perform a log-polar warp on a grayscale image:

    >>> warped = warp_polar(image, scaling='log')

    Perform a log-polar warp on a grayscale image while specifying center,
    radius, and output shape:

    >>> warped = warp_polar(image, (100,100), radius=100,
    ...                     output_shape=image.shape, scaling='log')

    Perform a log-polar warp on a color image:

    >>> image = cp.array(data.astronaut())
    >>> warped = warp_polar(image, scaling='log', channel_axis=-1)
    """
    multichannel = channel_axis is not None
    if image.ndim != 2 and not multichannel:
        raise ValueError(f'Input array must be 2-dimensional when '
                         f'`channel_axis=None`, got {image.ndim}')

    if image.ndim != 3 and multichannel:
        raise ValueError(f'Input array must be 3-dimensional when '
                         f'`channel_axis` is specified, got {image.ndim}')

    if center is None:
        center = (np.array(image.shape)[:2] / 2) - 0.5

    if radius is None:
        w, h = np.array(image.shape)[:2] / 2
        radius = np.sqrt(w ** 2 + h ** 2)

    if output_shape is None:
        height = 360
        width = math.ceil(radius)
        output_shape = (height, width)
    else:
        output_shape = safe_as_int(output_shape)
        height = output_shape[0]
        width = output_shape[1]

    if scaling == 'linear':
        k_radius = width / radius
        map_func = _linear_polar_mapping
    elif scaling == 'log':
        k_radius = width / math.log(radius)
        map_func = _log_polar_mapping
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")

    k_angle = height / (2 * np.pi)
    warp_args = {'k_angle': k_angle, 'k_radius': k_radius, 'center': center}

    warped = warp(image, map_func, map_args=warp_args,
                  output_shape=output_shape, **kwargs)

    return warped


def _local_mean_weights(old_size, new_size, grid_mode, dtype):
    """Create a 2D weight matrix for resizing with the local mean.

    Parameters
    ----------
    old_size: int
        Old size.
    new_size: int
        New size.
    grid_mode : bool
        Whether to use grid data model of pixel/voxel model for
        average weights computation.
    dtype: dtype
        Output array data type.

    Returns
    -------
    weights: (new_size, old_size) array
        Rows sum to 1.

    """
    if grid_mode:
        old_breaks = cp.linspace(0, old_size, num=old_size + 1, dtype=dtype)
        new_breaks = cp.linspace(0, old_size, num=new_size + 1, dtype=dtype)
    else:
        old, new = old_size - 1, new_size - 1
        old_breaks = pad(cp.linspace(0.5, old - 0.5, old, dtype=dtype),
                         1, 'constant', constant_values=(0, old))
        if new == 0:
            val = np.inf
        else:
            val = 0.5 * old / new
        new_breaks = pad(cp.linspace(val, old - val, new, dtype=dtype),
                         1, 'constant', constant_values=(0, old))

    upper = cp.minimum(new_breaks[1:, np.newaxis], old_breaks[np.newaxis, 1:])
    lower = cp.maximum(new_breaks[:-1, np.newaxis],
                       old_breaks[np.newaxis, :-1])

    weights = cp.maximum(upper - lower, 0)
    weights /= weights.sum(axis=1, keepdims=True)

    return weights


def resize_local_mean(image, output_shape, grid_mode=True,
                      preserve_range=False, *, channel_axis=None):
    """Resize an array with the local mean / bilinear scaling.

    Parameters
    ----------
    image : ndarray
        Input image. If this is a multichannel image, the axis corresponding
        to channels should be specified using `channel_axis`
    output_shape : tuple or ndarray
        Size of the generated output image. When `channel_axis` is not None,
        the `channel_axis` should either be omitted from `output_shape` or the
        ``output_shape[channel_axis]`` must match
        ``image.shape[channel_axis]``. If the length of `output_shape` exceeds
        image.ndim, additional singleton dimensions will be appended to the
        input ``image`` as needed.
    grid_mode : bool, optional
        Defines ``image`` pixels position: if True, pixels are assumed to be at
        grid intersections, otherwise at cell centers. As a consequence,
        for example, a 1d signal of length 5 is considered to have length 4
        when `grid_mode` is False, but length 5 when `grid_mode` is True. See
        the following visual illustration:

        .. code-block:: text

                | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                     |<-------------------------------------->|
                                        vs.
                |<----------------------------------------------->|

        The starting point of the arrow in the diagram above corresponds to
        coordinate location 0 in each mode.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    See Also
    --------
    resize, downscale_local_mean

    Notes
    -----
    This method is sometimes referred to as "area-based" interpolation or
    "pixel mixing" interpolation [1]_. When `grid_mode` is True, it is
    equivalent to using OpenCV's resize with `INTER_AREA` interpolation mode.
    It is commonly used for image downsizing. If the downsizing factors are
    integers, then `downscale_local_mean` should be preferred instead.

    References
    ----------
    .. [1] http://entropymine.com/imageworsener/pixelmixing/

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.transform import resize_local_mean
    >>> image = cp.array(data.camera())
    >>> resize_local_mean(image, (100, 100)).shape
    (100, 100)

    """
    if channel_axis is not None:
        if channel_axis < -image.ndim or channel_axis >= image.ndim:
            raise ValueError("invalid channel_axis")

        # move channels to last position
        image = cp.moveaxis(image, channel_axis, -1)
        nc = image.shape[-1]

        output_ndim = len(output_shape)
        if output_ndim == image.ndim - 1:
            # insert channels dimension at the end
            output_shape = output_shape + (nc,)
        elif output_ndim == image.ndim:
            if output_shape[channel_axis] != nc:
                raise ValueError(
                    "Cannot reshape along the channel_axis. Use "
                    "channel_axis=None to reshape along all axes."
                )
            # move channels to last position in output_shape
            channel_axis = channel_axis % image.ndim
            output_shape = (
                output_shape[:channel_axis] + output_shape[channel_axis:] +
                (nc,)
            )
        else:
            raise ValueError(
                "len(output_shape) must be image.ndim or (image.ndim - 1) "
                "when a channel_axis is specified."
            )
        resized = image
    else:
        resized, output_shape = _preprocess_resize_output_shape(image,
                                                                output_shape)
    resized = convert_to_float(resized, preserve_range)
    dtype = resized.dtype

    for axis, (old_size, new_size) in enumerate(zip(image.shape,
                                                    output_shape)):
        if old_size == new_size:
            continue
        weights = _local_mean_weights(old_size, new_size, grid_mode, dtype)
        product = cp.tensordot(resized, weights, [[axis], [-1]])
        resized = cp.moveaxis(product, -1, axis)

    if channel_axis is not None:
        # restore channels to original axis
        resized = cp.moveaxis(resized, -1, channel_axis)

    return resized
