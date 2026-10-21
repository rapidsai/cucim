# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math
from collections.abc import Iterable
from numbers import Real

from .._shared.utils import convert_to_float
from .._vendored import ndimage as ndi
from ._pyramid_kernels import _gaussian_decimate
from ._warps import _validate_assume_finite, resize


def _smooth(image, sigma, mode, cval, channel_axis):
    """Return image with each channel smoothed by the Gaussian filter."""
    sigma_msg = "Sigma values less than zero are not valid"
    if not isinstance(sigma, Iterable):
        if sigma < 0:
            raise ValueError(sigma_msg)
    elif any(s < 0 for s in sigma):
        raise ValueError(sigma_msg)

    # apply Gaussian filter to all channels independently
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        axes = tuple(ax for ax in range(image.ndim) if ax != channel_axis)
    else:
        axes = None
    return ndi.gaussian_filter(
        image,
        sigma,
        mode=mode,
        cval=cval,
        axes=axes,
    )


def _check_factor(factor):
    if factor <= 1:
        raise ValueError("scale factor must be greater than 1")


def pyramid_reduce(
    image,
    downscale=2,
    sigma=None,
    order=1,
    mode="reflect",
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
    assume_finite=False,
):
    """Smooth and then downsample image.

    Parameters
    ----------
    image : ndarray
        Input image.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    assume_finite : bool, optional
        If True, assume the input and all intermediate values remain finite.
        Skip NaN detection during clipping. The input is not validated and
        constant mode requires a finite `cval`. Default is False, retaining
        NaN-aware clipping. Pyramid generators reuse this promise at every
        level without checking the image again.

    Returns
    -------
    out : array
        Smoothed and downsampled float image.

    Notes
    -----
    `assume_finite` is a cuCIM-only keyword-only argument; scikit-image does
    not provide it. If the finite-value promise is violated, output clipping
    may be skipped.

    Eligible even-sized float inputs with ``downscale=2`` and ``order=1``
    combine smoothing and downsampling in separable strided correlations.
    Reassociated arithmetic can change the last bits of floating-point results.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(downscale)
    _validate_assume_finite(mode, cval, assume_finite)

    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple(
            math.ceil(d / float(downscale)) if ax != channel_axis else d
            for ax, d in enumerate(image.shape)
        )
    else:
        out_shape = tuple(math.ceil(d / float(downscale)) for d in image.shape)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    # Half-pixel samples are adjacent pairs only on even, exactly halved axes.
    # Other scales, layouts, and large Gaussian supports use smooth + resize.
    # Bound the input size so kernel indexing and reflection periods fit int32.
    axes = tuple(ax for ax in range(image.ndim) if ax != channel_axis)
    if (
        downscale == 2
        and order == 1
        and image.dtype.char in "fd"
        and image.flags.c_contiguous
        and image.size < (1 << 30)
        and axes
        and all(image.shape[ax] > 0 and image.shape[ax] % 2 == 0 for ax in axes)
        and isinstance(sigma, Real)
        and 0 <= sigma <= 4
        and mode in ("reflect", "constant", "wrap")
        and isinstance(cval, Real)
        and math.isfinite(cval)
    ):
        return _gaussian_decimate(image, float(sigma), axes, mode, cval)

    smoothed = _smooth(image, sigma, mode, cval, channel_axis)
    out = resize(
        smoothed,
        out_shape,
        order=order,
        mode=mode,
        cval=cval,
        anti_aliasing=False,
        preserve_range=True,
        assume_finite=assume_finite,
    )

    return out


def pyramid_expand(
    image,
    upscale=2,
    sigma=None,
    order=1,
    mode="reflect",
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
    assume_finite=False,
):
    """Upsample and then smooth image.

    Parameters
    ----------
    image : ndarray
        Input image.
    upscale : float, optional
        Upscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * upscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of upsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    assume_finite : bool, optional
        If True, assume the input and all intermediate values remain finite.
        Skip NaN detection during clipping. The input is not validated and
        constant mode requires a finite `cval`. Default is False, retaining
        NaN-aware clipping. Pyramid generators reuse this promise at every
        level without checking the image again.

    Returns
    -------
    out : array
        Upsampled and smoothed float image.

    Notes
    -----
    `assume_finite` is a cuCIM-only keyword-only argument; scikit-image does
    not provide it. If the finite-value promise is violated, output clipping
    may be skipped.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(upscale)
    _validate_assume_finite(mode, cval, assume_finite)
    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple(
            math.ceil(upscale * d) if ax != channel_axis else d
            for ax, d in enumerate(image.shape)
        )
    else:
        out_shape = tuple(math.ceil(upscale * d) for d in image.shape)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * upscale / 6.0

    resized = resize(
        image,
        out_shape,
        order=order,
        mode=mode,
        cval=cval,
        anti_aliasing=False,
        preserve_range=True,
        assume_finite=assume_finite,
    )
    out = _smooth(resized, sigma, mode, cval, channel_axis)

    return out


def pyramid_gaussian(
    image,
    max_layer=-1,
    downscale=2,
    sigma=None,
    order=1,
    mode="reflect",
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
    assume_finite=False,
):
    """Yield images of the Gaussian pyramid formed by the input image.

    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.

    Note that the first image of the pyramid will be the original, unscaled
    image. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    assume_finite : bool, optional
        If True, assume the input and all intermediate values remain finite.
        Skip NaN detection during clipping. The input is not validated and
        constant mode requires a finite `cval`. Default is False, retaining
        NaN-aware clipping. Pyramid generators reuse this promise at every
        level without checking the image again.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    Notes
    -----
    `assume_finite` is a cuCIM-only keyword-only argument; scikit-image does
    not provide it. If the finite-value promise is violated, output clipping
    may be skipped.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(downscale)
    _validate_assume_finite(mode, cval, assume_finite)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim

    layer = 0
    current_shape = image.shape

    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        next_shape = tuple(
            math.ceil(d / float(downscale)) if ax != channel_axis else d
            for ax, d in enumerate(current_shape)
        )
        if next_shape == current_shape:
            break

        layer += 1

        layer_image = pyramid_reduce(
            prev_layer_image,
            downscale,
            sigma,
            order,
            mode,
            cval,
            preserve_range=True,
            channel_axis=channel_axis,
            assume_finite=assume_finite,
        )

        prev_layer_image = layer_image
        current_shape = layer_image.shape

        yield layer_image


def pyramid_laplacian(
    image,
    max_layer=-1,
    downscale=2,
    sigma=None,
    order=1,
    mode="reflect",
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
    assume_finite=False,
):
    """Yield images of the laplacian pyramid formed by the input image.

    Each layer contains the difference between the downsampled and the
    downsampled, smoothed image::

        layer = resize(prev_layer) - smooth(resize(prev_layer))

    Note that the first image of the pyramid will be the difference between the
    original, unscaled image and its smoothed version. The total number of
    images is `max_layer + 1`. In case all layers are computed, the last image
    is either a one-pixel image or the image where the reduction does not
    change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    assume_finite : bool, optional
        If True, assume the input and all intermediate values remain finite.
        Skip NaN detection during clipping. The input is not validated and
        constant mode requires a finite `cval`. Default is False, retaining
        NaN-aware clipping. Pyramid generators reuse this promise at every
        level without checking the image again.


    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    Notes
    -----
    `assume_finite` is a cuCIM-only keyword-only argument; scikit-image does
    not provide it. If the finite-value promise is violated, output clipping
    may be skipped.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf
    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html

    """  # noqa
    _check_factor(downscale)
    _validate_assume_finite(mode, cval, assume_finite)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    current_shape = image.shape

    smoothed_image = _smooth(image, sigma, mode, cval, channel_axis)
    yield image - smoothed_image

    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        shape_without_channels = list(current_shape)
        shape_without_channels.pop(channel_axis)
        shape_without_channels = tuple(shape_without_channels)
    else:
        shape_without_channels = current_shape

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    if max_layer == -1:
        max_layer = math.ceil(math.log(max(shape_without_channels), downscale))

    for layer in range(max_layer):
        if channel_axis is not None:
            out_shape = tuple(
                math.ceil(d / float(downscale)) if ax != channel_axis else d
                for ax, d in enumerate(current_shape)
            )
        else:
            out_shape = tuple(
                math.ceil(d / float(downscale)) for d in current_shape
            )

        resized_image = resize(
            smoothed_image,
            out_shape,
            order=order,
            mode=mode,
            cval=cval,
            anti_aliasing=False,
            preserve_range=True,
            assume_finite=assume_finite,
        )
        smoothed_image = _smooth(resized_image, sigma, mode, cval, channel_axis)
        current_shape = resized_image.shape

        yield resized_image - smoothed_image
