# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import operator

import cupy as cp

from .._shared import utils
from .._shared.utils import convert_to_float
from ._nl_means_denoising import _nl_means_denoising


@utils.channel_as_last_axis()
def denoise_nl_means(
    image,
    patch_size=7,
    patch_distance=11,
    h=0.1,
    fast_mode=True,
    sigma=0.0,
    *,
    preserve_range=False,
    channel_axis=None,
    compute_dtype=None,
    symmetric_patch=False,
):
    """Perform non-local means denoising on 2D-4D grayscale or RGB images.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image to be denoised, which can be 2D, 3D or 4D, and grayscale
        or multichannel (see ``channel_axis`` parameter). There can be any
        number of channels (does not strictly have to be RGB).
    patch_size : int, optional
        Size of patches used for denoising. Even values are increased by one.
    patch_distance : int, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : float, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches. A higher h results in a smoother image,
        at the expense of blurring features. For a Gaussian noise of standard
        deviation sigma, a rule of thumb is to choose the value of h to be
        sigma of slightly less.
    fast_mode : bool, optional
        If True (default value), a fast version of the non-local means
        algorithm is used. If False, the original version of non-local means is
        used. See the Notes section for more details about the algorithms.
    sigma : float, optional
        The standard deviation of the (Gaussian) noise.  If provided, a more
        robust computation of patch weights is computed that takes the expected
        noise variance into account (see Notes below).
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    compute_dtype : {None, numpy.float32, numpy.float64}, optional
        Floating point precision used for the computation. By default (None),
        the floating point type of the converted image is used (see Notes). A
        float32 computation is much faster on GPUs with limited double
        precision throughput, while float64 more closely reproduces
        scikit-image, which always uses double precision in fast mode. This
        option does not change the output dtype. This parameter is specific
        to cuCIM.
    symmetric_patch : bool, optional
        If True, the patch distances of ``fast_mode=True`` are computed over
        the full patch centered on each pixel. By default (False), they use
        the same patch as scikit-image, which has one element fewer per axis
        and is not centered on the pixel (see Notes). This has no effect when
        ``fast_mode=False``, whose patch is always centered. This parameter is
        specific to cuCIM.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as `image`.

    Notes
    -----

    The non-local means algorithm is well suited for denoising images with
    specific textures. The principle of the algorithm is to average the value
    of a given pixel with values of other pixels in a limited neighborhood,
    provided that the *patches* centered on the other pixels are similar enough
    to the patch centered on the pixel of interest.

    In the original version of the algorithm [1]_, corresponding to
    ``fast=False``, the computational complexity is::

        image.size * patch_size ** image.ndim * patch_distance ** image.ndim

    Hence, changing the size of patches or their maximal distance has a
    strong effect on computing times, especially for 3-D images.

    However, the default behavior corresponds to ``fast_mode=True``, for which
    another version of non-local means [2]_ is used, corresponding to a
    complexity of::

        image.size * patch_distance ** image.ndim

    The computing time depends only weakly on the patch size, because for
    each shift the patch distances of all pixels are computed together,
    reusing the partial sums shared by overlapping patches [2]_. In cuCIM,
    these are separable sums along each axis computed in GPU shared memory,
    so the cost per pixel and shift grows only linearly with the patch size.
    This approach has been proven to be more efficient compared to other
    alternatives, see e.g. [3]_.

    Compared to the classic algorithm, all pixels of a patch contribute
    to the distance to another patch with the same weight, no matter
    their distance to the center of the patch. This coarser computation
    of the distance can result in a slightly poorer denoising
    performance. Moreover, for small images (images with a linear size
    that is only a few times the patch size), the classic algorithm can
    be faster due to boundary effects.

    The image is padded using the `reflect` mode of `skimage.util.pad`
    before denoising.

    If the noise standard deviation, `sigma`, is provided a more robust
    computation of patch weights is used.  Subtracting the known noise variance
    from the computed patch distances improves the estimates of patch
    similarity, giving a moderate improvement to denoising performance [4]_.
    It was also mentioned as an option for the fast variant of the algorithm in
    [3]_.

    When `sigma` is provided, a smaller `h` should typically be used to
    avoid oversmoothing.  The optimal value for `h` depends on the image
    content and noise level, but a reasonable starting point is
    ``h = 0.8 * sigma`` when `fast_mode` is `True`, or ``h = 0.6 * sigma`` when
    `fast_mode` is `False`.

    The following notes are specific to cuCIM:

    The GPU implementation reproduces the patch weights of scikit-image's
    implementation (including its fast approximation of the exponential), so
    results agree with scikit-image up to floating point rounding. The
    Gaussian patch weights of ``fast_mode=False`` are also separable, so both
    modes use the same separable sums and the computational cost of
    ``fast_mode=False`` is only modestly higher than that of
    ``fast_mode=True``.

    scikit-image computes the ``fast_mode=True`` patch distances from an
    integral image of the squared differences over the whole image, so their
    rounding error grows with the image size (to a relative error of about
    1e-9 for an 8192 x 8192 image). cuCIM sums only the values within each
    patch, so its patch distances are more accurate, with a relative error
    independent of the image size. This also keeps the float32 computation
    accurate (relative error of about 2e-7), which would not be the case with
    a float32 integral image.

    For ``fast_mode=True``, scikit-image's integral-image lookup uses corners
    at offsets of ``-r`` and ``+r`` from each pixel, where
    ``r = patch_size // 2``. The resulting box covers ``2 * r`` elements per
    axis, from ``r - 1`` elements before the pixel to ``r`` after it, instead
    of the ``2 * r + 1`` elements of the patch centered on the pixel, while
    the distance is still normalized by the size of the centered patch. The
    default behavior reproduces this, so that results match scikit-image. A
    consequence is that the result depends on the image orientation:
    denoising a flipped image does not give the flipped denoised image.
    ``symmetric_patch=True`` instead uses the centered patch, as
    ``fast_mode=False`` does. Its results differ from scikit-image's, and it
    is slightly slower because the patch is larger.

    Images with 8 or 16-bit integer (or boolean) dtypes are processed and
    returned as float32, while 32 or 64-bit integer images use float64
    (scikit-image returns float64 for all integer dtypes). float16 images are
    processed as float32.

    The output always has the same shape as `image`. Unlike scikit-image,
    singleton spatial dimensions are not removed from the result.

    For ``fast_mode=False``, scikit-image can produce non-finite weights when
    the final patch distance exceeds ~700. Such patches are given zero weight
    here instead.

    References
    ----------
    .. [1] A. Buades, B. Coll, & J-M. Morel. A non-local algorithm for image
           denoising. In CVPR 2005, Vol. 2, pp. 60-65, IEEE.
           :DOI:`10.1109/CVPR.2005.38`

    .. [2] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
           nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
           International Symposium on Biomedical Imaging: From Nano to Macro,
           2008, pp. 1331-1334.
           :DOI:`10.1109/ISBI.2008.4541250`

    .. [3] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
           Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
           :DOI:`10.5201/ipol.2014.120`

    .. [4] A. Buades, B. Coll, & J-M. Morel. Non-Local Means Denoising.
           Image Processing On Line, 2011, vol. 1, pp. 208-212.
           :DOI:`10.5201/ipol.2011.bcm_nlm`

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.restoration import denoise_nl_means
    >>> a = cp.zeros((40, 40))
    >>> a[10:-10, 10:-10] = 1.
    >>> rng = cp.random.default_rng()
    >>> a += 0.3 * rng.standard_normal(a.shape)
    >>> denoised_a = denoise_nl_means(a, 7, 5, 0.1)
    """
    if channel_axis is None:
        multichannel = False
        image = image[..., cp.newaxis]
    else:
        multichannel = True

    ndim_no_channel = image.ndim - 1
    if (ndim_no_channel < 2) or (ndim_no_channel > 4):
        raise NotImplementedError(
            "Non-local means denoising is only implemented for 2D, "
            "3D or 4D grayscale or multichannel images."
        )
    if not fast_mode:
        if ndim_no_channel == 3 and multichannel:
            raise NotImplementedError(
                "Multichannel 3D requires fast_mode to be True."
            )
        if ndim_no_channel == 4:
            raise NotImplementedError("4D requires fast_mode to be True.")

    patch_size = operator.index(patch_size)
    patch_distance = operator.index(patch_distance)
    if patch_size < 0:
        raise ValueError("patch_size must be non-negative")
    if patch_distance < 0:
        raise ValueError("patch_distance must be non-negative")
    h = float(h)
    if not h > 0:
        raise ValueError("h must be positive")
    sigma = float(sigma)

    if image.dtype.kind == "c":
        raise ValueError("complex-valued images are not supported")
    image = convert_to_float(image, preserve_range)
    float_dtype = image.dtype
    if compute_dtype is None:
        compute_dtype = float_dtype
    else:
        compute_dtype = cp.dtype(compute_dtype)
        if compute_dtype.char not in "fd":
            raise ValueError(
                "compute_dtype must be None, numpy.float32 or numpy.float64"
            )

    out = _nl_means_denoising(
        image.astype(compute_dtype, copy=False),
        patch_size,
        patch_distance,
        h,
        sigma * sigma,
        fast_mode,
        symmetric_patch=symmetric_patch,
    )
    out = out.astype(float_dtype, copy=False)
    if not multichannel:
        out = out[..., 0]
    return out
