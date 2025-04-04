import math

import cupy as cp
import numpy as np

from cucim.skimage._vendored import ndimage as ndi, pad

from ._regionprops_gpu_basic_kernels import regionprops_bbox_coords
from ._regionprops_gpu_intensity_kernels import (
    _get_intensity_img_kernel_dtypes,
    get_intensity_measure_kernel,
)
from ._regionprops_gpu_utils import _find_close_labels, _get_min_integer_dtype

__all__ = [
    "regionprops_euler",
    "regionprops_perimeter",
    "regionprops_perimeter_crofton",
]

misc_deps = dict()
misc_deps["perimeter"] = ["slice"]
misc_deps["perimeter_crofton"] = ["slice"]
misc_deps["euler_number"] = ["slice"]


def _weighted_sum_of_filtered_image(
    label_image, max_label, image_filtered, coefs, pixels_per_thread=16
):
    """Compute weighted sums of pixels for each label.

    1. Apply the coefs LUT to the filtered image to get a coefficient image
    2. Sum the values in the coefficient image for each labeled region

    This function is used during computation of the Euler characteristic and
    perimeter properties.

    Parameters
    ----------
    label_image : cupy.ndarray
        Label image.
    max_label : int
        Maximum label value.
    image_filtered : (M, N) ndarray
        Filtered image (must have integer values that can be used to index into
        the coefs LUT)
    coefs : cupy.ndarray
        Coefficients look-up table (LUT).
    pixels_per_thread : int, optional
        Number of pixels per thread.

    Returns
    -------
    output : cupy.ndarray
        Weighted sum of pixels for each label.
    """
    coefs_image = coefs[image_filtered]

    # generate kernel for per-label weighted sum
    coefs_sum_kernel = get_intensity_measure_kernel(
        coefs_image.dtype,
        num_channels=1,
        compute_num_pixels=False,
        compute_sum=True,
        compute_sum_sq=False,
        compute_min=False,
        compute_max=False,
        pixels_per_thread=pixels_per_thread,
    )

    # prepare output array
    sum_dtype, _, _, _ = _get_intensity_img_kernel_dtypes(coefs_image.dtype)
    output = cp.zeros((max_label,), dtype=sum_dtype)

    coefs_sum_kernel(
        label_image,
        label_image.size,
        coefs_image,
        output,
        size=math.ceil(label_image.size / pixels_per_thread),
    )
    return output


@cp.memoize(for_each_device=True)
def _get_perimeter_weights_and_coefs(coefs_dtype=cp.float32):
    # convolution weights
    weights = cp.array(
        [[10, 2, 10], [2, 1, 2], [10, 2, 10]],
    )

    # LUT for weighted sum
    coefs = np.zeros(50, dtype=coefs_dtype)
    coefs[[5, 7, 15, 17, 25, 27]] = 1
    coefs[[21, 33]] = math.sqrt(2)
    coefs[[13, 23]] = (1 + math.sqrt(2)) / 2
    coefs = cp.asarray(coefs)
    return weights, coefs


def regionprops_perimeter(
    labels,
    neighborhood=4,
    *,
    max_label=None,
    robust=True,
    labels_close=None,
    props_dict=None,
    pixels_per_thread=10,
):
    """Calculate total perimeter of all objects in binary image.

    when `robust` is ``True``, reuses "slice" from `props_dict`

    writes "perimeter" to `props_dict`

    Parameters
    ----------
    labels : (M, N) ndarray
        Binary input image.
    neighborhood : 4 or 8, optional
        Neighborhood connectivity for border pixel determination. It is used to
        compute the contour. A higher neighborhood widens the border on which
        the perimeter is computed.

    Extra Parameters
    ----------------
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        If True, extra computation will be done to detect if any labeled
        regions are <=2 pixel spacing from another. Any regions that meet that
        criteria will have their perimeter recomputed in isolation to avoid
        possible error that would otherwise occur in this case. Turning this
        on will make the run time substantially longer, so it should only be
        used when labeled regions may have a non-negligible portion of their
        boundary within a <2 pixel gap from another label.
    labels_close : numpy.ndarray or sequence of int
        List of labeled regions that are less than 2 pixel gap from another
        label. Used when robust=True. If not provided and robust=True, it
        will be computed internally.
    props_dict : dict or None, optional
        Dictionary of pre-computed properties (e.g. "slice"). The output of this
        function will be stored under key "perimeter" within this dictionary.
    pixels_per_thread : int, optional
        Number of pixels processed per thread on the GPU during the final
        weighted summation.

    Returns
    -------
    perimeter : float
        Total perimeter of all objects in binary image.

    Notes
    -----
    The `perimeter` method does not consider the boundary along the image edge
    as image as part of the perimeter, while the `perimeter_crofton` method
    does. In any case, an object touching the image edge likely extends outside
    of the field of view, so an accurate perimeter cannot be measured for such
    objects.

    If the labeled regions have holes, the hole edges will be included in this
    measurement. If this is not desired, use regionprops_label_filled to fill
    the holes and then pass the filled labels image to this function.

    References
    ----------
    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of
           a Perimeter Estimator. The Queen's University of Belfast.
           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
    """
    if max_label is None:
        max_label = int(labels.max())

    binary_image = labels > 0
    if robust and labels_close is None:
        labels_close = _find_close_labels(labels, binary_image, max_label)
    if neighborhood == 4:
        footprint = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.uint8)
    else:
        footprint = 3

    eroded_image = ndi.binary_erosion(binary_image, footprint, border_value=0)
    border_image = binary_image.view(cp.uint8) - eroded_image

    perimeter_weights, perimeter_coefs = _get_perimeter_weights_and_coefs(
        cp.float32
    )

    perimeter_image = ndi.convolve(
        border_image,
        perimeter_weights,
        mode="constant",
        cval=0,
        output=cp.uint8,
    )

    min_integer_type = _get_min_integer_dtype(max_label, signed=False)

    # dilate labels by 1 pixel so we can sum with values in XF to give
    # unique histogram bins for each labeled regions (as long as no labeled
    # regions are within < 2 pixels from another labeled region)
    labels_dilated = ndi.grey_dilation(
        labels, 3, mode="constant", output=min_integer_type
    )

    if robust and labels_close.size > 0:
        if props_dict is not None and "slice" in props_dict:
            slices = props_dict["slice"]
        else:
            _, slices = regionprops_bbox_coords(labels, return_slices=True)

    # sum the coefficients for each label to compute the perimeter
    perimeters = _weighted_sum_of_filtered_image(
        label_image=labels_dilated,
        max_label=max_label,
        image_filtered=perimeter_image,
        coefs=perimeter_coefs,
        pixels_per_thread=pixels_per_thread,
    )
    if robust:
        # recompute perimeter in isolation for each region that may be too
        # close to another one
        shape = binary_image.shape
        for lab in labels_close:
            sl = slices[lab - 1]

            # keep boundary of 1 so object is not at 'edge' of cropped
            # region (unless it is at a true image edge)
            ld = labels[
                max(sl[0].start - 1, 0) : min(sl[0].stop + 1, shape[0]),
                max(sl[1].start - 1, 0) : min(sl[1].stop + 1, shape[1]),
            ]

            # print(f"{lab=}, {sl=}")
            # import matplotlib.pyplot as plt
            # plt.figure(); plt.imshow(ld.get()); plt.show()

            p = regionprops_perimeter(
                ld == lab, max_label=1, neighborhood=neighborhood, robust=False
            )
            perimeters[lab - 1] = p[0]
    if props_dict is not None:
        props_dict["perimeter"] = perimeters
    return perimeters


@cp.memoize(for_each_device=True)
def _get_perimeter_crofton_weights_and_coefs(
    directions, coefs_dtype=cp.float32
):
    # determine convolution weights
    filter_weights = cp.array(
        [[0, 0, 0], [0, 1, 4], [0, 2, 8]], dtype=cp.float32
    )

    if directions == 2:
        coefs = [
            0,
            np.pi / 2,
            0,
            0,
            0,
            np.pi / 2,
            0,
            0,
            np.pi / 2,
            np.pi,
            0,
            0,
            np.pi / 2,
            np.pi,
            0,
            0,
            0,
        ]
    else:
        sq2 = math.sqrt(2)
        coefs = [
            0,
            np.pi / 4 * (1 + 1 / sq2),
            np.pi / (4 * sq2),
            np.pi / (2 * sq2),
            0,
            np.pi / 4 * (1 + 1 / sq2),
            0,
            np.pi / (4 * sq2),
            np.pi / 4,
            np.pi / 2,
            np.pi / (4 * sq2),
            np.pi / (4 * sq2),
            np.pi / 4,
            np.pi / 2,
            0,
            0,
            0,
            0,
            0,
        ]
    coefs = cp.asarray(coefs, dtype=coefs_dtype)
    return filter_weights, coefs


def regionprops_perimeter_crofton(
    labels,
    directions=4,
    *,
    max_label=None,
    robust=True,
    omit_image_edges=False,
    labels_close=None,
    props_dict=None,
    pixels_per_thread=10,
):
    """Calculate total Crofton perimeter of all objects in binary image.

    when `robust` is ``True``, reuses "slice" from `props_dict`

    writes "perimeter_crofton" to `props_dict`

    Parameters
    ----------
    labels : (M, N) ndarray
        Input image. If image is not binary, all values greater than zero
        are considered as the object.
    directions : 2 or 4, optional
        Number of directions used to approximate the Crofton perimeter. By
        default, 4 is used: it should be more accurate than 2.
        Computation time is the same in both cases.

    Extra Parameters
    ----------------
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        If True, extra computation will be done to detect if any labeled
        regions are <=2 pixel spacing from another. Any regions that meet that
        criteria will have their perimeter recomputed in isolation to avoid
        possible error that would otherwise occur in this case. Turning this
        on will make the run time substantially longer, so it should only be
        used when labeled regions may have a non-negligible portion of their
        boundary within a <2 pixel gap from another label.
    omit_image_edges : bool, optional
        This can be set to avoid an additional padding step that includes the
        edges of objects that correspond to the image edge as part of the
        perimeter. We cannot accurately estimate the perimeter of objects
        falling partly outside of `image`, so it seems acceptable to just set
        this to True. The default remains False for consistency with upstream
        scikit-image.
    labels_close : numpy.ndarray or sequence of int
        List of labeled regions that are less than 2 pixel gap from another
        label. Used when robust=True. If not provided and robust=True, it
        will be computed internally.
    props_dict : dict or None, optional
        Dictionary of pre-computed properties (e.g. "slice"). The output of this
        function will be stored under key "perimeter_crofton" within this
        dictionary.
    pixels_per_thread : int, optional
        Number of pixels processed per thread on the GPU during the final
        weighted summation.

    Returns
    -------
    perimeter : float
        Total perimeter of all objects in binary image.

    Notes
    -----
    This measure is based on Crofton formula [1], which is a measure from
    integral geometry. It is defined for general curve length evaluation via
    a double integral along all directions. In a discrete
    space, 2 or 4 directions give a quite good approximation, 4 being more
    accurate than 2 for more complex shapes.

    Similar to :func:`~.measure.perimeter`, this function returns an
    approximation of the perimeter in continuous space.

    The `perimeter` method does not consider the boundary along the image edge
    as image as part of the perimeter, while the `perimeter_crofton` method
    does. In any case, an object touching the image edge likely extends outside
    of the field of view, so an accurate perimeter cannot be measured for such
    objects.

    If the labeled regions have holes, the hole edges will be included in this
    measurement. If this is not desired, use regionprops_label_filled to fill
    the holes and then pass the filled labels image to this function.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Crofton_formula
    .. [2] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010.
           Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838
    """
    if max_label is None:
        max_label = int(labels.max())

    ndim = labels.ndim
    if ndim not in [2, 3]:
        raise ValueError("labels must be 2D or 3D")

    binary_image = labels > 0
    if robust and labels_close is None:
        labels_close = _find_close_labels(labels, binary_image, max_label)

    footprint = 3  # scalar 3 -> (3, ) * ndim array of ones

    if not omit_image_edges:
        # Dilate labels by 1 pixel so we can sum with values in image_filtered
        # to give unique histogram bins for each labeled regions (As long as no
        # labeled regions are within < 2 pixels from another labeled region)
        labels_pad = cp.pad(labels, pad_width=1, mode="constant")
        labels_dilated = ndi.grey_dilation(labels_pad, 3, mode="constant")
        binary_image = pad(binary_image, pad_width=1, mode="constant")
        # need dilated mask for later use for indexing into
        # `image_filtered_labeled` for bincount
        binary_image_mask = ndi.binary_dilation(binary_image, footprint)
        binary_image_mask = cp.logical_xor(
            binary_image_mask, ndi.binary_erosion(binary_image, footprint)
        )
    else:
        labels_dilated = ndi.grey_dilation(labels, footprint, mode="constant")
        binary_image_mask = binary_image

    # determine convolution weights and LUT for weighted sum
    filter_weights, coefs = _get_perimeter_crofton_weights_and_coefs(
        directions, cp.float32
    )

    image_filtered = ndi.convolve(
        binary_image.view(cp.uint8),
        filter_weights,
        mode="constant",
        cval=0,
        output=cp.uint8,
    )

    if robust and labels_close.size > 0:
        if props_dict is not None and "slice" in props_dict:
            slices = props_dict["slice"]
        else:
            _, slices = regionprops_bbox_coords(labels, return_slices=True)

    # sum the coefficients for each label to compute the perimeter
    perimeters = _weighted_sum_of_filtered_image(
        label_image=labels_dilated,
        max_label=max_label,
        image_filtered=image_filtered,
        coefs=coefs,
        pixels_per_thread=pixels_per_thread,
    )
    if robust:
        # recompute perimeter in isolation for each region that may be too
        # close to another one
        shape = labels_dilated.shape
        for lab in labels_close:
            sl = slices[lab - 1]
            ld = labels[
                max(sl[0].start, 0) : min(sl[0].stop, shape[0]),
                max(sl[1].start, 0) : min(sl[1].stop, shape[1]),
            ]
            p = regionprops_perimeter_crofton(
                ld == lab,
                max_label=1,
                directions=directions,
                omit_image_edges=False,
                robust=False,
            )
            perimeters[lab - 1] = p[0]
    if props_dict is not None:
        props_dict["perimeter_crofton"] = perimeters
    return perimeters


@cp.memoize(for_each_device=True)
def _get_euler_weights_and_coefs(ndim, connectivity, coefs_dtype=cp.float32):
    from cucim.skimage.measure._regionprops_utils import (
        EULER_COEFS2D_4,
        EULER_COEFS2D_8,
        EULER_COEFS3D_26,
    )

    if ndim not in [2, 3]:
        raise ValueError("only 2D and 3D images are supported")

    if ndim == 2:
        filter_weights = cp.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]])
        if connectivity == 1:
            coefs = EULER_COEFS2D_4
        else:
            coefs = EULER_COEFS2D_8
        coefs = cp.asarray(coefs, dtype=coefs_dtype)
    else:  # 3D images
        if connectivity == 2:
            raise NotImplementedError(
                "For 3D images, Euler number is implemented "
                "for connectivities 1 and 3 only"
            )

        filter_weights = cp.array(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 4], [0, 2, 8]],
                [[0, 0, 0], [0, 16, 64], [0, 32, 128]],
            ]
        )
        if connectivity == 1:
            coefs = EULER_COEFS3D_26[::-1]
        else:
            coefs = EULER_COEFS3D_26
        coefs = cp.asarray(0.125 * coefs, dtype=coefs_dtype)

    return filter_weights, coefs


def regionprops_euler(
    labels,
    connectivity=None,
    *,
    max_label=None,
    robust=True,
    labels_close=None,
    props_dict=None,
    pixels_per_thread=10,
):
    """Calculate the Euler characteristic in binary image.

    For 2D objects, the Euler number is the number of objects minus the number
    of holes. For 3D objects, the Euler number is obtained as the number of
    objects plus the number of holes, minus the number of tunnels, or loops.

    when `robust` is ``True``, reuses "slice" from `props_dict`

    writes "euler_number" to `props_dict`

    Parameters
    ----------
    labels: (M, N[, P]) cupy.ndarray
        Input image. If image is not binary, all values greater than zero
        are considered as the object.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.
        4 or 8 neighborhoods are defined for 2D images (connectivity 1 and 2,
        respectively).
        6 or 26 neighborhoods are defined for 3D images, (connectivity 1 and 3,
        respectively). Connectivity 2 is not defined.

    Extra Parameters
    ----------------
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        If True, extra computation will be done to detect if any labeled
        regions are <=2 pixel spacing from another. Any regions that meet that
        criteria will have their perimeter recomputed in isolation to avoid
        possible error that would otherwise occur in this case. Turning this
        on will make the run time substantially longer, so it should only be
        used when labeled regions may have a non-negligible portion of their
        boundary within a <2 pixel gap from another label.
    labels_close : numpy.ndarray or sequence of int
        List of labeled regions that are less than 2 pixel gap from another
        label. Used when robust=True. If not provided and robust=True, it
        will be computed internally.
    props_dict : dict or None, optional
        Dictionary of pre-computed properties (e.g. "slice"). The output of this
        function will be stored under key "euler_number" within this dictionary.
    pixels_per_thread : int, optional
        Number of pixels processed per thread on the GPU during the final
        weighted summation.

    Returns
    -------
    euler_number : cp.ndarray of int
        Euler characteristic of the set of all objects in the image.

    Notes
    -----
    The Euler characteristic is an integer number that describes the
    topology of the set of all objects in the input image. If object is
    4-connected, then background is 8-connected, and conversely.

    The computation of the Euler characteristic is based on an integral
    geometry formula in discretized space. In practice, a neighborhood
    configuration is constructed, and a LUT is applied for each
    configuration. The coefficients used are the ones of Ohser et al.

    It can be useful to compute the Euler characteristic for several
    connectivities. A large relative difference between results
    for different connectivities suggests that the image resolution
    (with respect to the size of objects and holes) is too low.

    References
    ----------
    .. [1] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010. Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838
    .. [2] Ohser J., Nagel W., Schladitz K. (2002) The Euler Number of
           Discretized Sets - On the Choice of Adjacency in Homogeneous
           Lattices. In: Mecke K., Stoyan D. (eds) Morphology of Condensed
           Matter. Lecture Notes in Physics, vol 600. Springer, Berlin,
           Heidelberg.
    """

    if max_label is None:
        max_label = int(labels.max())

    # check connectivity
    if connectivity is None:
        connectivity = labels.ndim

    binary_image = labels > 0

    if robust and labels_close is None:
        labels_close = _find_close_labels(labels, binary_image, max_label)

    filter_weights, coefs = _get_euler_weights_and_coefs(
        labels.ndim, connectivity, cp.float32
    )
    binary_image = pad(binary_image, pad_width=1, mode="constant")
    image_filtered = ndi.convolve(
        binary_image.view(cp.uint8),
        filter_weights,
        mode="constant",
        cval=0,
        output=cp.uint8,
    )

    if robust and labels_close.size > 0:
        if props_dict is not None and "slice" in props_dict:
            slices = props_dict["slice"]
        else:
            _, slices = regionprops_bbox_coords(labels, return_slices=True)

    min_integer_type = _get_min_integer_dtype(max_label, signed=False)
    if labels.dtype != min_integer_type:
        labels = labels.astype(min_integer_type)
    # dilate labels by 1 pixel so we can sum with values in XF to give
    # unique histogram bins for each labeled regions (as long as no labeled
    # regions are within < 2 pixels from another labeled region)
    labels_pad = pad(labels, pad_width=1, mode="constant")
    labels_dilated = ndi.grey_dilation(labels_pad, 3, mode="constant")

    # sum the coefficients for each label to compute the Euler number
    euler_number = _weighted_sum_of_filtered_image(
        label_image=labels_dilated,
        max_label=max_label,
        image_filtered=image_filtered,
        coefs=coefs,
        pixels_per_thread=pixels_per_thread,
    )
    euler_number = euler_number.astype(cp.int64, copy=False)

    if robust:
        # recompute perimeter in isolation for each region that may be too
        # close to another one
        shape = labels_dilated.shape
        for lab in labels_close:
            sl = slices[lab - 1]
            # keep boundary of 1 so object is not at 'edge' of cropped
            # region (unless it is at a true image edge)
            # + 2 is because labels_pad is padded, but labels was not
            ld = labels_pad[
                max(sl[0].start, 0) : min(sl[0].stop + 2, shape[0]),
                max(sl[1].start, 0) : min(sl[1].stop + 2, shape[1]),
            ]
            euler_num = regionprops_euler(
                ld == lab, connectivity=connectivity, max_label=1, robust=False
            )
            euler_number[lab - 1] = euler_num[0]
    if props_dict is not None:
        props_dict["euler_number"] = euler_number
    return euler_number
