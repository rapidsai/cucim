# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi
from cucim.skimage.measure import label
from cucim.skimage.morphology import convex_hull_image

from ._regionprops_gpu_utils import (
    _get_coord_dtype,
    _get_count_dtype,
    _get_label_run_loop,
    _get_min_integer_dtype,
    _includes,
    _unravel_loop_index,
    _unravel_loop_index_declarations,
    _warp_includes,
)

__all__ = [
    "equivalent_diameter_area",
    "equivalent_diameter_area_2d",
    "equivalent_diameter_area_3d",
    "regionprops_area",
    "regionprops_area_bbox",
    "regionprops_bbox_coords",
    "regionprops_coords",
    "regionprops_extent",
    "regionprops_image",
    "regionprops_num_pixels",
    # extra functions for cuCIM not currently in scikit-image
    "regionprops_label_filled",
    "regionprops_num_boundary_pixels",
    "equivalent_spherical_perimeter",
    "regionprops_num_perimeter_pixels",
]


# Store information on which other properties a given property depends on
# This information will be used by `regionprops_dict` to make sure that when
# a particular property is requested any dependent properties are computed
# first.
basic_deps = dict()
basic_deps["area"] = ["num_pixels"]
basic_deps["area_bbox"] = ["bbox"]
basic_deps["area_filled"] = ["label_filled", "num_pixels_filled"]
basic_deps["bbox"] = []
basic_deps["coords"] = ["num_pixels"]
basic_deps["coords_scaled"] = ["num_pixels"]
basic_deps["equivalent_diameter_area"] = ["area"]
basic_deps["equivalent_spherical_perimeter"] = ["equivalent_diameter_area"]
basic_deps["extent"] = ["area", "area_bbox"]
basic_deps["image"] = []
basic_deps["image_filled"] = ["label_filled"]
basic_deps["image_intensity"] = []
basic_deps["label"] = []
basic_deps["label_filled"] = []
basic_deps["num_boundary_pixels"] = []
basic_deps["num_perimeter_pixels"] = []
basic_deps["num_pixels"] = []
basic_deps["num_pixels_filled"] = ["label_filled"]
basic_deps["perimeter_on_border_ratio"] = [
    "num_perimeter_pixels",
    "num_boundary_pixels",
]
basic_deps["slice"] = ["bbox"]


def _get_bbox_code(uint_t, ndim):
    """
    Notes
    -----
    Variables created (per run of a label):

        - bbox_min0, ..., bbox_min{ndim - 1}
            minimum coordinates of the run
        - bbox_max0, ..., bbox_max{ndim - 1}
            maximum coordinates of the run (exclusive)

    Output variables written to:

        - bbox : shape (max_label, 2 * ndim)
    """
    source_pre = ""
    source_start = ""
    source_operation = ""
    source_flush = ""
    reduce = []
    for d in range(ndim):
        reduce += [(f"bbox_min{d}", "min"), (f"bbox_max{d}", "max")]
        source_pre += f"""
    {uint_t} bbox_min{d}, bbox_max{d};"""
        source_start += f"""
          bbox_min{d} = ~static_cast<{uint_t}>(0);
          bbox_max{d} = 0;"""
        # op uses external coordinate array variables:
        #    in_coord[0]...in_coord[ndim - 1] : coordinates
        #        coordinates in the labeled image at the current index
        source_operation += f"""
          bbox_min{d} = min(in_coord[{d}], bbox_min{d});
          bbox_max{d} = max(in_coord[{d}] + 1, bbox_max{d});"""
        # flush uses external variables:
        #     lab : label value of the run
        #     bbox : output with shape (max_label, 2 * ndim)
        source_flush += f"""
          atomicMin(&bbox[(lab - 1)*{2 * ndim} + {d}], bbox_min{d});
          atomicMax(&bbox[(lab - 1)*{2 * ndim} + {d + ndim}], bbox_max{d});"""  # noqa: E501
    return source_pre, source_start, source_operation, source_flush, reduce


def _get_num_pixels_code():
    """
    Notes
    -----
    Variables created (per run of a label):

        - num_pixels : the number of pixels in the run

    Output variables written to:

        - counts : shape (max_label,)
    """
    source_pre = """
    uint32_t num_pixels;"""
    source_start = """
          num_pixels = 0;"""
    source_operation = """
          num_pixels += 1;"""
    # flush uses external variables:
    #     lab : label value of the run
    #     counts : output with shape (max_label,)
    source_flush = """
          atomicAdd(&counts[lab - 1], num_pixels);"""
    reduce = [("num_pixels", "sum")]
    return source_pre, source_start, source_operation, source_flush, reduce


def _get_coord_sums_code(coord_sum_ctype, ndim):
    """
    Notes
    -----
    Variables created (per run of a label):

        - coord_sum0, ..., coord_sum{ndim - 1}
            sum of the coordinates of the run

    Output variables written to:

        - coord_sums : shape (max_label, ndim)
    """
    source_pre = ""
    source_start = ""
    source_operation = ""
    source_flush = ""
    for d in range(ndim):
        source_pre += f"""
    {coord_sum_ctype} coord_sum{d};"""
        source_start += f"""
          coord_sum{d} = 0;"""
        # op uses external coordinate array variables:
        #    in_coord[0]...in_coord[ndim - 1] : coordinates
        #        coordinates in the labeled image at the current index
        source_operation += f"""
          coord_sum{d} += in_coord[{d}];"""
        # flush uses external variables:
        #     lab : label value of the run
        #     coord_sums : output with shape (max_label, ndim)
        source_flush += f"""
          atomicAdd(&coord_sums[(lab - 1) * {ndim} + {d}], coord_sum{d});"""
    reduce = [(f"coord_sum{d}", "sum") for d in range(ndim)]
    return source_pre, source_start, source_operation, source_flush, reduce


@cp.memoize(for_each_device=True)
def get_bbox_coords_kernel(
    ndim,
    int32_coords=True,
    int32_count=True,
    compute_bbox=True,
    compute_num_pixels=False,
    compute_coordinate_sums=False,
    pixels_per_thread=8,
):
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    if compute_num_pixels:
        count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)
    if compute_coordinate_sums:
        coord_sum_dtype = cp.dtype(cp.uint64)
        coord_sum_ctype = "uint64_t"

    if coord_dtype.itemsize <= 4:
        uint_t = "unsigned int"
    else:
        uint_t = "unsigned long long"

    if not (compute_bbox or compute_num_pixels or compute_coordinate_sums):
        raise ValueError("no computation requested")

    fragments = []
    if compute_bbox:
        fragments.append(_get_bbox_code(uint_t=uint_t, ndim=ndim))
    if compute_num_pixels:
        fragments.append(_get_num_pixels_code())
    if compute_coordinate_sums:
        fragments.append(
            _get_coord_sums_code(coord_sum_ctype=coord_sum_ctype, ndim=ndim)
        )
    pre, start, op, flush = ("".join(f[i] for f in fragments) for i in range(4))
    reduce = [r for f in fragments for r in f[4]]

    if compute_bbox or compute_coordinate_sums:
        pre += _unravel_loop_index_declarations("labels", ndim, uint_t=uint_t)
        op = (
            _unravel_loop_index(
                "labels",
                ndim=ndim,
                uint_t=uint_t,
                raveled_index="ii",
                omit_declarations=True,
            )
            + op
        )

    # store only counts for label > 0  (label = 0 is the background)
    source = _get_label_run_loop(
        pixels_per_thread, pre, start, op, flush, warp_reduce=reduce
    )

    inputs = "raw X labels, raw uint64 labels_size"
    outputs = []
    name = "cucim_"
    if compute_bbox:
        outputs.append(f"raw {coord_dtype.name} bbox")
        name += f"_bbox{ndim}d"
    if compute_num_pixels:
        outputs.append(f"raw {count_dtype.name} counts")
        name += f"_numpix_dtype{count_dtype.char}"
    if compute_coordinate_sums:
        outputs.append(f"raw {coord_sum_dtype.name} coord_sums")
        name += f"_csums_dtype{coord_sum_dtype.char}"
    outputs = ", ".join(outputs)
    name += f"_batch{pixels_per_thread}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_warp_includes, name=name
    )


def regionprops_num_perimeter_pixels(
    label_image,
    max_label=None,
    pixels_per_thread=16,
    props_dict=None,
):
    """Determine the number of pixels along the perimeter of each labeled
    region.

    This is a n-dimensional implementation so in 3D it is the number of pixels
    on the surface of the region.

    Writes "num_perimeter_pixels" to `props_dict` if it is not None.

    Notes
    -----
    If the labeled regions have holes, the hole edges will be included in this
    measurement. If this is not desired, use regionprops_label_filled to fill
    the holes and then pass the filled labels image to this function.

    For more accurate perimeter measurements, use `regionprops_perimeter` or
    `regionprops_perimeter_crofton` instead.
    """
    if max_label is None:
        max_label = int(label_image.max())
    # remove non-boundary pixels
    binary_label_mask = label_image > 0
    footprint = ndi.generate_binary_structure(label_image.ndim, connectivity=1)
    binary_label_mask_eroded = ndi.binary_erosion(binary_label_mask, footprint)
    labeled_edges = label_image * ~binary_label_mask_eroded

    num_perimeter_pixels = regionprops_num_pixels(
        labeled_edges,
        max_label=max_label,
        filled=False,
        pixels_per_thread=pixels_per_thread,
        props_dict=None,
    )
    if props_dict is not None:
        props_dict["num_perimeter_pixels"] = num_perimeter_pixels
    return num_perimeter_pixels


def regionprops_num_pixels(
    label_image,
    max_label=None,
    filled=False,
    pixels_per_thread=16,
    props_dict=None,
):
    """Determine the number of pixels in each labeled region.

    The `filled` flag should be used to indicate if the "label_image" has
    already had holes filled via `regionprops_label_filled`.

    if filled:
        - writes "num_pixels_filled" to `props_dict`
    else:
        - writes "num_pixels" to `props_dict`
    """

    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label
    num_pixels_prop_name = "num_pixels_filled" if filled else "num_pixels"

    count_dtype, int32_count = _get_count_dtype(label_image.size)

    pixels_kernel = get_bbox_coords_kernel(
        int32_count=int32_count,
        ndim=label_image.ndim,
        compute_bbox=False,
        compute_num_pixels=True,
        compute_coordinate_sums=False,
        pixels_per_thread=pixels_per_thread,
    )
    counts = cp.zeros(num_counts, dtype=count_dtype)

    # make a copy if the labels array is not C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    pixels_kernel(
        label_image,
        label_image.size,
        counts,
        size=math.ceil(label_image.size / pixels_per_thread),
    )
    if props_dict is not None:
        props_dict[num_pixels_prop_name] = counts
    return counts


def regionprops_area(
    label_image,
    spacing=None,
    max_label=None,
    dtype=cp.float32,
    filled=False,
    pixels_per_thread=16,
    props_dict=None,
):
    """Determine the area of each labeled region.

    if filled:
        - will reuse "num_pixels_filled" from `props_dict` if present
        - will write "area_filled" to `props_dict`
        - will write "num_pixels_filled" to `props_dict` if not already present
    else:
        - will reuse "num_pixels" from `props_dict` if present
        - will write "area" to `props_dict`
        - will write "num_pixels" to `props_dict` if not already present
    """
    if props_dict is None:
        props_dict = {}
    num_pixels_prop_name = "num_pixels_filled" if filled else "num_pixels"
    area_prop_name = "area_filled" if filled else "area"
    # integer atomicAdd is faster than floating point so better to convert
    # after counting
    if num_pixels_prop_name in props_dict:
        num_pixels = props_dict[num_pixels_prop_name]
    else:
        num_pixels = regionprops_num_pixels(
            label_image,
            max_label=max_label,
            filled=filled,
            pixels_per_thread=pixels_per_thread,
            props_dict=props_dict,
        )
        if num_pixels_prop_name not in props_dict:
            props_dict[num_pixels_prop_name] = num_pixels

    area = num_pixels.astype(dtype)
    if spacing is not None:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)
        area *= pixel_area

    if props_dict is not None:
        props_dict[area_prop_name] = area
    return area


@cp.fuse()
def equivalent_diameter_area_2d(area):
    """2d specialization of equivalent_diameter_area."""
    return cp.sqrt(4.0 * area / cp.pi)


@cp.fuse()
def equivalent_diameter_area_3d(area):
    """3d specialization of equivalent_diameter_area."""
    return cp.cbrt(6.0 * area / cp.pi)


@cp.fuse()
def equivalent_diameter_area_nd(area, ndim):
    """3d specialization of equivalent_diameter_area."""
    return cp.pow(2.0 * ndim * area / cp.pi, 1.0 / ndim)


def equivalent_diameter_area(area, ndim):
    """The formula is equivalent to ITK's HyperSphereRadiusFromVolume.

    This will be equal to 2 * GetEquivalentSphericalRadius() from ITK.

    Can be used to compute the "equivalent_diameter_area" property from
    the "area" property.
    """
    if ndim < 2:
        raise ValueError("ndim must be at least 2")
    if ndim == 2:
        return equivalent_diameter_area_2d(area)
    elif ndim == 3:
        return equivalent_diameter_area_3d(area)
    return equivalent_diameter_area_nd(area, float(ndim))


@cp.fuse()
def equivalent_spherical_perimeter(area, ndim, diameter):
    """Equivalent of ITK's GetEquivalentSphericalPerimeter

    Can be used to compute the "equivalent_spherical_perimeter" property from
    the "area" property.
    """
    return ndim * area / (0.5 * diameter)


def _regionprops_label_pass(
    label_image,
    max_label=None,
    *,
    compute_num_pixels=False,
    compute_bbox=False,
    compute_coordinate_sums=False,
    pixels_per_thread=16,
    props_dict=None,
):
    """Count pixels, find bounding boxes and sum coordinates in one pass.

    writes "num_pixels" to `props_dict` if `compute_num_pixels`
    writes "bbox" to `props_dict` if `compute_bbox`
    writes "coordinate_sums" (uint64 sums of the coordinates of each region,
    shape (max_label, ndim)) to `props_dict` if `compute_coordinate_sums`
    """
    if props_dict is None:
        props_dict = {}
    if max_label is None:
        max_label = int(label_image.max())
    ndim = label_image.ndim

    coord_dtype, int32_coords = _get_coord_dtype(label_image.size)
    count_dtype, int32_count = _get_count_dtype(label_image.size)
    kernel = get_bbox_coords_kernel(
        ndim=ndim,
        int32_coords=int32_coords,
        int32_count=int32_count,
        compute_bbox=compute_bbox,
        compute_num_pixels=compute_num_pixels,
        compute_coordinate_sums=compute_coordinate_sums,
        pixels_per_thread=pixels_per_thread,
    )

    # the outputs are passed in the order of the kernel's parameters
    outputs = []
    if compute_bbox:
        bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)
        # Initialize value for atomicMin on first ndim coordinates
        # The value for atomicMax columns is already 0 as desired.
        bbox_coords[:, :ndim] = cp.iinfo(coord_dtype).max
        outputs.append(bbox_coords)
    if compute_num_pixels:
        counts = cp.zeros(max_label, dtype=count_dtype)
        outputs.append(counts)
    if compute_coordinate_sums:
        coordinate_sums = cp.zeros((max_label, ndim), dtype=cp.uint64)
        outputs.append(coordinate_sums)

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    kernel(
        label_image,
        label_image.size,
        *outputs,
        size=math.ceil(label_image.size / pixels_per_thread),
    )
    if compute_bbox:
        props_dict["bbox"] = bbox_coords
    if compute_num_pixels:
        props_dict["num_pixels"] = counts
    if compute_coordinate_sums:
        props_dict["coordinate_sums"] = coordinate_sums
    return props_dict


def _bbox_slices(bbox_coords):
    """Convert bounding box coordinates to a list of tuples of slices."""
    ndim = bbox_coords.shape[1] // 2
    bbox_coords_cpu = cp.asnumpy(bbox_coords)
    if ndim == 2:
        # explicitly writing out the 2d case here for clarity
        return [
            (
                slice(int(box[0]), int(box[2])),
                slice(int(box[1]), int(box[3])),
            )
            for box in bbox_coords_cpu
        ]
    # general n-dimensional case
    return [
        tuple(slice(int(box[d]), int(box[d + ndim])) for d in range(ndim))
        for box in bbox_coords_cpu
    ]


def regionprops_bbox_coords(
    label_image,
    max_label=None,
    return_slices=False,
    pixels_per_thread=16,
    compute_num_pixels=False,
    props_dict=None,
):
    """Determine bounding box coordinates (and slices) of each labeled region.

    Writes "bbox" to `props_dict`

    if return_slices is True:
        - writes "slice" to `props_dict`

    if compute_num_pixels is True:
        - writes "num_pixels" to `props_dict` (counted in the same pass)

    Parameters
    ----------
    label_image : cp.ndarray
        Image containing labels where 0 is the background and sequential
        values > 0 are the labels.
    max_label : int or None
        The maximum label value present in label_image. Will be computed if not
        provided.
    return_slices : bool, optional
        If True, convert the bounding box coordinates array to a list of slice
        tuples.
    compute_num_pixels : bool, optional
        If True, also count the pixels of each region in the same pass and
        store the counts as "num_pixels" in `props_dict`.

    Returns
    -------
    bbox_coords : cp.ndarray
        Raw bounding box coordinates array. The first axis is indexed by
        ``label - 1``. The second axis has the minimum coordinate for dimension
        ``d`` at index ``2*d`` and the maximum for coordinate at dimension
        ``d`` at index ``2*d + 1``. Unlike for `bbox_slices`, the maximum
        coordinate in `bbox_coords` is **inclusive** (the region's bounding box
        includes both the min and max coordinate).
    bbox_slices : list[tuple[slice]] or None
        Will be None if return_slices is False. To get a mask corresponding to
        the ith label, use
        ``mask = label_image[bbox_slices[label - 1]] == label`` to get the
        region corresponding to the ith bounding box.
    """
    if props_dict is None:
        props_dict = {}
    _regionprops_label_pass(
        label_image,
        max_label=max_label,
        compute_num_pixels=compute_num_pixels,
        compute_bbox=True,
        pixels_per_thread=pixels_per_thread,
        props_dict=props_dict,
    )
    bbox_coords = props_dict["bbox"]
    if return_slices:
        bbox_slices = _bbox_slices(bbox_coords)
        props_dict["slice"] = bbox_slices
    else:
        bbox_slices = None

    return bbox_coords, bbox_slices


@cp.memoize(for_each_device=True)
def get_area_bbox_kernel(
    coord_dtype, area_dtype, ndim, compute_coordinate_sums=False
):
    coord_dtype = cp.dtype(coord_dtype)
    area_dtype = cp.dtype(area_dtype)
    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )

    source = f"""
       {uint_t} dim_max_offset;
       unsigned long long num_pixels_bbox = 1;
    """
    for d in range(ndim):
        source += f"""
           dim_max_offset = i * {2 * ndim} + {d + ndim};
           num_pixels_bbox *= bbox[dim_max_offset] - bbox[dim_max_offset - {ndim}];
        """  # noqa: E501
    source += """
        area_bbox = num_pixels_bbox * pixel_area;
    """
    inputs = f"raw {coord_dtype.name} bbox, float64 pixel_area"
    outputs = f"{area_dtype.name} area_bbox"
    name = f"cucim_area_bbox_{coord_dtype.name}_{area_dtype.name}_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_area_bbox(
    bbox, area_dtype=cp.float32, spacing=None, props_dict=None
):
    """Determine the area of the bounding box of each labeled region.

    Takes the "bbox" property as input.

    writes "area_bbox" to props_dict.
    """
    num_label = bbox.shape[0]
    ndim = bbox.shape[1] // 2

    if spacing is None:
        pixel_area = 1.0
    else:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)

    # make a copy if the inputs are not already C-contiguous
    if not bbox.flags.c_contiguous:
        bbox = cp.ascontiguousarray(bbox)

    kernel = get_area_bbox_kernel(bbox.dtype, area_dtype, ndim)
    area_bbox = cp.empty((num_label,), dtype=area_dtype)
    kernel(bbox, pixel_area, area_bbox)
    if props_dict is not None:
        props_dict["area_bbox"] = area_bbox
    return area_bbox


def regionprops_extent(area, area_bbox, props_dict=None):
    """Compute extent as the ratio of area / area_bbox for each labeled region.

    Takes the "area" and "area_bbox" properties as input.

    writes "extent" to props_dict.
    """
    extent = area / area_bbox
    if props_dict is not None:
        props_dict["extent"] = extent
    return extent


def regionprops_image(
    label_image,
    intensity_image=None,
    slices=None,
    max_label=None,
    compute_image=True,
    compute_convex=False,
    offset_coordinates=True,
    props_dict=None,
    on_cpu=False,
):
    """Return tuples of images of isolated label and/or intensity images.

        Each image incorporates only the bounding box region for a given label.

        This function can also optionally return convex images.

        Length of the tuple(s) is equal to `max_label`.

        reuses "slice" from `props_dict` if it is present

        if compute_image:
            - writes "image" to `props_dict`

        if compute_convex:
            - writes "image_convex" to `props_dict`
    4
        if intensity_image is not None:
            - writes "image_intensity" to `props_dict`
    """
    if max_label is None:
        max_label = int(label_image.max())
    if props_dict is None:
        props_dict = dict()

    if slices is None:
        if "slice" not in props_dict:
            regionprops_bbox_coords(
                label_image,
                max_label=max_label,
                return_slices=True,
                props_dict=props_dict,
            )
        slices = props_dict["slice"]

    # mask so there will only be a single label value in each returned slice
    masks = tuple(
        label_image[sl] == lab for lab, sl in enumerate(slices, start=1)
    )

    if compute_convex:
        convex_results = tuple(
            convex_hull_image(
                m,
                offset_coordinates=offset_coordinates,
                omit_empty_coords_check=True,
                float64_computation=True,
            )
            for m in masks
        )
        image_convex = convex_results

        if on_cpu:
            image_convex = tuple(cp.asnumpy(m) for m in image_convex)
        props_dict["image_convex"] = image_convex
    else:
        image_convex = None

    if on_cpu:
        masks = tuple(cp.asnumpy(m) for m in masks)
        if intensity_image is not None:
            intensity_image = cp.asnumpy(intensity_image)

    props_dict["image"] = masks

    if intensity_image is not None:
        if intensity_image.ndim > label_image.ndim:
            if intensity_image.ndim != label_image.ndim + 1:
                raise ValueError(
                    "Unexpected intensity_image.ndim. Should be "
                    "label_image.ndim or label_image.ndim + 1"
                )
            imslices = tuple(sl + (slice(None),) for sl in slices)
            intensity_images = tuple(
                intensity_image[sl] * mask[..., cp.newaxis]
                for img, (sl, mask) in enumerate(zip(imslices, masks), start=1)
            )

        else:
            intensity_images = tuple(
                intensity_image[sl] * mask
                for img, (sl, mask) in enumerate(zip(slices, masks), start=1)
            )
        if on_cpu:
            intensity_images = (cp.asnumpy(img) for img in intensity_images)
        props_dict["image_intensity"] = intensity_images
        if not compute_image:
            return props_dict["image_intensity"]
    else:
        intensity_images = None
    return masks, intensity_images, image_convex


def _get_compressed_labels(
    labels, max_label, intensity_image=None, sort_labels=True
):
    """Produce raveled list of coordinates and label values, excluding any
    background pixels.

    Some region properties can be applied to this data format more efficiently,
    than for the original labels image.

    Currently being used to compute "coords" and "coords_scaled" properties.
    """
    label_dtype = _get_min_integer_dtype(max_label, signed=False)
    if labels.dtype != label_dtype:
        labels = labels.astype(dtype=label_dtype)
    coords_dtype = _get_min_integer_dtype(max(labels.shape), signed=False)
    label_coords = cp.nonzero(labels)
    if label_coords[0].dtype != coords_dtype:
        label_coords = tuple(c.astype(coords_dtype) for c in label_coords)
    labels1d = labels[label_coords]
    if sort_labels:
        sort_indices = cp.argsort(labels1d)
        label_coords = tuple(c[sort_indices] for c in label_coords)
        labels1d = labels1d[sort_indices]
    if intensity_image:
        img1d = intensity_image[label_coords]
        return label_coords, labels1d, img1d
    return label_coords, labels1d


def regionprops_coords(
    label_image,
    max_label=None,
    spacing=None,
    compute_coords=True,
    compute_coords_scaled=False,
    props_dict=None,
):
    """Return tuple(s) of arrays of coordinates for each labeled region.

    Length of the tuple(s) is equal to `max_label`.

    reuses "num_pixels" from `props_dict` if it is present

    writes "coords" to `props_dict` if compute_coords is True

    writes "coords_scaled" to `props_dict` if compute_coords_scaled is True

    Notes
    -----
    This is provided only for completeness, but unlike for the RegionProps
    class, these are not needed in order to compute any of the other properties.
    """
    if max_label is None:
        max_label = int(label_image.max())
    if props_dict is None:
        props_dict = dict()

    coords_concat, _ = _get_compressed_labels(
        label_image, max_label=max_label, sort_labels=True
    )

    if "num_pixels" not in props_dict:
        num_pixels = regionprops_num_pixels(
            label_image, max_label=max_label, props_dict=props_dict
        )
    else:
        num_pixels = props_dict["num_pixels"]

    # stack ndim arrays into a single (pixels, ndim) array
    coords_concat = cp.stack(coords_concat, axis=-1)

    # scale based on spacing
    if compute_coords_scaled:
        max_exact_float32_int = 16777216  # 2 ** 24
        max_sz = max(label_image.shape)
        float_type = (
            cp.float32 if max_sz < max_exact_float32_int else cp.float64
        )
        coords_concat_scaled = coords_concat.astype(float_type)
        if spacing is not None:
            scale_factor = cp.asarray(spacing, dtype=float_type).reshape(1, -1)
            coords_concat_scaled *= scale_factor
        coords_scaled = []

    if compute_coords:
        coords = []

    # split separate labels out of the concatenated array above
    num_pixels_cpu = cp.asnumpy(num_pixels)
    slice_start = 0
    slice_stops = np.cumsum(num_pixels_cpu)
    for slice_stop in slice_stops:
        sl = slice(slice_start, slice_stop)
        if compute_coords:
            coords.append(coords_concat[sl, :])
        if compute_coords_scaled:
            coords_scaled.append(coords_concat_scaled[sl, :])
        slice_start = slice_stop

    if compute_coords:
        coords = tuple(coords)
        props_dict["coords"] = coords
        if not compute_coords_scaled:
            return coords
    if compute_coords_scaled:
        coords_scaled = tuple(coords_scaled)
        props_dict["coords_scaled"] = coords_scaled
        if not compute_coords:
            return coords_scaled
    return coords, coords_scaled


def _boundary_mask(labels):
    """Generate a binary mask corresponding to the pixels touching the image
    boundary.
    """
    ndim = labels.ndim
    slices = [
        slice(
            None,
        )
    ] * ndim
    boundary_mask = cp.zeros(labels.shape, dtype=bool)
    for d in range(ndim):
        edge_slices1 = slices[:d] + [slice(0, 1)] + slices[d + 1 :]
        edge_slices2 = slices[:d] + [slice(-1, None)] + slices[d + 1 :]
        boundary_mask[tuple(edge_slices1)] = 1
        boundary_mask[tuple(edge_slices2)] = 1
        slices[d] = slice(1, -1)
    return boundary_mask


def regionprops_num_boundary_pixels(labels, max_label=None, props_dict=None):
    """Determine the number of pixels touching the image boundary for each
    labeled region.

    writes "num_boundary_pixels" to props_dict.
    """
    if max_label is None:
        max_label = int(labels.max())

    # get mask of edge pixels
    boundary_mask = _boundary_mask(labels)

    # include a bin for the background
    nbins = max_label + 1
    # exclude background region from edge_counts
    edge_counts = cp.bincount(labels[boundary_mask], minlength=nbins)[1:]
    if props_dict is not None:
        props_dict["num_boundary_pixels"] = edge_counts
    return edge_counts


def _fill_holes_by_label(labels, max_label, bbox=None):
    """Fill the background holes of each labeled region.

    A hole is a connected component of the background (with full
    connectivity, as for ``binary_fill_holes`` with a ``3**ndim`` structuring
    element) that is not connected to the area outside of the image. A hole
    whose neighboring pixels all belong to a single region is filled with that
    region's label. This matches scikit-image's per-region ``image_filled`` for
    that region.

    scikit-image fills each region's holes treating all other labels as
    background, so a region's filled image also contains holes bordered by
    several regions and any regions nested in its holes. For the labels where
    this can happen, the returned ``label_filled`` is not exact and the labels
    are returned in ``labels_inexact`` so that the caller can recompute them
    on per-label crops. These are the labels that (a) border a hole that also
    borders other labels, or (b) touch a connected component of another label
    whose bounding box lies strictly inside their own (a component nested in
    a hole of another region must lie strictly inside that region's bounding
    box, but the other components of its label can lie anywhere, so the test
    uses the bounding boxes of the components, not of the labels).

    Returns
    -------
    label_filled : cupy.ndarray
        The label image with each single-region hole filled with its label.
    labels_inexact : numpy.ndarray
        Labels whose filled region may differ from scikit-image's.
    """
    ndim = labels.ndim
    footprint = ndi.generate_binary_structure(ndim, ndim)
    # with the padding, all background outside of the regions is a single
    # connected component
    padded = cp.pad(labels, 1, mode="constant", constant_values=0)
    work_dtype = cp.int32 if max_label < 2**31 - 1 else cp.int64
    padded = padded.astype(work_dtype, copy=False)
    background = padded == 0
    bg_labels = label(background, connectivity=ndim)
    num_bg = int(bg_labels.max())
    outer = int(bg_labels[(0,) * ndim])

    # largest and smallest label in the 3**ndim neighborhood of each pixel
    # (the background is excluded from the minimum)
    no_label = max_label + 1
    nb_max = ndi.grey_dilation(padded, size=3, mode="constant", cval=0)
    nb_min = ndi.grey_erosion(
        cp.where(background, no_label, padded).astype(work_dtype, copy=False),
        size=3,
        mode="constant",
        cval=no_label,
    )

    # labels bordering each background component
    bg_ids = bg_labels[background]
    hole_max = cp.zeros(num_bg + 1, dtype=work_dtype)
    hole_min = cp.full(num_bg + 1, no_label, dtype=work_dtype)
    cp.maximum.at(hole_max, bg_ids, nb_max[background])
    cp.minimum.at(hole_min, bg_ids, nb_min[background])
    component_ids = cp.arange(num_bg + 1)
    is_hole = (component_ids != 0) & (component_ids != outer)
    single = is_hole & (hole_min == hole_max)
    multi = is_hole & (hole_min != hole_max)

    fill_value = cp.where(single, hole_max, 0).astype(work_dtype, copy=False)
    filled = cp.where(background, fill_value[bg_labels], padded)
    core = (slice(1, -1),) * ndim
    label_filled = cp.ascontiguousarray(filled[core].astype(labels.dtype))

    # (a) labels bordering a hole that borders several labels
    inexact = []
    if bool(multi.any()):
        near_multi = ndi.binary_dilation(multi[bg_labels], structure=footprint)
        inexact.append(cp.unique(padded[near_multi & ~background]))

    # (b) labels touching a connected component of another label whose
    # bounding box lies strictly inside their own
    touching = (~background) & ((nb_max != padded) | (nb_min != padded))
    touching &= padded != 0
    if bool(touching.any()):
        if bbox is None:
            bbox, _ = regionprops_bbox_coords(labels, max_label=max_label)
        # connected components of each label (label separates neighboring
        # pixels with different values)
        components, num_components = label(
            labels, connectivity=ndim, return_num=True
        )
        bbox_components, _ = regionprops_bbox_coords(
            components, max_label=num_components
        )
        padded_components = cp.pad(
            components, 1, mode="constant", constant_values=0
        )
        no_component = num_components + 1
        pairs = []
        center = padded[core]
        for offset in np.ndindex((3,) * ndim):
            if all(o == 1 for o in offset):
                continue
            neighbor_slices = tuple(
                slice(o, o + s) for o, s in zip(offset, labels.shape)
            )
            neighbor = padded[neighbor_slices]
            mask = touching[core] & (neighbor != 0) & (neighbor != center)
            pairs.append(
                center[mask].astype(cp.int64) * no_component
                + padded_components[neighbor_slices][mask].astype(cp.int64)
            )
        pairs = cp.unique(cp.concatenate(pairs))
        outer_label = pairs // no_component
        inner_component = pairs % no_component
        bbox_outer = bbox[outer_label - 1].astype(cp.int64)
        bbox_inner = bbox_components[inner_component - 1].astype(cp.int64)
        nested = cp.all(
            bbox_inner[:, :ndim] > bbox_outer[:, :ndim], axis=1
        ) & cp.all(bbox_inner[:, ndim:] < bbox_outer[:, ndim:], axis=1)
        inexact.append(outer_label[nested])

    if inexact:
        labels_inexact = cp.asnumpy(cp.unique(cp.concatenate(inexact)))
    else:
        labels_inexact = np.zeros((0,), dtype=np.int64)
    return label_filled, labels_inexact


def regionprops_label_filled(
    labels,
    max_label=None,
    props_dict=None,
):
    """Fill holes in each labeled region.

    writes "label_filled" to props_dict

    Each background hole (a connected component of the background that is
    not connected to the area outside of the image) whose neighboring pixels
    all belong to one region is filled with that region's label. Holes that
    border several regions are left unfilled, and other labels are never
    changed.

    Parameters
    ----------
    labels : cupy.ndarray
        The label image
    max_label : the maximum label present in labels
        If None, will be determined internally.
    props_dict : dict or None
        Dictionary to store any measured properties.

    Returns
    -------
    label_filled : cupy.ndarray
        The label image, but with the holes in each region filled.
    """
    if max_label is None:
        max_label = int(labels.max())
    label_filled, _ = _fill_holes_by_label(labels, max_label)
    if props_dict is not None:
        props_dict["label_filled"] = label_filled
    return label_filled


def _regionprops_filled(
    label_image,
    label_filled,
    labels_inexact,
    max_label,
    spacing=None,
    compute_image=False,
    props_dict=None,
):
    """Compute num_pixels_filled, area_filled and (optionally) image_filled.

    The values come from `label_filled` (see `_fill_holes_by_label`), except
    for the labels in `labels_inexact`, which are recomputed one by one with
    ``binary_fill_holes`` on their bounding-box crop, as scikit-image does.

    writes "num_pixels_filled" and "area_filled" to `props_dict`
    writes "image_filled" to `props_dict` if `compute_image` is True
    """
    if props_dict is None:
        props_dict = {}
    num_pixels_filled = regionprops_num_pixels(
        label_filled, max_label=max_label
    )
    images = None
    if compute_image:
        images, _, _ = regionprops_image(
            label_filled,
            max_label=max_label,
            compute_image=True,
            compute_convex=False,
            props_dict=None,
        )
        images = list(images)
    if labels_inexact.size > 0:
        _, slices = regionprops_bbox_coords(
            label_image, max_label=max_label, return_slices=True
        )
        structure = cp.ones((3,) * label_image.ndim, dtype=bool)
        for lab in labels_inexact:
            image = label_image[slices[lab - 1]] == lab
            image_filled = ndi.binary_fill_holes(image, structure)
            num_pixels_filled[lab - 1] = image_filled.sum()
            if compute_image:
                images[lab - 1] = image_filled
    area_filled = num_pixels_filled.astype(cp.float32)
    if spacing is not None:
        area_filled *= math.prod(float(s) for s in spacing)
    props_dict["num_pixels_filled"] = num_pixels_filled
    props_dict["area_filled"] = area_filled
    if compute_image:
        props_dict["image_filled"] = tuple(images)
    return props_dict
