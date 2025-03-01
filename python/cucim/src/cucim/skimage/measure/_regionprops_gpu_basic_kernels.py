import math

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi
from cucim.skimage.measure import label
from cucim.skimage.morphology import convex_hull_image

from ._regionprops_gpu_utils import (
    _get_count_dtype,
    _get_min_integer_dtype,
    _includes,
    _unravel_loop_index,
    _unravel_loop_index_declarations,
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


def _get_bbox_code(uint_t, ndim, array_size):
    """
    Notes
    -----
    Local variables created:

        - bbox_min : shape (array_size, ndim)
            local minimum coordinates across the local set of labels encountered
        - bbox_max : shape (array_size, ndim)
            local maximum coordinates across the local set of labels encountered

    Output variables written to:

        - bbox : shape (max_label, 2 * ndim)
    """

    # declaration uses external variable:
    #    labels_size : total number of pixels in the label image
    source_pre = f"""
    // bounding box variables
    {uint_t} bbox_min[{ndim * array_size}];
    {uint_t} bbox_max[{ndim * array_size}] = {{0}};
    // initialize minimum coordinate to array size
    for (size_t ii = 0; ii < {ndim * array_size}; ii++) {{
      bbox_min[ii] = labels_size;
    }}\n"""

    # op uses external coordinate array variables:
    #    in_coord[0]...in_coord[ndim - 1] : coordinates
    #        coordinates in the labeled image at the current index
    source_operation = f"""
          bbox_min[{ndim}*offset] = min(in_coord[0], bbox_min[{ndim}*offset]);
          bbox_max[{ndim}*offset] = max(in_coord[0] + 1, bbox_max[{ndim}*offset]);"""  # noqa: E501
    for d in range(ndim):
        source_operation += f"""
          bbox_min[{ndim}*offset + {d}] = min(in_coord[{d}], bbox_min[{ndim}*offset + {d}]);
          bbox_max[{ndim}*offset + {d}] = max(in_coord[{d}] + 1, bbox_max[{ndim}*offset + {d}]);"""  # noqa: E501

    # post_operation uses external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     bbox : output with shape (max_label, 2 * ndim)
    source_post = f"""
          // bounding box outputs
          atomicMin(&bbox[(lab - 1)*{2 * ndim}], bbox_min[{ndim}*ii]);
          atomicMax(&bbox[(lab - 1)*{2 * ndim} + {ndim}], bbox_max[{ndim}*ii]);"""  # noqa: E501
    for d in range(1, ndim):
        source_post += f"""
          atomicMin(&bbox[(lab - 1)*{2*ndim} + {d}], bbox_min[{ndim}*ii + {d}]);
          atomicMax(&bbox[(lab - 1)*{2*ndim} + {d + ndim}], bbox_max[{ndim}*ii + {d}]);"""  # noqa: E501
    return source_pre, source_operation, source_post


def _get_num_pixels_code(pixels_per_thread, array_size):
    """
    Notes
    -----
    Local variables created:

        - num_pixels : shape (array_size, )
            The number of pixels encountered per label value

    Output variables written to:

        - counts : shape (max_label,)
    """
    pixel_count_dtype = "int8_t" if pixels_per_thread < 256 else "int16_t"

    source_pre = f"""
    // num_pixels variables
    {pixel_count_dtype} num_pixels[{array_size}] = {{0}};\n"""

    source_operation = """
        num_pixels[offset] += 1;\n"""

    # post_operation requires external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     counts : output with shape (max_label,)
    source_post = """
        atomicAdd(&counts[lab - 1], num_pixels[ii]);;"""
    return source_pre, source_operation, source_post


def _get_coord_sums_code(coord_sum_ctype, ndim, array_size):
    """
    Notes
    -----
    Local variables created:

        - coord_sum : shape (array_size, ndim)
            local sum of coordinates across the local set of labels encountered

    Output variables written to:

        - coord_sums : shape (max_label, 2 * ndim)
    """

    source_pre = f"""
    {coord_sum_ctype} coord_sum[{ndim * array_size}] = {{0}};\n"""

    # op uses external coordinate array variables:
    #    in_coord[0]...in_coord[ndim - 1] : coordinates
    #        coordinates in the labeled image at the current index
    source_operation = f"""
        coord_sum[{ndim}*offset] += in_coord[0];"""
    for d in range(1, ndim):
        source_operation += f"""
        coord_sum[{ndim}*offset + {d}] += in_coord[{d}];"""
    # post_operation uses external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     coord_sums : output with shape (max_label, ndim)
    source_post = f"""
        // bounding box outputs
        atomicAdd(&coord_sums[(lab - 1) * {ndim}], coord_sum[{ndim}*ii]);"""
    for d in range(1, ndim):
        source_post += f"""
        atomicAdd(&coord_sums[(lab - 1) * {ndim} + {d}],
                  coord_sum[{ndim}*ii + {d}]);"""
    return source_pre, source_operation, source_post


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

    array_size = pixels_per_thread

    if coord_dtype.itemsize <= 4:
        uint_t = "unsigned int"
    else:
        uint_t = "unsigned long long"

    if not (compute_bbox or compute_num_pixels or compute_coordinate_sums):
        raise ValueError("no computation requested")

    if compute_bbox:
        bbox_pre, bbox_op, bbox_post = _get_bbox_code(
            uint_t=uint_t, ndim=ndim, array_size=array_size
        )
    if compute_num_pixels:
        count_pre, count_op, count_post = _get_num_pixels_code(
            pixels_per_thread=pixels_per_thread, array_size=array_size
        )
    if compute_coordinate_sums:
        coord_sums_pre, coord_sums_op, coord_sums_post = _get_coord_sums_code(
            coord_sum_ctype=coord_sum_ctype, ndim=ndim, array_size=array_size
        )
    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      uint64_t start_index = {pixels_per_thread}*i;
    """
    if compute_bbox:
        source += bbox_pre
    if compute_num_pixels:
        source += count_pre
    if compute_coordinate_sums:
        source += coord_sums_pre

    inner_op = ""
    if compute_bbox or compute_coordinate_sums:
        source += _unravel_loop_index_declarations(
            "labels", ndim, uint_t=uint_t
        )

        inner_op += _unravel_loop_index(
            "labels",
            ndim=ndim,
            uint_t=uint_t,
            raveled_index="ii",
            omit_declarations=True,
        )
    if compute_bbox:
        inner_op += bbox_op
    if compute_num_pixels:
        inner_op += count_op
    if compute_coordinate_sums:
        inner_op += coord_sums_op

    source += f"""
      X encountered_labels[{array_size}] = {{0}};
      X current_label;
      X prev_label = labels[start_index];
      int offset = 0;
      encountered_labels[0] = prev_label;
      uint64_t ii_max = min(start_index + {pixels_per_thread}, labels_size);
      for (uint64_t ii = start_index; ii < ii_max; ii++) {{
        current_label = labels[ii];
        if (current_label == 0) {{ continue; }}
        if (current_label != prev_label) {{
            offset += 1;
            prev_label = current_label;
            encountered_labels[offset] = current_label;
        }}
        {inner_op}
      }}"""
    source += """
      for (size_t ii = 0; ii <= offset; ii++) {
        X lab = encountered_labels[ii];
        if (lab != 0) {"""

    if compute_bbox:
        source += bbox_post
    if compute_num_pixels:
        source += count_post
    if compute_coordinate_sums:
        source += coord_sums_post
    source += """
        }
      }\n"""

    # print(source)
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
        inputs, outputs, source, preamble=_includes, name=name
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


def regionprops_bbox_coords(
    label_image,
    max_label=None,
    return_slices=False,
    pixels_per_thread=16,
    props_dict=None,
):
    """Determine bounding box coordinates (and slices) of each labeled region.

    Writes "bbox" to `props_dict`

    if return_slices is True:
        - writes "slice" to `props_dict`

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
    if max_label is None:
        max_label = int(label_image.max())

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)

    bbox_kernel = get_bbox_coords_kernel(
        ndim=label_image.ndim,
        int32_coords=int32_coords,
        pixels_per_thread=pixels_per_thread,
    )

    ndim = label_image.ndim
    bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

    # Initialize value for atomicMin on first ndim coordinates
    # The value for atomicMax columns is already 0 as desired.
    bbox_coords[:, :ndim] = cp.iinfo(coord_dtype).max

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    bbox_kernel(
        label_image,
        label_image.size,
        bbox_coords,
        size=math.ceil(label_image.size / pixels_per_thread),
    )
    if props_dict is not None:
        props_dict["bbox"] = bbox_coords

    if return_slices:
        bbox_coords_cpu = cp.asnumpy(bbox_coords)
        if ndim == 2:
            # explicitly writing out the 2d case here for clarity
            bbox_slices = [
                (
                    slice(int(box[0]), int(box[2])),
                    slice(int(box[1]), int(box[3])),
                )
                for box in bbox_coords_cpu
            ]
        else:
            # general n-dimensional case
            bbox_slices = [
                tuple(
                    slice(int(box[d]), int(box[d + ndim])) for d in range(ndim)
                )
                for box in bbox_coords_cpu
            ]
        if props_dict is not None:
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


def regionprops_label_filled(
    labels,
    max_label=None,
    props_dict=None,
):
    """Fill holes in each labeled region.

    writes "label_filled" to props_dict

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

    # make sure all background pixels at the boundary have the same label
    labels = cp.pad(labels, 1, mode="constant", constant_values=0)

    binary_labels = labels > 0

    # assign unique labels the background and holes
    inverse_binary_mask = ~binary_labels
    inverse_labels = label(inverse_binary_mask)

    # After inversion, what was originally the background will now be the
    # first foreground label encountered. This is ensured due to the
    # single voxel padding done above and the fact that the `label`
    # function scans linearly through the array.
    background_index = 1
    # set the background back to 0 in the inverse mask so we have a mask
    # of just the holes
    inverse_binary_mask[inverse_labels == background_index] = 0

    # add binary holes to the original mask and relabel
    binary_holes_filled = cp.logical_or(binary_labels, inverse_binary_mask)
    label_filled = label(binary_holes_filled)

    if props_dict is not None:
        props_dict["label_filled"] = label_filled
    label_filled = label_filled[(slice(1, -1),) * label_filled.ndim]
    return cp.ascontiguousarray(label_filled)
