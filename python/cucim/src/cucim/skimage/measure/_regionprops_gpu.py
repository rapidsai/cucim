import warnings
from copy import copy

import cupy as cp

from cucim.skimage.measure._regionprops import (
    COL_DTYPES,
    PROPS,
)

from ._regionprops_gpu_basic_kernels import (
    basic_deps,
    equivalent_diameter_area,
    equivalent_spherical_perimeter,
    regionprops_area,
    regionprops_area_bbox,
    regionprops_bbox_coords,
    regionprops_coords,
    regionprops_extent,
    regionprops_image,
    regionprops_label_filled,
    regionprops_num_boundary_pixels,
    regionprops_num_perimeter_pixels,
    regionprops_num_pixels,
)
from ._regionprops_gpu_intensity_kernels import (
    intensity_deps,
    regionprops_intensity_mean,
    regionprops_intensity_min_max,
    regionprops_intensity_std,
)
from ._regionprops_gpu_utils import _get_min_integer_dtype

__all__ = [
    "equivalent_diameter_area",
    "regionprops_area",
    "regionprops_area_bbox",
    "regionprops_bbox_coords",
    "regionprops_coords",
    "regionprops_dict",
    "regionprops_extent",
    "regionprops_image",
    "regionprops_intensity_mean",
    "regionprops_intensity_min_max",
    "regionprops_intensity_std",
    # extra functions for cuCIM not currently in scikit-image
    "equivalent_spherical_perimeter",  # as in ITK
    "regionprops_num_boundary_pixels",
    "regionprops_num_perimeter_pixels",
    "regionprops_label_filled",
]


# Master list of properties currently supported by regionprops_dict for faster
# computation on the GPU.
#
# One caveat is that centroid/moment/inertia_tensor properties currently only
# support 2D and 3D data with moments up to 3rd order.

# all properties from PROPS have been implemented
PROPS_GPU = copy(PROPS)
# extra properties not currently in scikit-image
PROPS_GPU_EXTRA = {
    "num_pixels_filled": "num_pixels_filled",
    # a few extra parameters as in ITK
    "num_perimeter_pixels": "num_perimeter_pixels",
    "num_boundary_pixels": "num_boundary_pixels",
    "perimeter_on_border_ratio": "perimeter_on_border_ratio",
    "equivalent_spherical_perimeter": "equivalent_spherical_perimeter",
}
PROPS_GPU.update(PROPS_GPU_EXTRA)

CURRENT_PROPS_GPU = set(PROPS_GPU.values())

COL_DTYPES_EXTRA = {
    "num_pixels_filled": int,
    "num_perimeter_pixels": int,
    "num_boundary_pixels": int,
    "perimeter_on_border_ratio": float,
    "equivalent_spherical_perimeter": float,
}

# expand column dtypes from _regionprops to include the extra properties
COL_DTYPES_GPU = copy(COL_DTYPES)
COL_DTYPES_GPU.update(COL_DTYPES_EXTRA)

# Any extra 'property' that is computed on the full labels image and not
# per-region.
GLOBAL_PROPS = {"label_filled"}

# list of the columns that are stored as a numpy object array when converted
# to tabular format by `regionprops_table`
OBJECT_COLUMNS_GPU = [
    col for col, dtype in COL_DTYPES_GPU.items() if dtype == object
]


# `property_deps` is a dictionary where each key is a property and values are
# the other properties that property directly depends on (indirect dependencies
# do not need to be listed as that is handled by traversing a tree structure via
# get_property_dependencies below).
property_deps = dict()
property_deps.update(basic_deps)
property_deps.update(intensity_deps)

# set of properties that only supports 2D images
ndim_2_only = set()


def get_property_dependencies(dependencies, node):
    """Get all direct and indirect dependencies for a specific property"""
    visited = set()
    result = []

    def depth_first_search(n):
        if n not in visited:
            visited.add(n)
            if n in dependencies:
                for dep in dependencies[n]:
                    depth_first_search(dep)
            # If n is not in dependencies, assume it has no dependencies
            result.append(n)

    depth_first_search(node)
    return set(result)


# precompute full set of direct and indirect dependencies for each property
property_requirements = {
    k: get_property_dependencies(property_deps, k)
    for k in (CURRENT_PROPS_GPU | GLOBAL_PROPS)
}

# set of properties that require an intensity_image also be provided
need_intensity_image = set(intensity_deps.keys()) | {"image_intensity"}


def regionprops_dict(
    label_image,
    intensity_image=None,
    properties=[],
    *,
    spacing=None,
    max_label=None,
    pixels_per_thread=16,
):
    """Compute image properties and return them as a pandas-compatible table.

    The table is a dictionary mapping column names to value arrays. See Notes
    section below for details.

    .. versionadded:: 0.16

    Parameters
    ----------
    label_image : (M, N[, P]) ndarray
        Labeled input image. Labels with value 0 are ignored.
    intensity_image : (M, N[, P][, C]) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image, plus
        optionally an extra dimension for multichannel data. The channel
        dimension, if present, must be the last axis. Default is None.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    spacing : tuple of float, shape (ndim,)
        The pixel spacing along each axis of the image.

    Extra Parameters
    ----------------
    max_label : int or None
        The maximum label value. If not provided it will be computed from
        `label_image`.
    pixels_per_thread : int
        A number of properties support computation of multiple adjacent pixels
        from each GPU thread. The number of adjacent pixels processed
        corresponds to `pixels_per_thread` and can be used as a performance
        tuning parameter.
    """
    supported_properties = CURRENT_PROPS_GPU | GLOBAL_PROPS
    properties = set(properties)

    valid_names = properties & supported_properties
    invalid_names = set(properties) - valid_names
    valid_names = list(valid_names)

    # Use only the modern names internally, but keep list of mappings back to
    # any deprecated names in restore_legacy_names and use that at the end to
    # restore the requested deprecated property names.
    restore_legacy_names = dict()
    for name in invalid_names:
        if name in PROPS:
            vname = PROPS[name]
            if vname in valid_names:
                raise ValueError(
                    f"Property name: {name} is a duplicate of {vname}"
                )
            else:
                restore_legacy_names[vname] = name
                valid_names.append(vname)
        else:
            raise ValueError(f"Unrecognized property name: {name}")
    for v in restore_legacy_names.values():
        invalid_names.discard(v)
    # warn if there are any names that did not match a deprecated name
    if invalid_names:
        warnings.warn(
            "The following property names were unrecognized and will not be "
            "computed: {invalid_names}"
        )

    requested_props = set(sorted(valid_names))

    if len(requested_props) == 0:
        return {}

    required_props = set()
    for prop in requested_props:
        required_props.update(property_requirements[prop])

    ndim = label_image.ndim
    if ndim != 2:
        invalid_names = requested_props & ndim_2_only
        if any(invalid_names):
            raise ValueError(
                f"{label_image.ndim=}, but the following properties are for "
                "2D label images only: {invalid_names}"
            )
    if intensity_image is None:
        has_intensity = False
        invalid_names = requested_props & need_intensity_image
        if any(invalid_names):
            raise ValueError(
                "No intensity_image provided, but the following requested "
                "properties require one: {invalid_names}"
            )
    else:
        has_intensity = True

    out = {}
    if max_label is None:
        max_label = int(label_image.max())
    label_dtype = _get_min_integer_dtype(max_label, signed=False)
    # For performance, shrink label's data type to the minimum possible
    # unsigned integer type.
    if label_image.dtype != label_dtype:
        label_image = label_image.astype(label_dtype)

    # create vector of label values
    if "label" in required_props:
        out["label"] = cp.arange(1, max_label + 1, dtype=label_dtype)

    perf_kwargs = {}
    if pixels_per_thread is not None:
        perf_kwargs["pixels_per_thread"] = pixels_per_thread

    if "num_pixels" in required_props:
        regionprops_num_pixels(
            label_image,
            max_label=max_label,
            filled=False,
            **perf_kwargs,
            props_dict=out,
        )

    if "area" in required_props:
        regionprops_area(
            label_image,
            spacing=spacing,
            max_label=max_label,
            dtype=cp.float32,
            filled=False,
            **perf_kwargs,
            props_dict=out,
        )

        if "equivalent_diameter_area" in required_props:
            ed = equivalent_diameter_area(out["area"], ndim)
            out["equivalent_diameter_area"] = ed
            if "equivalent_spherical_perimeter" in required_props:
                out[
                    "equivalent_spherical_perimeter"
                ] = equivalent_spherical_perimeter(out["area"], ndim, ed)

    if has_intensity:
        if "intensity_std" in required_props:
            # std also computes mean
            regionprops_intensity_std(
                label_image,
                intensity_image,
                max_label=max_label,
                std_dtype=cp.float64,
                sample_std=False,
                **perf_kwargs,
                props_dict=out,
            )

        elif "intensity_mean" in required_props:
            regionprops_intensity_mean(
                label_image,
                intensity_image,
                max_label=max_label,
                mean_dtype=cp.float32,
                **perf_kwargs,
                props_dict=out,
            )

        compute_min = "intensity_min" in required_props
        compute_max = "intensity_max" in required_props
        if compute_min or compute_max:
            regionprops_intensity_min_max(
                label_image,
                intensity_image,
                max_label=max_label,
                compute_min=compute_min,
                compute_max=compute_max,
                **perf_kwargs,
                props_dict=out,
            )

    compute_bbox = "bbox" in required_props
    if compute_bbox:
        # compute bbox (and slice)
        regionprops_bbox_coords(
            label_image,
            max_label=max_label,
            return_slices="slice" in required_props,
            **perf_kwargs,
            props_dict=out,
        )

        if "area_bbox" in required_props:
            regionprops_area_bbox(
                out["bbox"],
                area_dtype=cp.float32,
                spacing=None,
                props_dict=out,
            )

        if "extent" in required_props:
            out["extent"] = out["area"] / out["area_bbox"]

    if "num_boundary_pixels" in required_props:
        regionprops_num_boundary_pixels(
            label_image,
            max_label=max_label,
            props_dict=out,
        )

    if "num_perimeter_pixels" in required_props:
        regionprops_num_perimeter_pixels(
            label_image,
            max_label=max_label,
            props_dict=out,
        )

    if "perimeter_on_border_ratio" in required_props:
        out["perimeter_on_border_ratio"] = (
            out["num_boundary_pixels"] / out["num_perimeter_pixels"]
        )

    compute_images = "image" in required_props
    compute_intensity_images = "image_intensity" in required_props
    compute_convex = "image_convex" in required_props
    if compute_intensity_images or compute_images or compute_convex:
        regionprops_image(
            label_image,
            intensity_image=intensity_image
            if compute_intensity_images
            else None,  # noqa: E501
            max_label=max_label,
            props_dict=out,
            compute_image=compute_images,
            compute_convex=compute_convex,
            offset_coordinates=True,
        )

    compute_coords = "coords" in required_props
    compute_coords_scaled = "coords_scaled" in required_props
    if compute_coords or compute_coords_scaled:
        regionprops_coords(
            label_image,
            max_label=max_label,
            spacing=spacing,
            compute_coords=compute_coords,
            compute_coords_scaled=compute_coords_scaled,
            props_dict=out,
        )

    if "label_filled" in required_props:
        regionprops_label_filled(
            label_image,
            max_label=max_label,
            props_dict=out,
        )
        if "area_filled" in required_props:
            # also handles "num_pixels_filled"
            out["area_filled"] = regionprops_area(
                out["label_filled"],
                max_label=max_label,
                filled=True,
                props_dict=out,
            )
        elif "num_pixels_filled" in required_props:
            regionprops_num_pixels(
                label_image,
                max_label=max_label,
                filled=True,
                **perf_kwargs,
                props_dict=out,
            )
        if "image_filled" in required_props:
            out["image_filled"], _, _ = regionprops_image(
                out["label_filled"],
                max_label=max_label,
                compute_image=True,
                compute_convex=False,
                props_dict=None,  # omit: using custom "image_filled" key
            )

    # If user had requested properties via their deprecated names, set the
    # canonical names for the computed properties to the corresponding
    # deprecated one.
    for k, v in restore_legacy_names.items():
        out[v] = out.pop(k)

    # only return the properties that were explicitly requested
    out = {k: out[k] for k in properties}

    return out
