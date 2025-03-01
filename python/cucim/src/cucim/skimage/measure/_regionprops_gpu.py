import warnings
from collections.abc import Callable
from copy import copy

import cupy as cp
import numpy as np

from cucim.skimage.measure._regionprops import (
    COL_DTYPES,
    PROPS,
    _infer_number_of_required_args,
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
from ._regionprops_gpu_convex import (
    convex_deps,
    regionprops_area_convex,
    regionprops_feret_diameter_max,
)
from ._regionprops_gpu_intensity_kernels import (
    intensity_deps,
    regionprops_intensity_mean,
    regionprops_intensity_min_max,
    regionprops_intensity_std,
)
from ._regionprops_gpu_misc_kernels import (
    misc_deps,
    regionprops_euler,
    regionprops_perimeter,
    regionprops_perimeter_crofton,
)
from ._regionprops_gpu_moments_kernels import (
    moment_deps,
    regionprops_centroid,
    regionprops_centroid_local,
    regionprops_centroid_weighted,
    regionprops_inertia_tensor,
    regionprops_inertia_tensor_eigvals,
    regionprops_moments,
    regionprops_moments_central,
    regionprops_moments_hu,
    regionprops_moments_normalized,
    required_order,
)
from ._regionprops_gpu_utils import _find_close_labels, _get_min_integer_dtype

__all__ = [
    "equivalent_diameter_area",
    "regionprops_area",
    "regionprops_area_bbox",
    "regionprops_area_convex",
    "regionprops_bbox_coords",
    "regionprops_centroid",
    "regionprops_centroid_local",
    "regionprops_centroid_weighted",
    "regionprops_coords",
    "regionprops_dict",
    "regionprops_euler",
    "regionprops_extent",
    "regionprops_feret_diameter_max",
    "regionprops_image",
    "regionprops_inertia_tensor",
    "regionprops_inertia_tensor_eigvals",
    "regionprops_intensity_mean",
    "regionprops_intensity_min_max",
    "regionprops_intensity_std",
    "regionprops_moments",
    "regionprops_moments_central",
    "regionprops_moments_hu",
    "regionprops_moments_normalized",
    "regionprops_num_pixels",
    "regionprops_perimeter",
    "regionprops_perimeter_crofton",
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
    "axis_lengths": "axis_lengths",
    "inertia_tensor_eigenvectors": "inertia_tensor_eigenvectors",
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
    "axis_lengths": float,
    "inertia_tensor_eigenvectors": float,
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
property_deps.update(convex_deps)
property_deps.update(intensity_deps)
property_deps.update(misc_deps)
property_deps.update(moment_deps)


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
need_intensity_image = (
    set(intensity_deps.keys())
    | {"image_intensity"}
    | set(p for p in CURRENT_PROPS_GPU if "weighted" in p)
)

# set of properties that can only be computed for 2D regions
ndim_2_only = {
    "eccentricity",
    "moments_hu",
    "moments_weighted_hu",
    "orientation",
    "perimeter",
    "perimeter_crofton",  # could be updated to nD as in ITK
}


def _check_moment_order(moment_order: int | None, requested_moment_props: set):
    """Helper function for input validation in `regionprops_dict`.

    Determines the minimum order required across all requested moment
    properties and validates the `moment_order` kwarg.
    """
    min_order_required = max(required_order[p] for p in requested_moment_props)
    if moment_order is not None:
        if moment_order < min_order_required:
            raise ValueError(
                f"can't compute {requested_moment_props} with moment_order<"
                f"{min_order_required}, but {moment_order=} was specified."
            )
        order = moment_order
    else:
        order = min_order_required
    return order


def regionprops_dict(
    label_image,
    intensity_image=None,
    properties=[],
    *,
    spacing=None,
    extra_properties=None,
    moment_order=None,
    max_label=None,
    pixels_per_thread=16,
    robust_perimeter=True,
    to_table=False,
    table_separator="-",
    table_on_host=False,
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
    spacing : tuple of float, shape (ndim,), optional
        The pixel spacing along each axis of the image.
    extra_properties : Iterable of callables
        Add extra property computation functions that are not included with
        skimage. The name of the property is derived from the function name,
        the dtype is inferred by calling the function on a small sample.
        If the name of an extra property clashes with the name of an existing
        property, the extra property will not be visible and a `UserWarning` is
        issued. A property computation function must take a region mask as its
        first argument. If the property requires an intensity image, it must
        accept the intensity image as the second argument.

    Extra Parameters
    ----------------
    moment_order : int or None, optional
        When computing moment properties, only moments up to this order are
        computed. The default value of None results in the minimum order
        required in order to compute the requested properties. For example,
        properties based on the inertia_tensor require moment_order >= 2.
    max_label : int or None, optional
        The maximum label value. If not provided it will be computed from
        `label_image`.
    pixels_per_thread : int, optional
        A number of properties support computation of multiple adjacent pixels
        from each GPU thread. The number of adjacent pixels processed
        corresponds to `pixels_per_thread` and can be used as a performance
        tuning parameter.
    robust_perimeter : bool, optional
        Batch computation of perimeter and euler characteristics can give
        incorrect results for perimeter pixels that are not more than 1 pixel
        spacing from another label. If True, a check for this condition is
        performed and any labels close to another label have their perimeter
        recomputed in isolation. Doing this check results in performance
        overhead so can optionally be disabled. This parameter effects the
        following regionprops: {"perimeter", "perimeter_crofton",
        "euler_number"}.
    to_table : bool, optional
        If true, split up vector/matrix properties into separate keys for
        the individual elements to match the output format of
        `regionprops_table` from scikit-image.
    table_separator : str, optional
        Separator character to use during conversion with `to_table`. Unused
        if `to_table` is false.
    table_on_host : bool, optional
        Copy any device arrays back to the host when creating the
        `regionprops_table` output. Unused if `to_table` is false.
    """
    supported_properties = CURRENT_PROPS_GPU | GLOBAL_PROPS
    properties = set(properties)
    if extra_properties is None:
        extra_properties = []

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
            f"computed: {invalid_names}"
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
                f"2D label images only: {invalid_names}"
            )
    if intensity_image is None:
        has_intensity = False
        invalid_names = requested_props & need_intensity_image
        if any(invalid_names):
            raise ValueError(
                "No intensity_image provided, but the following requested "
                f"properties require one: {invalid_names}"
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

    compute_unweighted_moments = "moments" in required_props
    compute_weighted_moments = "moments_weighted" in required_props
    compute_moments = compute_unweighted_moments or compute_weighted_moments
    compute_inertia_tensor = "inertia_tensor" in required_props

    if compute_moments:
        required_moment_props = set(moment_deps.keys()) & required_props
        # determine minimum necessary order (or validate the user-provided one)
        order = _check_moment_order(moment_order, required_moment_props)

        imgs = []
        if compute_unweighted_moments:
            imgs.append(None)
        if compute_weighted_moments:
            imgs.append(intensity_image)

        # compute raw moments (weighted and/or unweighted)
        for img in imgs:
            regionprops_moments(
                label_image,
                intensity_image=img,
                max_label=max_label,
                order=order,
                spacing=spacing,
                **perf_kwargs,
                props_dict=out,
            )

        compute_centroid_local = (
            "centroid_local" in required_moment_props
        )  # noqa:E501
        compute_centroid = "centroid" in required_moment_props
        if compute_centroid or compute_centroid_local:
            regionprops_centroid_weighted(
                moments_raw=out["moments"],
                ndim=label_image.ndim,
                bbox=out["bbox"],
                compute_local=compute_centroid_local,
                compute_global=compute_centroid,
                weighted=False,
                props_dict=out,
            )

        compute_centroid_weighted_local = (
            "centroid_weighted_local" in required_moment_props
        )
        compute_centroid_weighted = "centroid_weighted" in required_moment_props
        if compute_centroid_weighted or compute_centroid_weighted_local:
            regionprops_centroid_weighted(
                moments_raw=out["moments_weighted"],
                ndim=label_image.ndim,
                bbox=out["bbox"],
                compute_local=compute_centroid_weighted_local,
                compute_global=compute_centroid_weighted,
                weighted=True,
                props_dict=out,
            )

        if "moments_central" in required_moment_props:
            regionprops_moments_central(
                out["moments"], ndim=ndim, weighted=False, props_dict=out
            )

            if "moments_normalized" in required_moment_props:
                regionprops_moments_normalized(
                    out["moments_central"],
                    ndim=ndim,
                    spacing=None,
                    pixel_correction=False,
                    weighted=False,
                    props_dict=out,
                )
                if "moments_hu" in required_moment_props:
                    regionprops_moments_hu(
                        out["moments_normalized"],
                        weighted=False,
                        props_dict=out,
                    )

        if "moments_weighted_central" in required_moment_props:
            regionprops_moments_central(
                out["moments_weighted"], ndim, weighted=True, props_dict=out
            )

            if "moments_weighted_normalized" in required_moment_props:
                regionprops_moments_normalized(
                    out["moments_weighted_central"],
                    ndim=ndim,
                    spacing=None,
                    pixel_correction=False,
                    weighted=True,
                    props_dict=out,
                )

                if "moments_weighted_hu" in required_moment_props:
                    regionprops_moments_hu(
                        out["moments_weighted_normalized"],
                        weighted=True,
                        props_dict=out,
                    )

        # inertia tensor computations come after moment computations
        if compute_inertia_tensor:
            regionprops_inertia_tensor(
                out["moments_central"],
                ndim=ndim,
                compute_orientation=("orientation" in required_moment_props),
                props_dict=out,
            )

            if "inertia_tensor_eigvals" in required_moment_props:
                compute_axis_lengths = (
                    "axis_minor_length" in required_moment_props
                    or "axis_major_length" in required_moment_props
                )
                regionprops_inertia_tensor_eigvals(
                    out["inertia_tensor"],
                    compute_axis_lengths=compute_axis_lengths,
                    compute_eccentricity=(
                        "eccentricity" in required_moment_props
                    ),
                    compute_eigenvectors=(
                        "inertia_tensor_eigenvectors" in required_moment_props
                    ),
                    props_dict=out,
                )

    compute_perimeter = "perimeter" in required_props
    compute_perimeter_crofton = "perimeter_crofton" in required_props
    compute_euler = "euler_number" in required_props

    if compute_euler or compute_perimeter or compute_perimeter_crofton:
        # precompute list of labels with <2 pixels space between them
        if label_image.dtype == cp.uint8:
            labels_mask = label_image.view("bool")
        else:
            labels_mask = label_image > 0
        if robust_perimeter:
            # avoid repeatedly computing "labels_close" for
            # perimeter, perimeter_crofton and euler_number regionprops
            labels_close = _find_close_labels(
                label_image, binary_image=labels_mask, max_label=max_label
            )
            if labels_close.size > 0:
                print(
                    f"Found {labels_close.size} regions with <=1 background "
                    "pixel spacing from another region. Using slower robust "
                    "perimeter/euler measurements for these regions."
                )
        else:
            labels_close = None

        if compute_perimeter:
            regionprops_perimeter(
                label_image,
                neighborhood=4,
                max_label=max_label,
                robust=robust_perimeter,
                labels_close=labels_close,
                props_dict=out,
            )
        if compute_perimeter_crofton:
            regionprops_perimeter_crofton(
                label_image,
                directions=4,
                max_label=max_label,
                robust=robust_perimeter,
                omit_image_edges=False,
                labels_close=labels_close,
                props_dict=out,
            )

        if compute_euler:
            regionprops_euler(
                label_image,
                connectivity=None,
                max_label=max_label,
                robust=robust_perimeter,
                labels_close=labels_close,
                props_dict=out,
            )

    compute_images = ("image" in required_props) or (len(extra_properties) > 0)
    compute_intensity_images = ("image_intensity" in required_props) or (
        (intensity_image is not None) and (len(extra_properties) > 0)
    )
    compute_convex = "image_convex" in required_props
    if compute_intensity_images or compute_images or compute_convex:
        regionprops_image(
            label_image,
            intensity_image=(
                intensity_image if compute_intensity_images else None
            ),
            max_label=max_label,
            props_dict=out,
            compute_image=compute_images,
            compute_convex=compute_convex,
            offset_coordinates=True,
        )

    if "area_convex" in required_props:
        regionprops_area_convex(
            out["image_convex"], max_label=max_label, props_dict=out
        )

    if "solidity" in required_props:
        out["solidity"] = out["area"] / out["area_convex"]

    if "feret_diameter_max" in required_props:
        regionprops_feret_diameter_max(
            out["image_convex"],
            spacing=spacing,
            props_dict=out,
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

    # allow extra properties to be a list of callable functions
    # logic is set to match that in regionprops/regionprops_table
    for func in extra_properties:
        if not isinstance(func, Callable):
            raise ValueError(
                "extra_properties must be a list of callable functions, got "
                f"{type(func)}"
            )
        name = func.__name__
        n_args = _infer_number_of_required_args(func)
        images = out["image"]
        # determine whether func requires intensity image
        results = []
        if n_args == 2:
            if intensity_image is not None:
                images_intensity = out["image_intensity"]
                for image, image_intensity in zip(images, images_intensity):
                    multichannel = label_image.shape < intensity_image.shape
                    multichannel = label_image.shape < intensity_image.shape
                    if multichannel:
                        multichannel_list = [
                            func(images, images_intensity[..., i])
                            for i in range(images_intensity.shape[-1])
                        ]
                        results.append(cp.stack(multichannel_list, axis=-1))
                    else:
                        results.append(func(image, image_intensity))
            else:
                raise AttributeError(
                    f"intensity image required to calculate {name}"
                )
        elif n_args == 1:
            for image in images:
                results.append(func(image))
        else:
            raise AttributeError(
                f"Custom regionprop function's number of arguments must "
                f"be 1 or 2, but {name} takes {n_args} arguments."
            )
        is_cupy_array = isinstance(results[0], cp.ndarray)
        if is_cupy_array:
            out[name] = cp.stack(results, axis=0)
        else:
            out[name] = results

    # retain only the properties that were explicitly requested by the user
    out_properties = list(properties) + list(
        func.__name__ for func in extra_properties
    )
    out = {k: out[k] for k in out_properties}

    if to_table:
        out = _props_dict_to_table(
            out,
            list(out.keys()),
            separator=table_separator,
            copy_to_host=table_on_host,
            extra_property_names=tuple(f.__name__ for f in extra_properties),
        )
    return out


def _props_dict_to_table(
    props_dict,
    properties,
    extra_property_names=[],
    separator="-",
    copy_to_host=False,
):
    out = {}
    for prop in properties:
        # Copy the original property name so the output will have the
        # user-provided property name in the case of deprecated names.
        orig_prop = prop
        # determine the current property name for any deprecated property.
        prop = PROPS_GPU.get(prop, prop)
        # is_0dim_array = isinstance(rp, cp.ndarray) and rp.ndim == 0
        rp = props_dict[orig_prop]
        if prop in extra_property_names:
            dtype = np.object_
            if isinstance(rp, cp.ndarray):
                if rp.dtype.kind in "bui":
                    dtype = int
                else:
                    dtype = float
            else:
                dtype = np.object_
        else:
            # TODO: also update for GPU-only properties?
            dtype = COL_DTYPES_GPU[prop]

        is_scalar_prop = False
        is_multicolumn = False
        if isinstance(rp, cp.ndarray):
            is_scalar_prop = rp.ndim == 1
            is_multicolumn = not is_scalar_prop
        if is_scalar_prop:
            if copy_to_host:
                rp = cp.asnumpy(rp)
            out[orig_prop] = rp
        elif is_multicolumn:
            if copy_to_host:
                rp = cp.asnumpy(rp)
            shape = rp.shape[1:]
            # precompute property column names and locations
            modified_props = []
            locs = []
            for ind in np.ndindex(shape):
                modified_props.append(
                    separator.join(map(str, (orig_prop,) + ind))
                )
                locs.append((slice(None),) + ind)
            for i, modified_prop in enumerate(modified_props):
                out[modified_prop] = rp[locs[i]]
        elif prop in OBJECT_COLUMNS_GPU or prop in extra_property_names:
            n = len(rp)
            # keep objects in a NumPy array
            column_buffer = np.empty(n, dtype=dtype)
            if copy_to_host:
                for i in range(n):
                    column_buffer[i] = cp.asnumpy(rp[i])
                out[orig_prop] = column_buffer
            else:
                for i in range(n):
                    column_buffer[i] = rp[i]
                out[orig_prop] = np.copy(column_buffer)
        else:
            warnings.warn(
                f"Type unknown for property: {prop}, storing it as-is."
            )
            out[orig_prop] = rp
    return out
