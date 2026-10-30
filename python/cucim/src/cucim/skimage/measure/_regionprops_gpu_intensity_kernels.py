# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import cupy as cp

from ._regionprops_gpu_utils import (
    _check_intensity_image_shape,
    _get_count_dtype,
    _get_label_run_loop,
    _includes,
    _warp_includes,
)

__all__ = [
    "regionprops_intensity_mean",
    "regionprops_intensity_min_max",
    "regionprops_intensity_std",
]

# Store information on which other properties a given property depends on
# This information will be used by `regionprops_dict` to make sure that when
# a particular property is requested any dependent properties are computed
# first.
intensity_deps = dict()
intensity_deps["intensity_min"] = []
intensity_deps["intensity_max"] = []
intensity_deps["intensity_mean"] = ["num_pixels"]
intensity_deps["intensity_std"] = ["num_pixels"]


def _get_img_sums_code(
    c_sum_type,
    num_channels=1,
    compute_num_pixels=True,
    compute_sum=True,
    compute_sum_sq=False,
):
    """
    Notes
    -----
    Variables created (per run of a label):

        - num_pixels : the number of pixels in the run
        - img_sum{c}, img_sum_sq{c} : sums of the intensities of channel c
            and of their squares

    Output variables written to:

        - counts : shape (max_label,)
        - sums : shape (max_label, num_channels)
        - sumsqs : shape (max_label, num_channels)
    """
    source_pre = ""
    source_start = ""
    source_operation = ""
    source_flush = ""
    reduce = []
    if compute_num_pixels:
        reduce.append(("num_pixels", "sum"))
        source_pre += """
    uint32_t num_pixels;"""
        source_start += """
          num_pixels = 0;"""
        source_operation += """
          num_pixels += 1;"""
        source_flush += """
          atomicAdd(&counts[lab - 1], num_pixels);"""
    if compute_sum or compute_sum_sq:
        source_pre += f"""
    {c_sum_type} v;"""
    # op uses external variables:
    #     ii : index into labels array
    # flush uses external variables:
    #     lab : label value of the run
    nc = f"{num_channels}*" if num_channels > 1 else ""
    for c in range(num_channels):
        if compute_sum or compute_sum_sq:
            source_operation += f"""
          v = static_cast<{c_sum_type}>(img[{nc}ii + {c}]);"""
        if compute_sum:
            reduce.append((f"img_sum{c}", "sum"))
            source_pre += f"""
    {c_sum_type} img_sum{c};"""
            source_start += f"""
          img_sum{c} = 0;"""
            source_operation += f"""
          img_sum{c} += v;"""
            source_flush += f"""
          atomicAdd(&sums[{nc}(lab - 1) + {c}], img_sum{c});"""
        if compute_sum_sq:
            reduce.append((f"img_sum_sq{c}", "sum"))
            source_pre += f"""
    {c_sum_type} img_sum_sq{c};"""
            source_start += f"""
          img_sum_sq{c} = 0;"""
            source_operation += f"""
          img_sum_sq{c} += v * v;"""
            source_flush += f"""
          atomicAdd(&sumsqs[{nc}(lab - 1) + {c}], img_sum_sq{c});"""
    return source_pre, source_start, source_operation, source_flush, reduce


def _get_intensity_min_max_code(
    c_min_max_type,
    initial_min_val,
    initial_max_val,
    compute_min=True,
    compute_max=True,
    num_channels=1,
):
    """
    Notes
    -----
    Variables created (per run of a label):

        - min_val{c}, max_val{c} : minimum and maximum intensity of channel c

    Output variables written to:

        - minimums : shape (max_label, num_channels)
        - maximums : shape (max_label, num_channels)
    """
    c_type = c_min_max_type

    # Note: CuPy provides atomicMin and atomicMax for float and double in
    #       cupy/_core/include/atomics.cuh
    #       The integer variants are part of CUDA itself.
    #       The kernels call cucim_atomic_min and cucim_atomic_max (see
    #       _includes), because CuPy's double overloads round to float.

    source_pre = f"""
    {c_type} v_mm;"""
    source_start = ""
    source_operation = ""
    source_flush = ""
    reduce = []
    # op uses external variables:
    #     ii : index into labels array
    # flush uses external variables:
    #     lab : label value of the run
    nc = f"{num_channels}*" if num_channels > 1 else ""
    for c in range(num_channels):
        source_operation += f"""
          v_mm = static_cast<{c_type}>(img[{nc}ii + {c}]);"""
        if compute_min:
            reduce.append((f"min_val{c}", "min"))
            source_pre += f"""
    {c_type} min_val{c};"""
            source_start += f"""
          min_val{c} = {initial_min_val};"""
            source_operation += f"""
          min_val{c} = min(v_mm, min_val{c});"""
            source_flush += f"""
          cucim_atomic_min(&minimums[{nc}(lab - 1) + {c}], min_val{c});"""
        if compute_max:
            reduce.append((f"max_val{c}", "max"))
            source_pre += f"""
    {c_type} max_val{c};"""
            source_start += f"""
          max_val{c} = {initial_max_val};"""
            source_operation += f"""
          max_val{c} = max(v_mm, max_val{c});"""
            source_flush += f"""
          cucim_atomic_max(&maximums[{nc}(lab - 1) + {c}], max_val{c});"""
    return source_pre, source_start, source_operation, source_flush, reduce


@cp.memoize()
def _get_intensity_img_kernel_dtypes(image_dtype):
    """Determine CuPy dtype and C++ type for image sum operations."""
    image_dtype = cp.dtype(image_dtype)
    if image_dtype.kind == "f":
        # use double for accuracy of mean/std computations
        c_sum_type = "double"
        dtype = cp.float64
        # atomicMin, atomicMax support 32 and 64-bit float
        if image_dtype.itemsize > 4:
            min_max_dtype = cp.float64
            c_min_max_type = "double"
        else:
            min_max_dtype = cp.float32
            c_min_max_type = "float"
    elif image_dtype.kind in "bu":
        c_sum_type = "uint64_t"
        dtype = cp.uint64
        if image_dtype.itemsize > 4:
            min_max_dtype = cp.uint64
            c_min_max_type = "uint64_t"
        else:
            min_max_dtype = cp.uint32
            c_min_max_type = "uint32_t"
    elif image_dtype.kind in "i":
        c_sum_type = "int64_t"
        dtype = cp.int64
        if image_dtype.itemsize > 4:
            min_max_dtype = cp.int64
            c_min_max_type = "int64_t"
        else:
            min_max_dtype = cp.int32
            c_min_max_type = "int32_t"
    else:
        raise ValueError(
            f"Invalid intensity image dtype {image_dtype.name}. "
            "Must be an unsigned, integer or floating point type."
        )
    return cp.dtype(dtype), c_sum_type, cp.dtype(min_max_dtype), c_min_max_type


@cp.memoize()
def _get_intensity_range(image_dtype):
    """Determine the valid intensity range for the given image dtype."""
    image_dtype = cp.dtype(image_dtype)
    if image_dtype.kind == "f":
        # use double for accuracy of mean/std computations
        info = cp.finfo(image_dtype)
    elif image_dtype.kind in "bui":
        info = cp.iinfo(image_dtype)
    else:
        raise ValueError(
            f"Invalid intensity image dtype {image_dtype.name}. "
            "Must be an unsigned, integer or floating point type."
        )
    return (info.min, info.max)


@cp.memoize(for_each_device=True)
def get_intensity_measure_kernel(
    image_dtype=None,
    int32_count=True,
    num_channels=1,
    compute_num_pixels=True,
    compute_sum=True,
    compute_sum_sq=False,
    compute_min=False,
    compute_max=False,
    pixels_per_thread=8,
):
    if compute_num_pixels:
        count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)

    (
        sum_dtype,
        c_sum_type,
        min_max_dtype,
        c_min_max_type,
    ) = _get_intensity_img_kernel_dtypes(image_dtype)

    any_sums = compute_num_pixels or compute_sum or compute_sum_sq
    any_min_max = compute_min or compute_max
    if not (any_min_max or any_sums):
        raise ValueError("no output values requested")

    fragments = []
    if any_sums:
        fragments.append(
            _get_img_sums_code(
                c_sum_type=c_sum_type,
                num_channels=num_channels,
                compute_num_pixels=compute_num_pixels,
                compute_sum=compute_sum,
                compute_sum_sq=compute_sum_sq,
            )
        )
    if any_min_max:
        if min_max_dtype is None:
            raise ValueError("min_max_dtype must be specified")
        range_min, range_max = _get_intensity_range(min_max_dtype)
        fragments.append(
            _get_intensity_min_max_code(
                c_min_max_type=c_min_max_type,
                num_channels=num_channels,
                initial_max_val=range_min,
                initial_min_val=range_max,
                compute_min=compute_min,
                compute_max=compute_max,
            )
        )
    pre, start, op, flush = ("".join(f[i] for f in fragments) for i in range(4))
    reduce = [r for f in fragments for r in f[4]]

    # store only counts for label > 0  (label = 0 is the background)
    source = _get_label_run_loop(
        pixels_per_thread, pre, start, op, flush, warp_reduce=reduce
    )

    inputs = "raw X labels, raw uint64 labels_size, raw Y img"
    outputs = []
    name = "cucim_"
    if compute_num_pixels:
        outputs.append(f"raw {count_dtype.name} counts")
        name += f"_numpix_{count_dtype.char}"
    if compute_sum:
        outputs.append(f"raw {sum_dtype.name} sums")
        name += "_sum"
    if compute_sum_sq:
        outputs.append(f"raw {sum_dtype.name} sumsqs")
        name += "_sumsq"
    if compute_sum or compute_sum_sq:
        name += f"_{sum_dtype.char}"
    if compute_min:
        outputs.append(f"raw {min_max_dtype.name} minimums")
        name += "_min"
    if compute_max:
        outputs.append(f"raw {min_max_dtype.name} maximums")
        name += "_max"
    if compute_min or compute_max:
        name += f"{min_max_dtype.char}"
    outputs = ", ".join(outputs)
    name += f"_batch{pixels_per_thread}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_warp_includes, name=name
    )


def _regionprops_intensity(
    label_image,
    intensity_image,
    max_label=None,
    *,
    compute_mean=False,
    compute_std=False,
    compute_min=False,
    compute_max=False,
    mean_dtype=cp.float32,
    std_dtype=cp.float64,
    sample_std=False,
    pixels_per_thread=16,
    props_dict=None,
):
    """Compute the requested intensity statistics in one pass over the image.

    reuses "num_pixels" from `props_dict` if it exists

    writes "intensity_mean" to `props_dict` if `compute_mean` or `compute_std`
    writes "intensity_std" to `props_dict` if `compute_std`
    writes "intensity_min" to `props_dict` if `compute_min`
    writes "intensity_max" to `props_dict` if `compute_max`
    writes "num_pixels" to `props_dict` if it was not already present and the
    mean or standard deviation was computed

    When `compute_std` is True, the mean is returned with dtype `std_dtype`
    rather than `mean_dtype`.
    """
    if not (compute_mean or compute_std or compute_min or compute_max):
        raise ValueError("Nothing to compute")
    if props_dict is None:
        props_dict = {}
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_intensity_image_shape(label_image, intensity_image)

    image_dtype = intensity_image.dtype
    sum_dtype, _, min_max_dtype, _ = _get_intensity_img_kernel_dtypes(
        image_dtype
    )
    count_dtype, int32_count = _get_count_dtype(label_image.size)
    out_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )

    # the outputs are passed in the order of the kernel's parameters
    compute_sums = compute_mean or compute_std
    compute_num_pixels = False
    outputs = []
    if compute_sums:
        if "num_pixels" in props_dict:
            counts = props_dict["num_pixels"]
            if counts.dtype != count_dtype:
                counts = counts.astype(count_dtype, copy=False)
        else:
            counts = cp.zeros(num_counts, dtype=count_dtype)
            compute_num_pixels = True
            outputs.append(counts)
        sums = cp.zeros(out_shape, dtype=sum_dtype)
        outputs.append(sums)
        if compute_std:
            sumsqs = cp.zeros(out_shape, dtype=sum_dtype)
            outputs.append(sumsqs)
    if compute_min or compute_max:
        # use an appropriate data type supported by atomicMin and atomicMax
        range_min, range_max = _get_intensity_range(image_dtype)
        if compute_min:
            minimums = cp.full(out_shape, range_max, dtype=min_max_dtype)
            outputs.append(minimums)
        if compute_max:
            maximums = cp.full(out_shape, range_min, dtype=min_max_dtype)
            outputs.append(maximums)

    # TODO(grelee): May want to provide an approach with better numerical
    # stability (i.e.like the two-pass algorithm or Welford's online algorithm)
    kernel = get_intensity_measure_kernel(
        int32_count=int32_count,
        image_dtype=image_dtype,
        num_channels=num_channels,
        compute_num_pixels=compute_num_pixels,
        compute_sum=compute_sums,
        compute_sum_sq=compute_std,
        compute_min=compute_min,
        compute_max=compute_max,
        pixels_per_thread=pixels_per_thread,
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    kernel(
        label_image,
        label_image.size,
        intensity_image,
        *outputs,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

    if compute_std:
        if cp.dtype(std_dtype).kind != "f":
            raise ValueError("mean_dtype must be a floating point type")

        # compute means and standard deviations from the counts, sums and
        # squared sums (use float64 here since the numerical stability of this
        # approach is poor)
        means = cp.zeros(out_shape, dtype=cp.float64)
        stds = cp.zeros(out_shape, dtype=cp.float64)
        kernel2 = get_mean_var_kernel(stds.dtype, sample_std=sample_std)
        if num_channels > 1:
            kernel2(counts[..., cp.newaxis], sums, sumsqs, means, stds)
        else:
            kernel2(counts, sums, sumsqs, means, stds)

        means = means.astype(std_dtype, copy=False)
        stds = stds.astype(std_dtype, copy=False)
        props_dict["intensity_std"] = stds
        props_dict["intensity_mean"] = means
    elif compute_mean:
        if num_channels > 1:
            means = sums / counts[:, cp.newaxis]
        else:
            means = sums / counts
        means = means.astype(mean_dtype, copy=False)
        props_dict["intensity_mean"] = means
    if compute_sums and "num_pixels" not in props_dict:
        props_dict["num_pixels"] = counts
    if compute_min:
        props_dict["intensity_min"] = minimums
    if compute_max:
        props_dict["intensity_max"] = maximums
    return props_dict


def regionprops_intensity_mean(
    label_image,
    intensity_image,
    max_label=None,
    mean_dtype=cp.float32,
    pixels_per_thread=16,
    props_dict=None,
):
    """Compute the mean intensity of each region.

    reuses "num_pixels" from `props_dict` if it exists

    writes "intensity_mean" to `props_dict`
    writes "num_pixels" to `props_dict` if it was not already present
    """
    return _regionprops_intensity(
        label_image,
        intensity_image,
        max_label=max_label,
        compute_mean=True,
        mean_dtype=mean_dtype,
        pixels_per_thread=pixels_per_thread,
        props_dict=props_dict,
    )


@cp.memoize(for_each_device=True)
def get_mean_var_kernel(dtype, sample_std=False):
    dtype = cp.dtype(dtype)

    if dtype.kind != "f":
        raise ValueError("dtype must be a floating point type")
    if dtype == cp.float64:
        c_type = "double"
        nan_val = "CUDART_NAN"
    else:
        c_type = "float"
        nan_val = "CUDART_NAN_F"

    # The one-pass formula sumsq - m * m * count can round to a small negative
    # number when all values are equal (or nearly so), so the variance is
    # clipped at 0 before taking the square root (otherwise sqrt gives NaN).
    # The clip uses a comparison rather than max(), which is fmax in CUDA and
    # would also replace a NaN variance (from NaN or inf values) with 0.
    if sample_std:
        source = f"""
            if (count == 1) {{
              m = static_cast<{c_type}>(sum);
              var = {nan_val};
            }} else {{
              m = static_cast<double>(sum) / count;
              double v = (static_cast<double>(sumsq) - m * m * count)
                         / (count - 1);
              var = sqrt(v < 0.0 ? 0.0 : v);
            }}\n"""
    else:
        source = f"""
            if (count == 0) {{
              m = static_cast<{c_type}>(sum);
              var = {nan_val};
            }} else if (count == 1) {{
              m = static_cast<{c_type}>(sum);
              var = 0.0;
            }} else {{
              m = static_cast<double>(sum) / count;
              double v = (static_cast<double>(sumsq) - m * m * count) / count;
              var = sqrt(v < 0.0 ? 0.0 : v);
            }}\n"""
    inputs = "X count, Y sum, Y sumsq"
    outputs = "Z m, Z var"
    name = f"cucim_sample_std_naive_{dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_intensity_std(
    label_image,
    intensity_image,
    sample_std=False,
    max_label=None,
    std_dtype=cp.float64,
    pixels_per_thread=4,
    props_dict=None,
):
    """Compute the mean and standard deviation of the intensity of each region.

    reuses "num_pixels" from `props_dict` if it exists

    writes "intensity_mean" to `props_dict`
    writes "intensity_std" to `props_dict`
    writes "num_pixels" to `props_dict` if it was not already present
    """
    return _regionprops_intensity(
        label_image,
        intensity_image,
        max_label=max_label,
        compute_std=True,
        std_dtype=std_dtype,
        sample_std=sample_std,
        pixels_per_thread=pixels_per_thread,
        props_dict=props_dict,
    )


def regionprops_intensity_min_max(
    label_image,
    intensity_image,
    max_label=None,
    compute_min=True,
    compute_max=False,
    pixels_per_thread=8,
    props_dict=None,
):
    """Compute the minimum and maximum intensity of each region.

    writes "intensity_min" to `props_dict` if `compute_min` is True
    writes "intensity_max" to `props_dict` if `compute_max` is True
    """
    return _regionprops_intensity(
        label_image,
        intensity_image,
        max_label=max_label,
        compute_min=compute_min,
        compute_max=compute_max,
        pixels_per_thread=pixels_per_thread,
        props_dict=props_dict,
    )
