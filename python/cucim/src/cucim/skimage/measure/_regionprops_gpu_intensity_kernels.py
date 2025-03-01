import math

import cupy as cp

from ._regionprops_gpu_utils import (
    _check_intensity_image_shape,
    _get_count_dtype,
    _includes,
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
    pixels_per_thread,
    array_size,
    num_channels=1,
    compute_num_pixels=True,
    compute_sum=True,
    compute_sum_sq=False,
):
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

    source_pre = ""
    if compute_num_pixels:
        source_pre += f"""
    {pixel_count_dtype} num_pixels[{array_size}] = {{0}};"""
    if compute_sum:
        source_pre += f"""
    {c_sum_type} img_sums[{array_size * num_channels}] = {{0}};"""
    if compute_sum_sq:
        source_pre += f"""
    {c_sum_type} img_sum_sqs[{array_size * num_channels}] = {{0}};"""
    if compute_sum or compute_sum_sq:
        source_pre += f"""
    {c_sum_type} v = 0;\n"""

    # source_operation requires external variables:
    #     ii : index into labels array
    #     offset : index into local region's num_pixels array
    #              (number of unique labels encountered so far by this thread)
    source_operation = ""
    if compute_num_pixels:
        source_operation += """
            num_pixels[offset] += 1;"""
    nc = f"{num_channels}*" if num_channels > 1 else ""
    if compute_sum or compute_sum_sq:
        for c in range(num_channels):
            source_operation += f"""
            v = static_cast<{c_sum_type}>(img[{nc}ii + {c}]);"""
            if compute_sum:
                source_operation += f"""
            img_sums[{nc}offset + {c}] += v;"""
            if compute_sum_sq:
                source_operation += f"""
            img_sum_sqs[{nc}offset + {c}] += v * v;\n"""

    # post_operation requires external variables:
    #     jj : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     num_pixels : output with shape (max_label,)
    #     sums : output with shape (max_label, num_channels)
    #     sumsqs : output with shape (max_label, num_channels)
    source_post = ""
    if compute_num_pixels:
        source_post += """
            atomicAdd(&counts[lab - 1], num_pixels[jj]);"""
    if compute_sum:
        for c in range(num_channels):
            source_post += f"""
            atomicAdd(&sums[{nc}(lab - 1) + {c}], img_sums[{nc}jj + {c}]);"""
    if compute_sum_sq:
        for c in range(num_channels):
            source_post += f"""
            atomicAdd(&sumsqs[{nc}(lab - 1) + {c}], img_sum_sqs[{nc}jj + {c}]);"""  # noqa: E501
    return source_pre, source_operation, source_post


def _get_intensity_min_max_code(
    min_max_dtype,
    c_min_max_type,
    array_size,
    initial_min_val,
    initial_max_val,
    compute_min=True,
    compute_max=True,
    num_channels=1,
):
    min_max_dtype = cp.dtype(min_max_dtype)
    c_type = c_min_max_type

    # Note: CuPy provides atomicMin and atomicMax for float and double in
    #       cupy/_core/include/atomics.cuh
    #       The integer variants are part of CUDA itself.

    source_pre = ""
    if compute_min:
        source_pre += f"""
    {c_type} min_vals[{array_size * num_channels}];
    // initialize minimum coordinate to array size
    for (size_t ii = 0; ii < {array_size * num_channels}; ii++) {{
      min_vals[ii] = {initial_min_val};
    }}"""
    if compute_max:
        source_pre += f"""
    {c_type} max_vals[{array_size * num_channels}];
    // initialize minimum coordinate to array size
    for (size_t ii = 0; ii < {array_size * num_channels}; ii++) {{
      max_vals[ii] = {initial_max_val};
    }}"""
    source_pre += f"""
    {c_type} v = 0;\n"""

    # source_operation requires external variables:
    #     ii : index into labels array
    #     offset : index into local region's num_pixels array
    #              (number of unique labels encountered so far by this thread)
    source_operation = ""
    nc = f"{num_channels}*" if num_channels > 1 else ""
    if compute_min or compute_max:
        for c in range(num_channels):
            source_operation += f"""
            v = static_cast<{c_type}>(img[{nc}ii + {c}]);"""
            if compute_min:
                source_operation += f"""
            min_vals[{nc}offset + {c}] = min(v, min_vals[{nc}offset + {c}]);"""
            if compute_max:
                source_operation += f"""
            max_vals[{nc}offset + {c}] = max(v, max_vals[{nc}offset + {c}]);\n"""  # noqa: E501

    # post_operation requires external variables:
    #     jj : offset index into min_vals or max_vals array
    #     lab : label value that corresponds to location ii
    #     min_vals : output with shape (max_label, num_channels)
    #     max_vals : output with shape (max_label, num_channels)
    source_post = ""
    if compute_min:
        for c in range(num_channels):
            source_post += f"""
            atomicMin(&minimums[{nc}(lab - 1) + {c}], min_vals[{nc}jj + {c}]);"""  # noqa: E501
    if compute_max:
        for c in range(num_channels):
            source_post += f"""
            atomicMax(&maximums[{nc}(lab - 1) + {c}], max_vals[{nc}jj + {c}]);"""  # noqa: E501
    return source_pre, source_operation, source_post


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
    """Determine CuPy dtype and C++ type for image sum operations."""
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

    array_size = pixels_per_thread
    any_sums = compute_num_pixels or compute_sum or compute_sum_sq

    if any_sums:
        sums_pre, sums_op, sums_post = _get_img_sums_code(
            c_sum_type=c_sum_type,
            pixels_per_thread=pixels_per_thread,
            array_size=array_size,
            num_channels=num_channels,
            compute_num_pixels=compute_num_pixels,
            compute_sum=compute_sum,
            compute_sum_sq=compute_sum_sq,
        )

    any_min_max = compute_min or compute_max
    if any_min_max:
        if min_max_dtype is None:
            raise ValueError("min_max_dtype must be specified")
        range_min, range_max = _get_intensity_range(min_max_dtype)
        min_max_pre, min_max_op, min_max_post = _get_intensity_min_max_code(
            min_max_dtype=min_max_dtype,
            c_min_max_type=c_min_max_type,
            array_size=array_size,
            num_channels=num_channels,
            initial_max_val=range_min,
            initial_min_val=range_max,
            compute_min=compute_min,
            compute_max=compute_max,
        )

    if not (any_min_max or any_sums):
        raise ValueError("no output values requested")

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      uint64_t start_index = {pixels_per_thread}*i;
    """

    if any_sums:
        source += sums_pre
    if any_min_max:
        source += min_max_pre

    inner_op = ""
    if any_sums:
        inner_op += sums_op
    if any_min_max:
        inner_op += min_max_op

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
      for (size_t jj = 0; jj <= offset; jj++) {
        X lab = encountered_labels[jj];
        if (lab != 0) {"""

    if any_sums:
        source += sums_post
    if any_min_max:
        source += min_max_post
    source += """
        }
      }\n"""

    # print(source)
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
        inputs, outputs, source, preamble=_includes, name=name
    )


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
    if props_dict is None:
        props_dict = {}
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_intensity_image_shape(label_image, intensity_image)

    count_dtype, int32_count = _get_count_dtype(label_image.size)

    image_dtype = intensity_image.dtype
    sum_dtype, _, _, _ = _get_intensity_img_kernel_dtypes(image_dtype)

    if "num_pixels" in props_dict:
        counts = props_dict["num_pixels"]
        if counts.dtype != count_dtype:
            counts = counts.astype(count_dtype, copy=False)
        compute_num_pixels = False
    else:
        counts = cp.zeros(num_counts, dtype=count_dtype)
        compute_num_pixels = True

    sum_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    sums = cp.zeros(sum_shape, dtype=sum_dtype)

    kernel = get_intensity_measure_kernel(
        int32_count=int32_count,
        image_dtype=image_dtype,
        num_channels=num_channels,
        compute_num_pixels=compute_num_pixels,
        compute_sum=True,
        compute_sum_sq=False,
        pixels_per_thread=pixels_per_thread,
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    if compute_num_pixels:
        outputs = (counts, sums)
    else:
        outputs = (sums,)

    kernel(
        label_image,
        label_image.size,
        intensity_image,
        *outputs,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

    if num_channels > 1:
        means = sums / counts[:, cp.newaxis]
    else:
        means = sums / counts
    means = means.astype(mean_dtype, copy=False)
    props_dict["intensity_mean"] = means
    if "num_pixels" not in props_dict:
        props_dict["num_pixels"] = counts
    return props_dict


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

    if sample_std:
        source = f"""
            if (count == 1) {{
              m = static_cast<{c_type}>(sum);
              var = {nan_val};
            }} else {{
              m = static_cast<double>(sum) / count;
              var = sqrt(
                  (static_cast<double>(sumsq) - m * m * count) / (count - 1));
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
              var = sqrt(
                  (static_cast<double>(sumsq) - m * m * count) / count);
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
    if props_dict is None:
        props_dict = {}
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_intensity_image_shape(label_image, intensity_image)

    image_dtype = intensity_image.dtype
    sum_dtype, _, _, _ = _get_intensity_img_kernel_dtypes(image_dtype)

    count_dtype, int32_count = _get_count_dtype(label_image.size)

    if "num_pixels" in props_dict:
        counts = props_dict["num_pixels"]
        if counts.dtype != count_dtype:
            counts = counts.astype(count_dtype, copy=False)
        compute_num_pixels = False
    else:
        counts = cp.zeros(num_counts, dtype=count_dtype)
        compute_num_pixels = True

    sum_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    sums = cp.zeros(sum_shape, dtype=sum_dtype)
    sumsqs = cp.zeros(sum_shape, dtype=sum_dtype)

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    # TODO(grelee): May want to provide an approach with better numerical
    # stability (i.e.like the two-pass algorithm or Welford's online algorithm)
    kernel = get_intensity_measure_kernel(
        int32_count=int32_count,
        image_dtype=image_dtype,
        num_channels=num_channels,
        compute_num_pixels=compute_num_pixels,
        compute_sum=True,
        compute_sum_sq=True,
        pixels_per_thread=pixels_per_thread,
    )
    if compute_num_pixels:
        outputs = (counts, sums, sumsqs)
    else:
        outputs = (sums, sumsqs)
    kernel(
        label_image,
        label_image.size,
        intensity_image,
        *outputs,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

    if cp.dtype(std_dtype).kind != "f":
        raise ValueError("mean_dtype must be a floating point type")

    # compute means and standard deviations from the counts, sums and
    # squared sums (use float64 here since the numerical stability of this
    # approach is poor)
    means = cp.zeros(sum_shape, dtype=cp.float64)
    stds = cp.zeros(sum_shape, dtype=cp.float64)
    kernel2 = get_mean_var_kernel(stds.dtype, sample_std=sample_std)
    if num_channels > 1:
        kernel2(counts[..., cp.newaxis], sums, sumsqs, means, stds)
    else:
        kernel2(counts, sums, sumsqs, means, stds)

    means = means.astype(std_dtype, copy=False)
    stds = stds.astype(std_dtype, copy=False)
    props_dict["intensity_std"] = stds
    props_dict["intensity_mean"] = means
    if "num_pixels" not in props_dict:
        props_dict["num_pixels"] = counts
    return props_dict


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
    if not (compute_min or compute_max):
        raise ValueError("Nothing to compute")
    if props_dict is None:
        props_dict = {}

    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_intensity_image_shape(label_image, intensity_image)

    # use an appropriate data type supported by atomicMin and atomicMax
    image_dtype = intensity_image.dtype
    _, _, min_max_dtype, _ = _get_intensity_img_kernel_dtypes(image_dtype)
    range_min, range_max = _get_intensity_range(image_dtype)
    out_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    if compute_min:
        minimums = cp.full(out_shape, range_max, dtype=min_max_dtype)
    if compute_max:
        maximums = cp.full(out_shape, range_min, dtype=min_max_dtype)

    kernel = get_intensity_measure_kernel(
        image_dtype=image_dtype,
        num_channels=num_channels,
        compute_num_pixels=False,
        compute_sum=False,
        compute_sum_sq=False,
        compute_min=compute_min,
        compute_max=compute_max,
        pixels_per_thread=pixels_per_thread,
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    lab_size = label_image.size
    sz = math.ceil(label_image.size / pixels_per_thread)
    if compute_min and compute_max:
        outputs = (minimums, maximums)
    elif compute_min:
        outputs = (minimums,)
    else:
        outputs = (maximums,)

    kernel(
        label_image, lab_size, intensity_image, *outputs, size=sz
    )  # noqa: E501
    if compute_min:
        props_dict["intensity_min"] = minimums
    if compute_max:
        props_dict["intensity_max"] = maximums
    return props_dict
