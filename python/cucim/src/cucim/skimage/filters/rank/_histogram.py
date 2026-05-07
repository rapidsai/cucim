# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import cupy as cp

from ..._shared.utils import _to_np_mode
from ..._vendored import pad

_HISTOGRAM_OPS = {
    "percentile": 0,
    "threshold": 1,
    "mean": 2,
    "sum": 3,
    "pop": 4,
    "gradient": 5,
    "autolevel": 6,
    "entropy": 7,
    "enhance_contrast": 8,
    "subtract_mean": 9,
    "equalize": 10,
    "bilateral_mean": 11,
    "bilateral_pop": 12,
    "bilateral_sum": 13,
}

_HISTOGRAM_MIN_FOOTPRINT_AREA = {
    "sum": 15 * 15,
    "enhance_contrast": 17 * 17,
    "gradient": 17 * 17,
    "percentile": 17 * 17,
    "pop": 17 * 17,
    "threshold": 17 * 17,
    "mean": 19 * 19,
    "subtract_mean": 19 * 19,
    "autolevel": 21 * 21,
    "entropy": 25 * 25,
    "bilateral_sum": 27 * 27,
    "bilateral_pop": 29 * 29,
    "bilateral_mean": 33 * 33,
    "equalize": 91 * 91,
}

_DEFAULT_SCRATCH_MB = 256
_DEFAULT_MAX_PARTITIONS = 256
_INT16_MAX = 32767
_HISTOGRAM_COUNTER_TYPES = {
    "int16": (cp.int16, "short"),
    "int32": (cp.int32, "int"),
}
_HISTOGRAM_OUTPUT_TYPES = {
    "float32": (cp.float32, "float"),
    "float64": (cp.float64, "double"),
    "uint8": (cp.uint8, "unsigned char"),
}


def _can_use_rank_histogram(
    image,
    footprint_shape,
    output,
    mask,
    modes,
    origins,
    *,
    has_weights,
    operation,
    p0,
    p1,
):
    """Return True for the restricted uint8 2D histogram backend.

    This backend is intentionally narrow. It is selected only for supported
    rank operations on 2D uint8 images with an all-ones odd rectangular
    footprint, no mask, reflect mode and zero origin. Unsupported cases fall
    back to the generic rank implementation.
    """
    if operation not in _HISTOGRAM_OPS:
        return False
    if (
        operation
        in {
            "autolevel",
            "enhance_contrast",
            "mean",
            "subtract_mean",
            "sum",
            "pop",
            "gradient",
        }
        and p0 <= 0
        and p1 >= 100
    ):
        return False
    if image.ndim != 2 or image.dtype != cp.uint8:
        return False
    if output is not None and output.dtype != cp.uint8:
        if (
            operation != "entropy"
            or output.dtype.kind != "f"
            or output.dtype.name not in _HISTOGRAM_OUTPUT_TYPES
        ):
            return False
    if operation == "entropy" and output is None:
        return False
    if (
        operation != "entropy"
        and output is not None
        and output.dtype != cp.uint8
    ):
        return False
    if mask is not None or has_weights:
        return False
    if tuple(modes) != ("reflect", "reflect"):
        return False
    if any(origin != 0 for origin in origins):
        return False
    if len(footprint_shape) != 2:
        return False
    if any(size <= 1 or size % 2 == 0 for size in footprint_shape):
        return False
    radii = tuple(size // 2 for size in footprint_shape)
    if any(radius > size for radius, size in zip(radii, image.shape)):
        return False
    return True


def _should_use_rank_histogram(operation, footprint_shape):
    """Return True when benchmarks favor histogram over elementwise."""
    min_area = _HISTOGRAM_MIN_FOOTPRINT_AREA.get(operation)
    if min_area is None:
        return False
    return footprint_shape[0] * footprint_shape[1] >= min_area


def _get_env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _get_histogram_counter_dtype(footprint_shape):
    """Return the narrowest safe column-histogram counter dtype."""
    footprint_area = footprint_shape[0] * footprint_shape[1]
    if footprint_area <= _INT16_MAX:
        return cp.int16
    return cp.int32


def _get_rank_histogram_partitions(
    out_rows, cols, partitions=None, *, counter_dtype=cp.int32
):
    """Choose row partitions for the sliding-histogram backend.

    More partitions expose more row-band parallelism, but scratch memory grows
    linearly as ``partitions * cols * 256 * sizeof(counter_dtype)``.
    """
    if partitions is not None:
        return min(max(1, int(partitions)), out_rows)

    partitions = os.environ.get("CUCIM_RANK_HISTOGRAM_PARTITIONS")
    if partitions not in (None, ""):
        return min(max(1, int(partitions)), out_rows)

    scratch_mb = _get_env_int(
        "CUCIM_RANK_HISTOGRAM_SCRATCH_MB", _DEFAULT_SCRATCH_MB
    )
    max_partitions = _get_env_int(
        "CUCIM_RANK_HISTOGRAM_MAX_PARTITIONS", _DEFAULT_MAX_PARTITIONS
    )
    bytes_per_partition = cols * 256 * cp.dtype(counter_dtype).itemsize
    partitions_by_memory = max(1, (scratch_mb << 20) // bytes_per_partition)

    return min(max(1, out_rows // 2), max_partitions, partitions_by_memory)


@cp.memoize(for_each_device=True)
def _get_histogram_rank_kernel(
    operation, counter_dtype_name, output_dtype_name
):
    kernel_directory = os.path.join(os.path.dirname(__file__), "cuda")
    with open(os.path.join(kernel_directory, "histogram_rank.cu")) as f:
        code = "\n".join(f.readlines())

    _, counter_type = _HISTOGRAM_COUNTER_TYPES[counter_dtype_name]
    _, output_type = _HISTOGRAM_OUTPUT_TYPES[output_dtype_name]
    code = (
        f"#define RANK_HIST_OP {_HISTOGRAM_OPS[operation]}\n"
        f"#define HIST_COUNTER_T {counter_type}\n"
        f"#define RANK_HIST_OUTPUT_T {output_type}\n" + code
    )
    return cp.RawKernel(code=code, name="cuRankHistogram2DUint8")


def _rank_histogram(
    image,
    footprint_shape,
    operation,
    *,
    output=None,
    mode="reflect",
    cval=0,
    p0=0,
    p1=100,
    s0=0,
    s1=0,
    partitions=None,
):
    """Apply a uint8 2D rectangular rank filter using a sliding histogram."""
    image = cp.ascontiguousarray(image)
    radii = tuple(size // 2 for size in footprint_shape)
    npad = tuple((radius, radius) for radius in radii)
    np_mode = _to_np_mode(mode)
    if np_mode == "constant":
        pad_kwargs = dict(mode=np_mode, constant_values=cval)
    else:
        pad_kwargs = dict(mode=np_mode)
    padded = pad(image, npad, **pad_kwargs)

    out_dtype = output.dtype if output is not None else padded.dtype
    out = cp.empty(padded.shape, dtype=out_dtype)
    rows, cols = padded.shape
    out_rows = image.shape[0]
    counter_dtype = _get_histogram_counter_dtype(footprint_shape)
    counter_dtype_name = cp.dtype(counter_dtype).name
    output_dtype_name = cp.dtype(out.dtype).name
    partitions = _get_rank_histogram_partitions(
        out_rows, cols, partitions=partitions, counter_dtype=counter_dtype
    )

    hist = cp.zeros((partitions * cols * 256,), dtype=counter_dtype)
    op_code = _HISTOGRAM_OPS[operation]
    kernel = _get_histogram_rank_kernel(
        operation, counter_dtype_name, output_dtype_name
    )
    window_size = footprint_shape[0] * footprint_shape[1]
    kernel(
        (partitions,),
        (256,),
        (
            padded,
            out,
            hist,
            radii[0],
            radii[1],
            float(p0),
            float(p1),
            float(s0),
            float(s1),
            op_code,
            window_size,
            rows,
            cols,
        ),
    )

    out_sl = tuple(slice(radius, -radius) for radius in radii)
    result = out[out_sl]
    if output is not None:
        output[...] = result
        return output
    return result
