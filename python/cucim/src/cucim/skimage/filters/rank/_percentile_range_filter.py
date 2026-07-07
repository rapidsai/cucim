# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math

import cupy as cp
import numpy as np

import cucim.skimage._vendored._ndimage_filters_core as _filters_core
import cucim.skimage._vendored._ndimage_util as _util
from cucim.skimage._vendored._ndimage_filters import (
    __SHELL_SORT,
    _get_shell_gap,
)

from ._histogram import (
    _can_use_rank_histogram,
    _rank_histogram,
    _should_use_rank_histogram,
)


def _get_streaming_rank_kernel(
    p0,
    p1,
    operation,
    modes,
    w_shape,
    offsets,
    cval,
    int_type,
    has_weights,
    *,
    has_mask=False,
    dtype_max=255,
    s0=0.0,
    s1=0.0,
):
    """Generate a rank kernel that reduces during neighborhood traversal.

    This path is for operations that do not need sorted neighborhood values.
    It avoids allocating ``values`` and calling ``sort`` in the generated
    kernel, which is much cheaper for large footprints.
    """
    if operation in ("minimum", "maximum"):
        best_var = "min_val" if operation == "minimum" else "max_val"
        comparator = "<" if operation == "minimum" else ">"
        pre = f"int n_vals = 0;\nX {best_var};"
        update = f"""
            X v = {{value}};
            if (n_vals == 0 || v {comparator} {best_var}) {{
                {best_var} = v;
            }}
            n_vals++;
        """
        post = f"""
            if (n_vals == 0) {{
                y = cast<Y>(x[i]);
                return;
            }}
            y = cast<Y>({best_var});
        """
    elif operation in ("gradient", "enhance_contrast", "autolevel"):
        pre = "int n_vals = 0;\nX min_val;\nX max_val;"
        update = """
            X v = {value};
            if (n_vals == 0) {
                min_val = v;
                max_val = v;
            } else {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
            n_vals++;
        """
        if operation == "gradient":
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                y = cast<Y>(max_val - min_val);
            """
        elif operation == "enhance_contrast":
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                X g = x[i];
                if (max_val - g < g - min_val) {
                    y = cast<Y>(max_val);
                } else {
                    y = cast<Y>(min_val);
                }
            """
        else:
            post = f"""
                if (n_vals == 0) {{
                    y = cast<Y>(x[i]);
                    return;
                }}
                X g = x[i];
                X clamped = (g < min_val) ? min_val :
                    ((g > max_val) ? max_val : g);
                double delta = static_cast<double>(max_val - min_val);
                if (delta > 0) {{
                    double scaled = (static_cast<double>(clamped - min_val)
                                     / delta) * static_cast<double>({dtype_max});
                    y = cast<Y>(scaled);
                }} else {{
                    y = cast<Y>(0);
                }}
            """
    elif operation in ("mean", "sum", "subtract_mean", "threshold_mean"):
        pre = "int n_vals = 0;\ndouble sum = 0.0;"
        update = """
            X v = {value};
            sum += static_cast<double>(v);
            n_vals++;
        """
        if operation == "mean":
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                y = cast<Y>(sum / n_vals);
            """
        elif operation == "sum":
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                y = cast<Y>(sum);
            """
        elif operation == "subtract_mean":
            _mid_bin = (dtype_max + 1) // 2
            post = f"""
                if (n_vals == 0) {{
                    y = cast<Y>(x[i]);
                    return;
                }}
                double mean = sum / n_vals;
                X g = x[i];
                y = cast<Y>((static_cast<double>(g) - mean) * 0.5 + {_mid_bin});
            """
        else:
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                double mean = sum / n_vals;
                X g = x[i];
                y = (static_cast<double>(g) > mean) ? cast<Y>(1) : cast<Y>(0);
            """
    elif operation == "pop":
        pre = "int n_vals = 0;"
        update = "n_vals++;"
        post = """
            if (n_vals == 0) {
                y = cast<Y>(x[i]);
                return;
            }
            y = cast<Y>(n_vals);
        """
    elif operation == "equalize":
        pre = "int n_vals = 0;\nint eq_rank = 0;\nX g = x[i];"
        update = """
            X v = {value};
            if (v <= g) eq_rank++;
            n_vals++;
        """
        post = f"""
            if (n_vals == 0) {{
                y = cast<Y>(x[i]);
                return;
            }}
            y = cast<Y>(static_cast<double>({dtype_max}) * eq_rank / n_vals);
        """
    elif operation == "geometric_mean":
        pre = "int n_vals = 0;\ndouble log_sum = 0.0;"
        update = """
            X v = {value};
            log_sum += log(static_cast<double>(v) + 1.0);
            n_vals++;
        """
        post = """
            if (n_vals == 0) {
                y = cast<Y>(x[i]);
                return;
            }
            y = cast<Y>(round(exp(log_sum / n_vals) - 1.0));
        """
    elif operation == "noise_filter":
        pre = """
            int n_vals = 0;
            bool nf_found = false;
            bool nf_has_dist = false;
            typename RankNoiseDistance<X>::type nf_min_dist = 0;
            X g = x[i];
        """
        update = """
            X v = {value};
            if (v == g) {
                nf_found = true;
            } else {
                typedef typename RankNoiseDistance<X>::type DistanceT;
                DistanceT vd = static_cast<DistanceT>(v);
                DistanceT gd = static_cast<DistanceT>(g);
                DistanceT d = (v > g) ? vd - gd : gd - vd;
                if (!nf_has_dist || d < nf_min_dist) nf_min_dist = d;
                nf_has_dist = true;
            }
            n_vals++;
        """
        post = """
            if (n_vals == 0) {
                y = cast<Y>(x[i]);
                return;
            }
            y = nf_found ? cast<Y>(0) : cast<Y>(nf_min_dist);
        """
    elif operation in ("bilateral_mean", "bilateral_pop", "bilateral_sum"):
        pre = """
            int n_vals = 0;
            int bilat_pop = 0;
            double bilat_sum = 0.0;
            X g = x[i];
            double gd = static_cast<double>(g);
        """
        update = f"""
            X v = {{value}};
            double vd = static_cast<double>(v);
            if (gd > (vd - {s0}) && gd < (vd + {s1})) {{
                bilat_pop++;
                bilat_sum += vd;
            }}
            n_vals++;
        """
        if operation == "bilateral_mean":
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                y = (bilat_pop > 0) ? cast<Y>(bilat_sum / bilat_pop) : cast<Y>(0);
            """
        elif operation == "bilateral_pop":
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                y = cast<Y>(bilat_pop);
            """
        else:
            post = """
                if (n_vals == 0) {
                    y = cast<Y>(x[i]);
                    return;
                }
                y = (bilat_pop > 0) ? cast<Y>(bilat_sum) : cast<Y>(0);
            """
    else:
        raise ValueError(f"Unsupported streaming operation: {operation}")

    update = update.replace("{value}", "__VALUE__")
    update = update.replace("{", "{{").replace("}", "}}")
    update = update.replace("__VALUE__", "{value}")

    if has_mask:
        ndim = len(w_shape)
        index_expr = " + ".join([f"ix_{j}" for j in range(ndim)])
        found = (
            "{{ ptrdiff_t _neighbor_idx = (" + index_expr + ") / sizeof(X); "
            "if ((bool)mask[_neighbor_idx]) {{ " + update + " }} }}"
        )
    else:
        found = update

    op_name = operation.replace("_", "")
    mask_str = "_masked" if has_mask else ""
    preamble = ""
    if operation == "noise_filter":
        preamble = r"""
template <typename T, bool = std::is_floating_point<T>::value>
struct RankNoiseDistance {
    typedef typename std::make_unsigned<T>::type type;
};

template <typename T>
struct RankNoiseDistance<T, true> {
    typedef double type;
};
"""
    return _filters_core._generate_nd_kernel(
        f"rank_stream_{op_name}_{int(p0)}_{int(p1)}{mask_str}",
        pre,
        found,
        post,
        modes,
        w_shape,
        int_type,
        offsets,
        cval,
        has_weights=has_weights,
        has_mask=has_mask,
        preamble=preamble,
    )


@cp.memoize(for_each_device=True)
def _get_percentile_range_kernel(
    filter_size,
    p0,
    p1,
    operation,
    modes,
    w_shape,
    offsets,
    cval,
    int_type,
    has_weights,
    *,
    has_mask=False,
    dtype_max=255,
    s0=0.0,
    s1=0.0,
):
    """Generate a kernel for computing statistics on a percentile range.

    Parameters
    ----------
    filter_size : int
        Total number of values in the neighborhood (when mask is not used).
    p0 : float
        Lower percentile (0-100).
    p1 : float
        Upper percentile (0-100).
    operation : str
        The operation to perform on values in the percentile range.
        Supported operations:
        - 'mean': arithmetic mean
        - 'sum': sum of values
        - 'bilateral_mean': mean excluding center value
        - 'pop_mean': mean using center as reference
                      (percentile mean of |values - center|)
    modes : tuple of str
        Boundary handling modes.
    w_shape : tuple of int
        Shape of the footprint/kernel.
    offsets : tuple of int
        Offsets for the footprint origin.
    cval : float
        Constant value for 'constant' mode.
    int_type : str
        Integer type to use for indexing.
    has_weights : bool
        Whether a footprint mask is used.
    has_mask : bool
        Whether an image mask is used to filter neighborhood pixels.

    Returns
    -------
    kernel : cupy.ElementwiseKernel
        The compiled CUDA kernel.
    """
    # Convert percentiles to array indices
    # Note: Matches scikit-image's histogram-based percentile approach
    # where values are included if cumsum is in [p0*pop, p1*pop]
    _single_percentile_op = operation in ("percentile", "threshold")
    _bilateral_op = operation in (
        "bilateral_mean",
        "bilateral_pop",
        "bilateral_sum",
    )
    _full_range = p0 <= 0 and p1 >= 100
    _streaming_ops = {
        "bilateral_mean",
        "bilateral_pop",
        "bilateral_sum",
        "equalize",
        "geometric_mean",
        "minimum",
        "maximum",
        "noise_filter",
        "threshold_mean",
    }
    _full_range_streaming_ops = {
        "autolevel",
        "enhance_contrast",
        "gradient",
        "mean",
        "pop",
        "subtract_mean",
        "sum",
    }
    _skip_idx = _single_percentile_op or _bilateral_op
    if not _bilateral_op:
        if p0 < 0 or p0 > 100:
            raise ValueError("Percentiles must be in range [0, 100]")
        if not _single_percentile_op:
            if p1 < 0 or p1 > 100:
                raise ValueError("Percentiles must be in range [0, 100]")
            if p0 >= p1:
                raise ValueError("p0 must be less than p1")

    if operation in _streaming_ops or (
        _full_range and operation in _full_range_streaming_ops
    ):
        return _get_streaming_rank_kernel(
            p0,
            p1,
            operation,
            modes,
            w_shape,
            offsets,
            cval,
            int_type,
            has_weights,
            has_mask=has_mask,
            dtype_max=dtype_max,
            s0=s0,
            s1=s1,
        )

    # When has_mask is True, we need to dynamically calculate indices based on
    # the actual number of values collected (which depends on the mask).
    # We'll use runtime calculation in the CUDA code.
    # "percentile", "threshold", and bilateral ops compute their own indices,
    # so idx_start/idx_end are not needed.
    if not has_mask and not _skip_idx:
        # Calculate indices for the percentile range
        # (pre-computed at compile time)
        # Matches scikit-image's histogram-based approach where value at
        # index i is included if: (i + 1) >= p0 * N  AND  (i + 1) <= p1 * N
        # Rearranging: p0 * N - 1 <= i <= p1 * N - 1
        #
        # For integer indices in a sorted array (using Python's range with
        # exclusive upper bound):
        #   idx_start = ceil(p0 * N - 1) = ceil(p0 * N) - 1
        #   idx_end = floor(p1 * N - 1) + 1 = floor(p1 * N)
        idx_start = max(0, int(math.ceil(p0 * filter_size / 100.0)) - 1)
        # int() gives floor for positive values
        idx_end = int(p1 * filter_size / 100.0)

        # Ensure at least one value is included
        if idx_end <= idx_start:
            idx_end = idx_start + 1
        idx_end = min(idx_end, filter_size)
        n_values = idx_end - idx_start

    # Always use full sorting for percentile ranges
    array_size = filter_size
    sorter = __SHELL_SORT.format(gap=_get_shell_gap(filter_size))

    if has_mask:
        # Runtime calculation of indices based on actual count
        if operation in ("percentile", "threshold", "pop") or _bilateral_op:
            post = """
                if (iv == 0) {{
                    y = cast<Y>(x[i]);  // No valid values, keep original
                    return;
                }}
                sort(values, iv);"""
        else:
            post = f"""
                if (iv == 0) {{
                    y = cast<Y>(x[i]);  // No valid values, keep original
                    return;
                }}
                sort(values, iv);
                int actual_start = max(0, (int)ceil({p0 / 100.0} * iv) - 1);
                int actual_end = (int)({p1 / 100.0} * iv);
                if (actual_end <= actual_start) actual_end = actual_start + 1;
                if (actual_end > iv) actual_end = iv;"""
    else:
        post = f"""
            sort(values, {filter_size});"""

    # Generate the post-processing code based on the operation
    if operation == "mean":
        # Standard mean of values in percentile range
        if has_mask:
            # Runtime calculation of indices based on actual count
            post += """
                int n_vals = actual_end - actual_start;
                double sum = 0.0;
                for (int j = actual_start; j < actual_end; j++) {{
                    sum += static_cast<double>(values[j]);
                }}
                y = cast<Y>(sum / n_vals);
            """
        else:
            post += f"""
                double sum = 0.0;
                for (int j = {idx_start}; j < {idx_end}; j++) {{
                    sum += static_cast<double>(values[j]);
                }}
                y = cast<Y>(sum / {n_values});
            """
    elif operation == "sum":
        # Sum of values in percentile range
        if has_mask:
            post += """
                double sum = 0.0;
                for (int j = actual_start; j < actual_end; j++) {{
                    sum += static_cast<double>(values[j]);
                }}
                y = cast<Y>(sum);
            """
        else:
            post += f"""
                double sum = 0.0;
                for (int j = {idx_start}; j < {idx_end}; j++) {{
                    sum += static_cast<double>(values[j]);
                }}
                y = cast<Y>(sum);
            """
    elif operation == "gradient":
        # Gradient: max - min in percentile range
        if has_mask:
            post += """
                X min_val = values[actual_start];
                X max_val = values[actual_end - 1];
                y = cast<Y>(max_val - min_val);
            """
        else:
            post += f"""
                X min_val = values[{idx_start}];
                X max_val = values[{idx_end - 1}];
                y = cast<Y>(max_val - min_val);
            """
    elif operation == "subtract_mean":
        # Subtract mean: scikit-image formula:
        #   (g - mean) * 0.5 + mid_bin
        # where mid_bin = n_bins / 2 (128 for uint8, 32768 for uint16).
        # This centers the result so that g == mean maps to mid_bin.
        _mid_bin = (dtype_max + 1) // 2
        if has_mask:
            post += f"""
                int n_vals = actual_end - actual_start;
                double sum = 0.0;
                for (int j = actual_start; j < actual_end; j++) {{
                    sum += static_cast<double>(values[j]);
                }}
                double mean = sum / n_vals;
                X g = x[i];
                y = cast<Y>((static_cast<double>(g) - mean) * 0.5 + {_mid_bin});
            """
        else:
            post += f"""
                double sum = 0.0;
                for (int j = {idx_start}; j < {idx_end}; j++) {{
                    sum += static_cast<double>(values[j]);
                }}
                double mean = sum / {n_values};
                X g = x[i];
                y = cast<Y>((static_cast<double>(g) - mean) * 0.5 + {_mid_bin});
            """
    elif operation == "enhance_contrast":
        # Enhance contrast: replace with closer extreme (min or max)
        if has_mask:
            post += """
                X min_val = values[actual_start];
                X max_val = values[actual_end - 1];
                X g = x[i];
                // Replace with whichever extreme is closer
                if (max_val - g < g - min_val) {{
                    y = cast<Y>(max_val);
                }} else {{
                    y = cast<Y>(min_val);
                }}
            """
        else:
            post += f"""
                X min_val = values[{idx_start}];
                X max_val = values[{idx_end - 1}];
                X g = x[i];
                if (max_val - g < g - min_val) {{
                    y = cast<Y>(max_val);
                }} else {{
                    y = cast<Y>(min_val);
                }}
            """
    elif operation == "percentile":
        # Single percentile value (p0 determines which percentile)
        # Note: This returns the value AT the p0 percentile
        if has_mask:
            post += f"""
                int percentile_idx;
                if ({p0 / 100.0} == 1.0) {{
                    // p0 = 100%: return maximum
                    percentile_idx = iv - 1;
                }} else {{
                    // Find index where cumsum > p0 * pop
                    percentile_idx = (int)({p0 / 100.0} * iv);
                    if (percentile_idx >= iv) percentile_idx = iv - 1;
                }}
                y = cast<Y>(values[percentile_idx]);
            """
        else:
            # For no mask, we can use precomputed idx_start
            post += f"""
                int percentile_idx;
                if ({p0 / 100.0} == 1.0) {{
                    percentile_idx = {filter_size - 1};
                }} else {{
                    percentile_idx = (int)({p0 / 100.0} * {filter_size});
                    if (percentile_idx >= {filter_size}) {{
                        percentile_idx = {filter_size - 1};
                    }}
                }}
                y = cast<Y>(values[percentile_idx]);
            """
    elif operation == "pop":
        # Population: count of pixels in percentile range.
        # Must match scikit-image's histogram-bin grouping: groups of equal
        # values are included/excluded as a whole based on whether the
        # cumulative count after adding the group falls in [p0*pop, p1*pop].
        if has_mask:
            post += f"""
                int pop_n = 0;
                int pop_cumsum = 0;
                int pop_j = 0;
                while (pop_j < iv) {{
                    X pop_val = values[pop_j];
                    int pop_gs = 0;
                    while (pop_j < iv && values[pop_j] == pop_val) {{
                        pop_j++;
                        pop_gs++;
                    }}
                    pop_cumsum += pop_gs;
                    if ((double)pop_cumsum >= {p0 / 100.0} * iv &&
                        (double)pop_cumsum <= {p1 / 100.0} * iv) {{
                        pop_n += pop_gs;
                    }}
                }}
                y = cast<Y>(pop_n);
            """
        else:
            post += f"""
                int pop_n = 0;
                int pop_cumsum = 0;
                int pop_j = 0;
                while (pop_j < {filter_size}) {{
                    X pop_val = values[pop_j];
                    int pop_gs = 0;
                    while (pop_j < {filter_size} && values[pop_j] == pop_val) {{
                        pop_j++;
                        pop_gs++;
                    }}
                    pop_cumsum += pop_gs;
                    if ((double)pop_cumsum >= {p0 / 100.0} * {filter_size} &&
                        (double)pop_cumsum <= {p1 / 100.0} * {filter_size}) {{
                        pop_n += pop_gs;
                    }}
                }}
                y = cast<Y>(pop_n);
            """
    elif operation == "threshold":
        # Threshold: binary output comparing center pixel to p0 percentile.
        # scikit-image uses (n_bins - 1) * (g >= threshold), which gives the
        # dtype max value (e.g. 255 for uint8) or 0.
        if has_mask:
            post += f"""
                int threshold_idx = (int)({p0 / 100.0} * iv);
                if (threshold_idx >= iv) threshold_idx = iv - 1;
                X threshold_val = values[threshold_idx];
                X g = x[i];
                y = (g >= threshold_val) ? cast<Y>({dtype_max}) : cast<Y>(0);
            """
        else:
            post += f"""
                int threshold_idx = (int)({p0 / 100.0} * {filter_size});
                if (threshold_idx >= {filter_size}) {{
                    threshold_idx = {filter_size - 1};
                }}
                X threshold_val = values[threshold_idx];
                X g = x[i];
                y = (g >= threshold_val) ? cast<Y>({dtype_max}) : cast<Y>(0);
            """
    elif operation == "autolevel":
        # Autolevel: stretch pixel values to full dtype range based on local
        # percentile min/max. scikit-image formula:
        #   (n_bins - 1) * (clamp(g, imin, imax) - imin) / (imax - imin)
        # Scales output to [0, dtype_max], NOT [0, local_max].
        if has_mask:
            post += f"""
                X min_val = values[actual_start];
                X max_val = values[actual_end - 1];
                X g = x[i];
                X clamped = (g < min_val) ? min_val : \
((g > max_val) ? max_val : g);
                double delta = static_cast<double>(max_val - min_val);
                if (delta > 0) {{
                    double scaled = (static_cast<double>(clamped - min_val) \
/ delta) * static_cast<double>({dtype_max});
                    y = cast<Y>(scaled);
                }} else {{
                    y = cast<Y>(0);
                }}
            """
        else:
            post += f"""
                X min_val = values[{idx_start}];
                X max_val = values[{idx_end - 1}];
                X g = x[i];
                X clamped = (g < min_val) ? min_val : \
((g > max_val) ? max_val : g);
                double delta = static_cast<double>(max_val - min_val);
                if (delta > 0) {{
                    double scaled = (static_cast<double>(clamped - min_val) \
/ delta) * static_cast<double>({dtype_max});
                    y = cast<Y>(scaled);
                }} else {{
                    y = cast<Y>(0);
                }}
            """
    elif operation == "modal":
        # Modal: most frequent value (mode) in the neighborhood.
        # Scan sorted array for the longest run of equal values.
        if has_mask:
            post += """
                X mode_val = values[0];
                int mode_max = 1;
                int mode_cur = 1;
                for (int j = 1; j < iv; j++) {
                    if (values[j] == values[j - 1]) {
                        mode_cur++;
                    } else {
                        if (mode_cur > mode_max) {
                            mode_max = mode_cur;
                            mode_val = values[j - 1];
                        }
                        mode_cur = 1;
                    }
                }
                if (mode_cur > mode_max) mode_val = values[iv - 1];
                y = cast<Y>(mode_val);
            """
        else:
            post += f"""
                X mode_val = values[0];
                int mode_max = 1;
                int mode_cur = 1;
                for (int j = 1; j < {filter_size}; j++) {{
                    if (values[j] == values[j - 1]) {{
                        mode_cur++;
                    }} else {{
                        if (mode_cur > mode_max) {{
                            mode_max = mode_cur;
                            mode_val = values[j - 1];
                        }}
                        mode_cur = 1;
                    }}
                }}
                if (mode_cur > mode_max) mode_val = values[{filter_size} - 1];
                y = cast<Y>(mode_val);
            """
    elif operation == "entropy":
        # Shannon entropy in bits: -sum(p * log2(p)) where p = count / N.
        # Uses run-length counting on the sorted array to get value counts.
        # log2(p) = log(p) / log(2); 0.6931471805599453 = log(2).
        if has_mask:
            post += """
                double ent = 0.0;
                int ent_j = 0;
                while (ent_j < iv) {
                    int ent_count = 1;
                    while (ent_j + ent_count < iv &&
                           values[ent_j + ent_count] == values[ent_j])
                        ent_count++;
                    double p = static_cast<double>(ent_count) / iv;
                    ent -= p * log(p) / 0.6931471805599453;
                    ent_j += ent_count;
                }
                y = cast<Y>(ent);
            """
        else:
            post += f"""
                double ent = 0.0;
                int ent_j = 0;
                while (ent_j < {filter_size}) {{
                    int ent_count = 1;
                    while (ent_j + ent_count < {filter_size} &&
                           values[ent_j + ent_count] == values[ent_j])
                        ent_count++;
                    double p = static_cast<double>(ent_count) / {filter_size};
                    ent -= p * log(p) / 0.6931471805599453;
                    ent_j += ent_count;
                }}
                y = cast<Y>(ent);
            """
    else:
        raise ValueError(
            f"Unsupported operation: {operation}. "
            "Supported sorted operations: 'mean', 'sum', 'gradient', "
            "'subtract_mean', 'enhance_contrast', 'percentile', 'pop', "
            "'threshold', 'autolevel', 'modal', 'entropy'"
        )

    # Sanitize operation name for kernel name (replace special chars)
    op_name = operation.replace("_", "")

    # Build the pre string and found string with neighborhood-level masking
    pre = f"int iv = 0;\nX values[{array_size}];"

    if has_mask:
        # Neighborhood-level masking: check mask at each neighbor's position
        # Calculate the neighbor's element index from the byte offset.
        # Note: ix_{j} are calculated in _generate_nd_kernel (in
        # _ndimage_filters_core.py) within the neighborhood iteration loops.
        # Each ix_{j} = coordinate_{j} * xstride_{j}, where xstride_{j} comes
        # from x.strides()[{j}] (byte stride). The sum (ix_0 + ix_1 + ...)
        # gives the total byte offset to the neighbor pixel from the start
        # of the array.
        ndim = len(w_shape)
        index_expr = " + ".join([f"ix_{j}" for j in range(ndim)])
        # Use string concatenation (not f-string) so that {{ / }} are
        # correctly interpreted as literal braces by .format() later.
        found = (
            "{{ ptrdiff_t _neighbor_idx = (" + index_expr + ") / sizeof(X); "
            "if ((bool)mask[_neighbor_idx]) {{ "
            "values[iv++] = {value}; "
            "}} }}"
        )
    else:
        found = "values[iv++] = {value};"

    mask_str = "_masked" if has_mask else ""
    return _filters_core._generate_nd_kernel(
        f"percentile_range_{filter_size}_{int(p0)}_{int(p1)}_{op_name}{mask_str}",
        pre,
        found,
        post,
        modes,
        w_shape,
        int_type,
        offsets,
        cval,
        has_weights=has_weights,
        has_mask=has_mask,
        preamble=sorter,
    )


def _is_decomposed_footprint(footprint):
    """Return True for morphology footprint decomposition sequences."""
    if not isinstance(footprint, (tuple, list)) or len(footprint) == 0:
        return False

    for item in footprint:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return False
        footprint_part, num_iter = item
        if not hasattr(footprint_part, "ndim"):
            return False
        if not isinstance(num_iter, (int, np.integer)):
            return False

    return True


def _skimage_rank_filter(
    input,
    p0,
    p1,
    operation="mean",
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
    *,
    mask=None,
    s0=0,
    s1=0,
    backend="auto",
):
    """Internal helper for percentile range filters.

    This function computes statistics (mean, sum, etc.) on values within
    a specified percentile range [p0, p1] of the neighborhood.

    Parameters
    ----------
    input : cupy.ndarray
        The input array.
    p0 : float
        Lower percentile (0-100).
    p1 : float
        Upper percentile (0-100).
    operation : str, optional
        The operation to perform. Supported: 'mean', 'sum', 'bilateral_mean',
        'pop_mean'. Default is 'mean'.
    size : int or sequence of int, optional
        Size of the neighborhood. One of `size` or `footprint` must be provided.
    footprint : cupy.ndarray, optional
        Boolean array specifying the neighborhood shape.
    output : cupy.ndarray, dtype or None, optional
        The array in which to place the output.
    mode : str, optional
        Boundary handling mode. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges if mode is 'constant'. Default is 0.0.
    origin : int or sequence of int, optional
        Origin of the footprint. Default is 0.
    axes : tuple of int or None, optional
        Axes along which to apply the filter. Default is None (all axes).
    mask : cupy.ndarray or None, optional
        If provided, only neighbor pixels where mask is True are included
        when computing statistics. This matches scikit-image's filters.rank
        behavior where the mask filters which pixels in the local neighborhood
        contribute to the computation. Output is computed for all pixels, but
        each uses a different set of neighbors based on the mask.

    Returns
    -------
    output : cupy.ndarray
        The filtered array.

    Notes
    -----
    This function is for internal use as a common implementation for filters
    under cucim.skimage.filters.rank.
    """
    if backend not in ("auto", "histogram", "elementwise"):
        raise ValueError(
            "backend must be one of 'auto', 'histogram' or 'elementwise'"
        )

    ndim = input.ndim
    axes = _util._check_axes(axes, ndim)
    num_axes = len(axes)
    default_footprint = footprint is None
    if _is_decomposed_footprint(footprint):
        raise ValueError(
            "decomposed footprint sequences are not supported by rank filters"
        )
    sizes, footprint, _ = _filters_core._check_size_footprint_structure(
        num_axes,
        size,
        footprint,
        None,
        force_footprint=operation == "noise_filter",
    )
    if cval is cp.nan:
        raise NotImplementedError("NaN cval is unsupported")

    # Validate percentiles
    p0 = float(p0)
    p1 = float(p1)
    _bilateral_op = operation in (
        "bilateral_mean",
        "bilateral_pop",
        "bilateral_sum",
    )
    _single_percentile_op = operation in ("percentile", "threshold")
    if not _bilateral_op:
        if p0 < 0 or p0 > 100:
            raise ValueError("Percentiles must be in range [0, 100]")
        if not _single_percentile_op:
            if p1 < 0 or p1 > 100:
                raise ValueError("Percentiles must be in range [0, 100]")
            if p0 >= p1:
                raise ValueError("p0 must be less than p1")

    has_weights = True
    if sizes is not None:
        has_weights = False
        filter_size = math.prod(sizes)
        if filter_size == 0:
            return cp.zeros_like(input)
        footprint_shape = tuple(sizes)
        (
            axes,
            footprint,
            origins,
            modes,
            int_type,
        ) = _filters_core._check_nd_args(
            input,
            None,
            mode,
            origin,
            "footprint",
            axes=axes,
            sizes=footprint_shape,
        )
    else:
        if footprint.size == 0:
            return cp.zeros_like(input)

        (
            axes,
            footprint,
            origins,
            modes,
            int_type,
        ) = _filters_core._check_nd_args(
            input, footprint, mode, origin, "footprint", axes=axes
        )

        if operation == "noise_filter":
            # The footprint anchor addresses the center input pixel. Exclude
            # it so that it cannot make every pixel appear non-isolated.
            footprint = footprint.copy()
            anchor = _filters_core._origins_to_offsets(origins, footprint.shape)
            footprint[anchor] = False

        if default_footprint:
            filter_size = footprint.size
        else:
            footprint_shape = footprint.shape
            filter_size = int(footprint.sum())
            if filter_size == footprint.size:
                # can omit passing the footprint if it is all ones
                sizes = footprint.shape
                has_weights = False

    if not has_weights:
        footprint = None

    offsets = _filters_core._origins_to_offsets(origins, footprint_shape)
    if num_axes < ndim and not has_weights:
        offsets = tuple(_util._expand_origin(ndim, axes, offsets))
        modes = tuple(_util._expand_mode(ndim, axes, modes))
        footprint_shape_temp = [1] * ndim
        for s, ax in zip(footprint_shape, axes):
            footprint_shape_temp[ax] = s
        footprint_shape = tuple(footprint_shape_temp)

    has_mask = mask is not None

    # Compute dtype max for threshold operation (binary output needs
    # the type's max value, not the local neighborhood max).
    _out_dtype = output.dtype if output is not None else input.dtype
    if np.issubdtype(_out_dtype, np.integer):
        _dtype_max = int(np.iinfo(_out_dtype).max)
    else:
        _dtype_max = 1.0

    can_use_histogram = _can_use_rank_histogram(
        input,
        footprint_shape,
        output,
        mask,
        modes,
        origins,
        has_weights=has_weights,
        operation=operation,
        p0=p0,
        p1=p1,
    )
    if backend == "histogram" and not can_use_histogram:
        raise ValueError(
            "backend='histogram' requires a supported uint8 2D rank "
            "operation, compatible output, no mask, zero shifts, reflect "
            "mode, and an all-ones odd rectangular footprint"
        )

    if backend == "histogram" or (
        backend == "auto"
        and can_use_histogram
        and _should_use_rank_histogram(operation, footprint_shape)
    ):
        return _rank_histogram(
            input,
            footprint_shape,
            operation,
            output=output,
            mode=modes[0],
            cval=cval,
            p0=p0,
            p1=p1,
            s0=s0,
            s1=s1,
            dtype_max=_dtype_max,
        )

    kernel = _get_percentile_range_kernel(
        filter_size,
        p0,
        p1,
        operation,
        modes,
        footprint_shape,
        offsets,
        float(cval),
        int_type,
        has_weights=has_weights,
        has_mask=has_mask,
        dtype_max=_dtype_max,
        s0=float(s0),
        s1=float(s1),
    )
    kwargs = dict(weights_dtype=bool)
    if has_mask:
        kwargs["mask"] = mask
    return _filters_core._call_kernel(
        kernel, input, footprint, output, **kwargs
    )
