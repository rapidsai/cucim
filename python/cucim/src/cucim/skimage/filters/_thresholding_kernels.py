# SPDX-FileCopyrightText: 2009-2015 Board of Regents of the University of Wisconsin-Madison, Broad Institute of MIT and Harvard, and Max Planck Institute of Molecular Cell Biology and Genetics
# SPDX-FileCopyrightText: 2009 Zachary Pincus
# SPDX-FileCopyrightText: 2009 Almar Klein
# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU kernels of the thresholding functions (see ``thresholding.py``).

The functions that make the kernels (``_get_*_kernel``) and the helpers that
launch them (``_threshold_*_gpu``, ``_multiotsu_threshold_indices``) are here, so
that ``thresholding.py`` keeps the structure of the scikit-image implementation.
"""

from functools import cache

import cupy as cp
import numpy as np

_DTYPE_TO_CTYPE = {
    "?": "bool",
    "b": "signed char",
    "B": "unsigned char",
    "h": "short",
    "H": "unsigned short",
    "i": "int",
    "I": "unsigned int",
    "l": "long",
    "L": "unsigned long",
    "q": "long long",
    "Q": "unsigned long long",
    "f": "float",
    "d": "double",
}


# Number of threads of the (single) block of the histogram threshold kernels:
# a block can scan histograms of any number of bins, each thread handling a
# chunk of consecutive bins (for 65,536 bins the kernels take 0.2 to 0.7 ms).
_BLOCK_THREADS = 512


@cache
def _get_threshold_otsu_kernel(bin_centers_dtype):
    ctype = _DTYPE_TO_CTYPE[np.dtype(bin_centers_dtype).char]
    code = f"""
#define BLOCK {_BLOCK_THREADS}

// One block of BLOCK threads processes the bins. Thread t owns the contiguous
// chunk of bins [t * chunk, (t + 1) * chunk): it sums its chunk, the chunk sums
// are scanned in shared memory (so every thread knows the sums over the bins
// before its chunk), and then every thread sequentially evaluates the criterion
// for the bins of its chunk. The best candidate of each thread is reduced in
// the order of the threads, which keeps the first maximum.

    extern "C" __global__ void threshold_otsu(
            const float* counts,
            const {ctype}* bin_centers,
            const long long n_bins,
            double* threshold)
    {{
        __shared__ double s_count[BLOCK];
        __shared__ double s_weighted[BLOCK];
        __shared__ double s_best[BLOCK];
        __shared__ long long s_idx[BLOCK];
        __shared__ double s_total[2];

        const int tid = threadIdx.x;
        if (n_bins <= 0) {{
            if (tid == 0) {{
                threshold[0] = 0.0;
            }}
            return;
        }}

        const long long chunk = (n_bins + BLOCK - 1) / BLOCK;
        const long long begin = min(static_cast<long long>(tid) * chunk, n_bins);
        const long long end = min(begin + chunk, n_bins);

        double sum_count = 0.0;
        double sum_weighted = 0.0;
        for (long long i = begin; i < end; ++i) {{
            const double count = static_cast<double>(counts[i]);
            sum_count += count;
            sum_weighted += count * static_cast<double>(bin_centers[i]);
        }}
        s_count[tid] = sum_count;
        s_weighted[tid] = sum_weighted;
        __syncthreads();

        if (tid == 0) {{
            double run_count = 0.0;
            double run_weighted = 0.0;
            for (int t = 0; t < BLOCK; ++t) {{
                const double c = s_count[t];
                const double w = s_weighted[t];
                s_count[t] = run_count;
                s_weighted[t] = run_weighted;
                run_count += c;
                run_weighted += w;
            }}
            s_total[0] = run_count;
            s_total[1] = run_weighted;
        }}
        __syncthreads();

        const double total_count = s_total[0];
        const double total_weighted = s_total[1];
        if (n_bins == 1 || total_count <= 0.0) {{
            if (tid == 0) {{
                threshold[0] = static_cast<double>(bin_centers[0]);
            }}
            return;
        }}

        double weight1 = s_count[tid];
        double weighted1 = s_weighted[tid];
        double best_variance = -1.0;
        long long best_idx = -1;
        for (long long i = begin; i < end && i < n_bins - 1; ++i) {{
            const double count = static_cast<double>(counts[i]);
            weight1 += count;
            weighted1 += count * static_cast<double>(bin_centers[i]);

            const double weight2 = total_count - weight1;
            if (weight1 <= 0.0 || weight2 <= 0.0) {{
                continue;
            }}

            const double mean1 = weighted1 / weight1;
            const double mean2 = (total_weighted - weighted1) / weight2;
            const double diff = mean1 - mean2;
            const double variance = weight1 * weight2 * diff * diff;

            // Match np.argmax tie-breaking by keeping the first maximum.
            if (variance > best_variance) {{
                best_variance = variance;
                best_idx = i;
            }}
        }}
        s_best[tid] = best_variance;
        s_idx[tid] = best_idx;
        __syncthreads();

        if (tid == 0) {{
            double best = -1.0;
            long long idx = 0;
            for (int t = 0; t < BLOCK; ++t) {{
                if (s_best[t] > best) {{
                    best = s_best[t];
                    idx = s_idx[t];
                }}
            }}
            threshold[0] = static_cast<double>(bin_centers[idx]);
        }}
    }}
    """
    return cp.RawKernel(code, "threshold_otsu")


def _threshold_otsu_gpu(counts, bin_centers):
    counts = cp.ascontiguousarray(counts, dtype=cp.float32)
    bin_centers = cp.ascontiguousarray(bin_centers)
    if bin_centers.dtype.char not in _DTYPE_TO_CTYPE:
        bin_centers = bin_centers.astype(cp.float64)
    threshold = cp.empty((), dtype=cp.float64)
    kernel = _get_threshold_otsu_kernel(bin_centers.dtype)
    kernel(
        (1,),
        (_BLOCK_THREADS,),
        (counts, bin_centers, np.int64(counts.size), threshold),
    )
    return threshold


@cache
def _get_threshold_yen_kernel(bin_centers_dtype):
    ctype = _DTYPE_TO_CTYPE[np.dtype(bin_centers_dtype).char]
    code = f"""

#define BLOCK {_BLOCK_THREADS}

// One block of BLOCK threads processes the bins. Thread t owns the contiguous
// chunk of bins [t * chunk, (t + 1) * chunk): it sums its chunk, the chunk sums
// are scanned in shared memory (so every thread knows the sums over the bins
// before its chunk), and then every thread sequentially evaluates the criterion
// for the bins of its chunk. The best candidate of each thread is reduced in
// the order of the threads, which keeps the first maximum.

    extern "C" __global__ void threshold_yen(
            const float* counts,
            const {ctype}* bin_centers,
            const long long n_bins,
            double* threshold)
    {{
        __shared__ double s_count[BLOCK];
        __shared__ double s_squared[BLOCK];
        __shared__ double s_best[BLOCK];
        __shared__ long long s_idx[BLOCK];
        __shared__ double s_total[2];

        const int tid = threadIdx.x;
        if (n_bins <= 0) {{
            if (tid == 0) {{
                threshold[0] = 0.0;
            }}
            return;
        }}
        if (n_bins == 1) {{
            if (tid == 0) {{
                threshold[0] = static_cast<double>(bin_centers[0]);
            }}
            return;
        }}

        const long long chunk = (n_bins + BLOCK - 1) / BLOCK;
        const long long begin = min(static_cast<long long>(tid) * chunk, n_bins);
        const long long end = min(begin + chunk, n_bins);

        double sum_count = 0.0;
        double sum_squared = 0.0;
        for (long long i = begin; i < end; ++i) {{
            const double count = static_cast<double>(counts[i]);
            sum_count += count;
            sum_squared += count * count;
        }}
        s_count[tid] = sum_count;
        s_squared[tid] = sum_squared;
        __syncthreads();

        if (tid == 0) {{
            double run_count = 0.0;
            double run_squared = 0.0;
            for (int t = 0; t < BLOCK; ++t) {{
                const double c = s_count[t];
                const double q = s_squared[t];
                s_count[t] = run_count;
                s_squared[t] = run_squared;
                run_count += c;
                run_squared += q;
            }}
            s_total[0] = run_count;
            s_total[1] = run_squared;
        }}
        __syncthreads();

        const double total_count = s_total[0];
        const double total_squared_count = s_total[1];

        double cum_count = s_count[tid];
        double cum_squared_count = s_squared[tid];
        double best_score = -1.0;
        long long best_idx = -1;
        for (long long i = begin; i < end && i < n_bins - 1; ++i) {{
            const double count = static_cast<double>(counts[i]);
            cum_count += count;
            cum_squared_count += count * count;

            const double suffix_count = total_count - cum_count;
            const double suffix_squared_count =
                    total_squared_count - cum_squared_count;
            if (cum_count <= 0.0 || suffix_count <= 0.0 ||
                    cum_squared_count <= 0.0 ||
                    suffix_squared_count <= 0.0) {{
                continue;
            }}

            // Equivalent to scikit-image's Yen criterion, with histogram
            // normalization factors cancelled out and without the logarithm
            // (which does not change the position of the maximum):
            // (P1 * (1 - P1))^2 / (P1_sq * P2_sq).
            const double product = cum_count * suffix_count;
            const double score = product * product /
                    (cum_squared_count * suffix_squared_count);

            // Match np.argmax tie-breaking by keeping the first maximum.
            if (score > best_score) {{
                best_score = score;
                best_idx = i;
            }}
        }}
        s_best[tid] = best_score;
        s_idx[tid] = best_idx;
        __syncthreads();

        if (tid == 0) {{
            double best = -1.0;
            long long idx = 0;
            for (int t = 0; t < BLOCK; ++t) {{
                if (s_best[t] > best) {{
                    best = s_best[t];
                    idx = s_idx[t];
                }}
            }}
            threshold[0] = static_cast<double>(bin_centers[idx]);
        }}
    }}
    """
    return cp.RawKernel(code, "threshold_yen")


def _threshold_yen_gpu(counts, bin_centers):
    counts = cp.ascontiguousarray(counts, dtype=cp.float32)
    bin_centers = cp.ascontiguousarray(bin_centers)
    if bin_centers.dtype.char not in _DTYPE_TO_CTYPE:
        bin_centers = bin_centers.astype(cp.float64)
    threshold = cp.empty((), dtype=cp.float64)
    kernel = _get_threshold_yen_kernel(bin_centers.dtype)
    kernel(
        (1,),
        (_BLOCK_THREADS,),
        (counts, bin_centers, np.int64(counts.size), threshold),
    )
    return threshold


@cache
def _get_threshold_isodata_kernel(bin_centers_dtype):
    ctype = _DTYPE_TO_CTYPE[np.dtype(bin_centers_dtype).char]
    code = f"""
#define BLOCK {_BLOCK_THREADS}

// One block of BLOCK threads processes the bins. Thread t owns the contiguous
// chunk of bins [t * chunk, (t + 1) * chunk): it sums its chunk, the chunk sums
// are scanned in shared memory (so every thread knows the sums over the bins
// before its chunk), and then every thread sequentially evaluates the criterion
// for the bins of its chunk. The best candidate of each thread is reduced in
// the order of the threads, which keeps the first maximum.

    extern "C" __global__ void threshold_isodata(
            const float* counts,
            const {ctype}* bin_centers,
            const long long n_bins,
            double* threshold)
    {{
        __shared__ double s_count[BLOCK];
        __shared__ double s_weighted[BLOCK];
        __shared__ long long s_idx[BLOCK];
        __shared__ double s_total[2];

        const int tid = threadIdx.x;
        if (n_bins <= 0) {{
            if (tid == 0) {{
                threshold[0] = 0.0;
            }}
            return;
        }}
        if (n_bins == 1) {{
            if (tid == 0) {{
                threshold[0] = static_cast<double>(bin_centers[0]);
            }}
            return;
        }}

        const long long chunk = (n_bins + BLOCK - 1) / BLOCK;
        const long long begin = min(static_cast<long long>(tid) * chunk, n_bins);
        const long long end = min(begin + chunk, n_bins);

        double sum_count = 0.0;
        double sum_weighted = 0.0;
        for (long long i = begin; i < end; ++i) {{
            const double count = static_cast<double>(counts[i]);
            sum_count += count;
            sum_weighted += count * static_cast<double>(bin_centers[i]);
        }}
        s_count[tid] = sum_count;
        s_weighted[tid] = sum_weighted;
        __syncthreads();

        if (tid == 0) {{
            double run_count = 0.0;
            double run_weighted = 0.0;
            for (int t = 0; t < BLOCK; ++t) {{
                const double c = s_count[t];
                const double w = s_weighted[t];
                s_count[t] = run_count;
                s_weighted[t] = run_weighted;
                run_count += c;
                run_weighted += w;
            }}
            s_total[0] = run_count;
            s_total[1] = run_weighted;
        }}
        __syncthreads();

        const double total_count = s_total[0];
        const double total_weighted = s_total[1];
        const double bin_width =
                static_cast<double>(bin_centers[1]) -
                static_cast<double>(bin_centers[0]);

        double lower_count = s_count[tid];
        double lower_weighted = s_weighted[tid];
        long long found = n_bins;  // no valid bin in this chunk
        for (long long i = begin; i < end && i < n_bins - 1; ++i) {{
            const double count = static_cast<double>(counts[i]);
            const double center = static_cast<double>(bin_centers[i]);
            lower_count += count;
            lower_weighted += count * center;

            const double higher_count = total_count - lower_count;
            if (lower_count <= 0.0 || higher_count <= 0.0) {{
                continue;
            }}

            const double lower = lower_weighted / lower_count;
            const double higher =
                    (total_weighted - lower_weighted) / higher_count;
            const double all_mean = (lower + higher) / 2.0;
            const double distance = all_mean - center;
            if (distance >= 0.0 && distance < bin_width) {{
                found = i;
                break;
            }}
        }}
        s_idx[tid] = found;
        __syncthreads();

        if (tid == 0) {{
            // the lowest bin that satisfies the equality
            long long idx = 0;  // degenerate histograms: the first bin
            for (int t = 0; t < BLOCK; ++t) {{
                if (s_idx[t] < n_bins) {{
                    idx = s_idx[t];
                    break;
                }}
            }}
            threshold[0] = static_cast<double>(bin_centers[idx]);
        }}
    }}
    """
    return cp.RawKernel(code, "threshold_isodata")


def _threshold_isodata_gpu(counts, bin_centers):
    counts = cp.ascontiguousarray(counts, dtype=cp.float32)
    bin_centers = cp.ascontiguousarray(bin_centers)
    if bin_centers.dtype.char not in _DTYPE_TO_CTYPE:
        bin_centers = bin_centers.astype(cp.float64)
    threshold = cp.empty((), dtype=cp.float64)
    kernel = _get_threshold_isodata_kernel(bin_centers.dtype)
    kernel(
        (1,),
        (_BLOCK_THREADS,),
        (counts, bin_centers, np.int64(counts.size), threshold),
    )
    return threshold


# The largest histogram that `threshold_minimum` smooths in shared memory (two
# float32 copies of it).
_MINIMUM_MAX_BINS = 4096


@cache
def _get_threshold_minimum_kernel():
    # The histogram is smoothed by repeated uniform filters of size 3 until
    # it has at most two local maxima, as in scikit-image. All the iterations
    # run in one block, in shared memory and without synchronizations with the
    # host: the threads smooth the histogram (the mean of three values, in
    # double precision, stored as float32) and one thread looks for the local
    # maxima with the same search as scikit-image (plateaus are handled by the
    # direction of the previous change).
    code = r"""
    #define BLOCK 256

    __device__ __forceinline__ long long clamp_index(long long k, long long n)
    {
        return k < 0 ? 0 : (k >= n ? n - 1 : k);
    }

    extern "C" __global__ void threshold_minimum(
            const float* counts, const long long n_bins,
            const long long max_num_iter, long long* result)
    {
        extern __shared__ float smem[];
        __shared__ long long s_found[4];  // count, first, second, counter
        float* cur = smem;
        float* next = smem + n_bins;
        const int tid = threadIdx.x;

        for (long long i = tid; i < n_bins; i += BLOCK) {
            cur[i] = counts[i];
        }
        __syncthreads();

        long long count = 0;
        long long counter = -1;
        for (long long it = 0; it < max_num_iter; ++it) {
            // uniform_filter1d(cur, 3), mode 'reflect'
            for (long long l = tid; l < n_bins; l += BLOCK) {
                const double sum = static_cast<double>(cur[clamp_index(l - 1, n_bins)])
                        + static_cast<double>(cur[l])
                        + static_cast<double>(cur[clamp_index(l + 1, n_bins)]);
                next[l] = static_cast<float>(sum / 3.0);
            }
            __syncthreads();
            float* swap = cur;
            cur = next;
            next = swap;

            if (tid == 0) {
                long long found = 0;
                long long first = -1;
                long long second = -1;
                int direction = 1;
                for (long long i = 0; i < n_bins - 1; ++i) {
                    if (direction > 0) {
                        if (cur[i + 1] < cur[i]) {
                            direction = -1;
                            if (found == 0) {
                                first = i;
                            } else if (found == 1) {
                                second = i;
                            }
                            ++found;
                        }
                    } else if (cur[i + 1] > cur[i]) {
                        direction = 1;
                    }
                }
                s_found[0] = found;
                s_found[1] = first;
                s_found[2] = second;
            }
            __syncthreads();
            count = s_found[0];
            counter = it;
            if (count < 3) {
                break;
            }
        }

        if (tid == 0) {
            result[0] = 0;
            result[1] = 0;
            if (count != 2) {
                result[1] = 1;  // unable to find two maxima
            } else if (counter == max_num_iter - 1) {
                result[1] = 2;  // maximum number of iterations reached
            } else {
                // the lowest point (the first one) between the maxima
                const long long first = s_found[1];
                const long long second = s_found[2];
                long long best = first;
                for (long long i = first + 1; i <= second; ++i) {
                    if (cur[i] < cur[best]) {
                        best = i;
                    }
                }
                result[0] = best;
            }
        }
    }
    """
    return cp.RawKernel(code, "threshold_minimum", options=("--fmad=false",))


def _threshold_minimum_gpu(counts, bin_centers, max_num_iter):
    counts = cp.ascontiguousarray(counts, dtype=cp.float32)
    n_bins = counts.size
    result = cp.zeros(2, dtype=cp.int64)
    _get_threshold_minimum_kernel()(
        (1,),
        (256,),
        (counts, np.int64(n_bins), np.int64(max_num_iter), result),
        shared_mem=2 * n_bins * 4,
    )
    index, status = (int(x) for x in cp.asnumpy(result))  # synchronize
    if status == 1:
        raise RuntimeError("Unable to find two maxima in histogram")
    elif status == 2:
        raise RuntimeError("Maximum iteration reached for histogramsmoothing")
    return bin_centers[index]


@cache
def _get_threshold_triangle_kernel(bin_centers_dtype):
    ctype = _DTYPE_TO_CTYPE[np.dtype(bin_centers_dtype).char]
    code = f"""
#define BLOCK {_BLOCK_THREADS}

// One block of BLOCK threads processes the bins. Thread t owns the contiguous
// chunk of bins [t * chunk, (t + 1) * chunk): it sums its chunk, the chunk sums
// are scanned in shared memory (so every thread knows the sums over the bins
// before its chunk), and then every thread sequentially evaluates the criterion
// for the bins of its chunk. The best candidate of each thread is reduced in
// the order of the threads, which keeps the first maximum.

    extern "C" __global__ void threshold_triangle(
            const float* hist,
            const {ctype}* bin_centers,
            const long long n_bins,
            double* threshold)
    {{
        __shared__ double s_value[BLOCK];
        __shared__ long long s_arg[BLOCK];
        __shared__ long long s_low[BLOCK];
        __shared__ long long s_high[BLOCK];
        __shared__ double s_peak_height;
        __shared__ long long s_arg_low_level;
        __shared__ long long s_width;
        __shared__ int s_flip;
        __shared__ int s_done;

        const int tid = threadIdx.x;
        if (n_bins <= 0) {{
            if (tid == 0) {{
                threshold[0] = 0.0;
            }}
            return;
        }}

        // Pass 1: the peak (first maximum) and the first and last non-empty bin.
        const long long chunk = (n_bins + BLOCK - 1) / BLOCK;
        const long long begin = min(static_cast<long long>(tid) * chunk, n_bins);
        const long long end = min(begin + chunk, n_bins);
        double peak = -1.0e300;
        long long arg_peak = -1;
        long long low = -1;
        long long high = -1;
        for (long long i = begin; i < end; ++i) {{
            const double value = static_cast<double>(hist[i]);
            if (value > peak) {{
                peak = value;
                arg_peak = i;
            }}
            if (value != 0.0) {{
                if (low < 0) {{
                    low = i;
                }}
                high = i;
            }}
        }}
        s_value[tid] = peak;
        s_arg[tid] = arg_peak;
        s_low[tid] = low;
        s_high[tid] = high;
        __syncthreads();

        if (tid == 0) {{
            double peak_height = s_value[0];
            long long arg_peak_height = s_arg[0];
            long long arg_low_level = -1;
            long long arg_high_level = -1;
            for (int t = 0; t < BLOCK; ++t) {{
                if (s_value[t] > peak_height) {{
                    peak_height = s_value[t];
                    arg_peak_height = s_arg[t];
                }}
                if (s_low[t] >= 0) {{
                    if (arg_low_level < 0) {{
                        arg_low_level = s_low[t];
                    }}
                    arg_high_level = s_high[t];
                }}
            }}
            s_done = 1;
            s_width = 0;
            s_flip = 0;
            s_peak_height = peak_height;
            if (arg_low_level < 0) {{
                threshold[0] = static_cast<double>(bin_centers[0]);
            }} else if (arg_low_level == arg_high_level) {{
                threshold[0] = static_cast<double>(bin_centers[arg_low_level]);
            }} else {{
                const bool flip = (
                        arg_peak_height - arg_low_level <
                        arg_high_level - arg_peak_height);
                if (flip) {{
                    arg_low_level = n_bins - arg_high_level - 1;
                    arg_peak_height = n_bins - arg_peak_height - 1;
                }}
                const long long width = arg_peak_height - arg_low_level;
                if (width <= 0) {{
                    const long long arg_level = flip
                            ? n_bins - arg_low_level - 1
                            : arg_low_level;
                    threshold[0] = static_cast<double>(bin_centers[arg_level]);
                }} else {{
                    s_done = 0;
                    s_flip = flip ? 1 : 0;
                    s_width = width;
                    s_arg_low_level = arg_low_level;
                }}
            }}
        }}
        __syncthreads();
        if (s_done) {{
            return;
        }}

        // Pass 2: the distance to the line from the peak to the end of the
        // tail, over the width of the tail.
        const double peak_height = s_peak_height;
        const long long width = s_width;
        const long long arg_low_level = s_arg_low_level;
        const bool flip = s_flip != 0;
        const long long chunk2 = (width + BLOCK - 1) / BLOCK;
        const long long begin2 = min(static_cast<long long>(tid) * chunk2, width);
        const long long end2 = min(begin2 + chunk2, width);
        double best_score = -1.0e300;
        long long best_x = -1;
        for (long long x = begin2; x < end2; ++x) {{
            const long long hist_idx = flip
                    ? n_bins - (x + arg_low_level) - 1
                    : x + arg_low_level;
            const double y = static_cast<double>(hist[hist_idx]);
            // The positive normalization factor used by scikit-image does not
            // affect the argmax, so compare the unnormalized score.
            const double score = peak_height * static_cast<double>(x) -
                    static_cast<double>(width) * y;
            if (score > best_score) {{
                best_score = score;
                best_x = x;
            }}
        }}
        s_value[tid] = best_score;
        s_arg[tid] = best_x;
        __syncthreads();

        if (tid == 0) {{
            double best = -1.0e300;
            long long arg_level = arg_low_level;
            for (int t = 0; t < BLOCK; ++t) {{
                if (s_value[t] > best) {{
                    best = s_value[t];
                    arg_level = s_arg[t] + arg_low_level;
                }}
            }}
            if (flip) {{
                arg_level = n_bins - arg_level - 1;
            }}
            threshold[0] = static_cast<double>(bin_centers[arg_level]);
        }}
    }}
    """
    return cp.RawKernel(code, "threshold_triangle")


def _threshold_triangle_gpu(hist, bin_centers):
    hist = cp.ascontiguousarray(hist, dtype=cp.float32)
    bin_centers = cp.ascontiguousarray(bin_centers)
    if bin_centers.dtype.char not in _DTYPE_TO_CTYPE:
        bin_centers = bin_centers.astype(cp.float64)
    threshold = cp.empty((), dtype=cp.float64)
    kernel = _get_threshold_triangle_kernel(bin_centers.dtype)
    kernel(
        (1,),
        (_BLOCK_THREADS,),
        (hist, bin_centers, np.int64(hist.size), threshold),
    )
    return threshold


_mean_std_kernel = cp.ElementwiseKernel(
    "float64 sum, float64 sum_sq, float64 window_size",
    "F mean, F std",
    """
    const double m = sum / window_size;
    const double g2 = sum_sq / window_size;
    mean = static_cast<F>(m);
    const double variance = g2 - m * m;
    std = static_cast<F>(variance > 0.0 ? sqrt(variance) : 0.0);
    """,
    "cucim_mean_std",
)


@cache
def _get_multiotsu_kernels():
    """Kernels of the dynamic programming search of `threshold_multiotsu`."""
    code = r"""
    #define BLOCK 256
    #define PREPARE_BLOCK 512

    // Between-class term of the class made of the bins u, ..., v - 1, from the
    // prefix sums of the counts (P) and of the counts times the bin index (S).
    __device__ __forceinline__ double class_term(
            const double* P, const double* S, long long u, long long v)
    {
        const double weight = P[v] - P[u];
        if (weight > 0.0) {
            const double moment = S[v] - S[u];
            return moment * moment / weight;
        }
        return 0.0;
    }

    // One block of PREPARE_BLOCK threads (every thread owns a chunk of
    // consecutive bins): the prefix sums P and S of the counts and of the
    // counts times the bin index, the best value G_last[u] of the last class
    // (the bins u, ..., n_bins - 1) and the number of non-empty bins.
    extern "C" __global__ void multiotsu_prepare(
            const float* counts, const long long n_bins, double* P, double* S,
            double* G_last, int* n_nonzero)
    {
        __shared__ double s_count[PREPARE_BLOCK];
        __shared__ double s_moment[PREPARE_BLOCK];
        __shared__ int s_nonzero[PREPARE_BLOCK];
        const int tid = threadIdx.x;
        const long long chunk = (n_bins + PREPARE_BLOCK - 1) / PREPARE_BLOCK;
        const long long begin = min(static_cast<long long>(tid) * chunk, n_bins);
        const long long end = min(begin + chunk, n_bins);

        double sum_count = 0.0;
        double sum_moment = 0.0;
        int nonzero = 0;
        for (long long i = begin; i < end; ++i) {
            const double count = static_cast<double>(counts[i]);
            sum_count += count;
            sum_moment += count * static_cast<double>(i);
            nonzero += count != 0.0;
        }
        s_count[tid] = sum_count;
        s_moment[tid] = sum_moment;
        s_nonzero[tid] = nonzero;
        __syncthreads();

        if (tid == 0) {
            double run_count = 0.0;
            double run_moment = 0.0;
            int run_nonzero = 0;
            for (int t = 0; t < PREPARE_BLOCK; ++t) {
                const double c = s_count[t];
                const double m = s_moment[t];
                const int z = s_nonzero[t];
                s_count[t] = run_count;
                s_moment[t] = run_moment;
                run_count += c;
                run_moment += m;
                run_nonzero += z;
            }
            P[0] = 0.0;
            S[0] = 0.0;
            n_nonzero[0] = run_nonzero;
        }
        __syncthreads();

        double p = s_count[tid];
        double m = s_moment[tid];
        for (long long i = begin; i < end; ++i) {
            const double count = static_cast<double>(counts[i]);
            p += count;
            m += count * static_cast<double>(i);
            P[i + 1] = p;
            S[i + 1] = m;
        }
        __syncthreads();

        for (long long u = begin; u < end; ++u) {
            G_last[u] = class_term(P, S, u, n_bins);
        }
    }

    // One stage of the dynamic programming. G_next[v] is the best value of the
    // classes k + 1, ..., M when the class k + 1 starts at bin v, and
    //   G_out[u] = max over v in (u, v_max] of H(u, v) + G_next[v]
    // is the best value of the classes k, ..., M when the class k starts at bin
    // u (the smallest v among equal maxima, so that the thresholds found are the
    // lexicographically smallest ones). The best v of the class that starts at u
    // is non-decreasing with u (the stage matrix is totally monotone, see the
    // docstring of threshold_multiotsu), so the rows are solved by divide and
    // conquer, level by level: at level d the rows i = (2 j + 1) * step - 1
    // (i = u - u_begin, step = 2^(D - 1 - d)) are searched only between the best
    // v of the rows i - step and i + step, which an earlier level solved. One
    // block searches one row.
    __device__ __forceinline__ void dc_row_bounds(
            const int* back, const long long u_begin, const long long n_rows,
            const long long step, const long long v_max, const long long j,
            long long* u, long long* v_low, long long* v_high)
    {
        const long long i = (2 * j + 1) * step - 1;
        *u = u_begin + i;
        *v_low = (j > 0) ? static_cast<long long>(back[u_begin + i - step])
                         : *u + 1;
        *v_high = (i + step < n_rows)
                ? static_cast<long long>(back[u_begin + i + step]) : v_max;
        if (*v_low < *u + 1) {
            *v_low = *u + 1;
        }
    }

    extern "C" __global__ void multiotsu_stage(
            const double* P, const double* S, const double* G_next,
            double* G_out, int* back,
            const long long u_begin, const long long n_rows,
            const long long step, const long long v_max)
    {
        __shared__ double s_best[BLOCK];
        __shared__ long long s_v[BLOCK];
        long long u, v_low, v_high;
        dc_row_bounds(back, u_begin, n_rows, step, v_max, blockIdx.x,
                      &u, &v_low, &v_high);
        const int tid = threadIdx.x;
        double best = -1.0;
        long long best_v = v_low;
        for (long long v = v_low + tid; v <= v_high; v += BLOCK) {
            const double value = class_term(P, S, u, v) + G_next[v];
            if (value > best) {
                best = value;
                best_v = v;
            }
        }
        s_best[tid] = best;
        s_v[tid] = best_v;
        __syncthreads();
        for (int offset = BLOCK / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const double other = s_best[tid + offset];
                const long long other_v = s_v[tid + offset];
                if (other > s_best[tid]
                        || (other == s_best[tid] && other_v < s_v[tid])) {
                    s_best[tid] = other;
                    s_v[tid] = other_v;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            G_out[u] = s_best[0];
            back[u] = static_cast<int>(s_v[0]);
        }
    }

    // The first class always starts at bin 0: one block reduces over its end v.
    extern "C" __global__ void multiotsu_first(
            const double* P, const double* S, const double* G_next,
            int* back, const long long v_max)
    {
        __shared__ double s_best[BLOCK];
        __shared__ long long s_v[BLOCK];
        const int tid = threadIdx.x;
        double best = -1.0;
        long long best_v = 1;
        for (long long v = 1 + tid; v <= v_max; v += BLOCK) {
            const double value = class_term(P, S, 0, v) + G_next[v];
            if (value > best) {
                best = value;
                best_v = v;
            }
        }
        s_best[tid] = best;
        s_v[tid] = best_v;
        __syncthreads();
        if (tid == 0) {
            double b = -1.0;
            long long bv = 1;
            // a value equal to the maximum at a smaller v wins
            for (int t = 0; t < BLOCK; ++t) {
                if (s_best[t] > b || (s_best[t] == b && s_v[t] < bv)) {
                    b = s_best[t];
                    bv = s_v[t];
                }
            }
            back[0] = static_cast<int>(bv);
        }
    }

    // Follow the back pointers from the first class to get the thresholds
    // (the index of the last bin of every class but the last one).
    extern "C" __global__ void multiotsu_thresholds(
            const int* back, const long long L1, const int n_thresholds,
            long long* thresholds)
    {
        long long u = 0;
        for (int k = 0; k < n_thresholds; ++k) {
            const long long v = back[k * L1 + u];
            thresholds[k] = v - 1;
            u = v;
        }
    }
    """
    module = cp.RawModule(code=code)
    return (
        module.get_function("multiotsu_prepare"),
        module.get_function("multiotsu_stage"),
        module.get_function("multiotsu_first"),
        module.get_function("multiotsu_thresholds"),
    )


def _multiotsu_threshold_indices(counts, classes):
    """Indices of the thresholds that maximize the variance between classes.

    This is an exact search by dynamic programming. The sum of the
    between-class variances is a sum of independent terms ``H(u, v) = S(u, v)**2
    / P(u, v)`` of the classes (``P`` and ``S`` are the sums of the counts and
    of the counts times the bin index over the bins ``u, ..., v - 1``), so the
    best value of the last ``m`` classes that start at a given bin can be
    computed from the one for ``m - 1`` classes. The cost is ``O(classes *
    nbins**2)`` instead of the ``O(nbins**(classes - 1))`` of the exhaustive
    search. The best end bin of the class that starts at a given bin is
    non-decreasing with that bin, so every stage is searched by divide and
    conquer, level by level (``O(classes * nbins * log(nbins))`` evaluations).
    All the arithmetic is in float64.

    The histogram must have at least ``classes`` bins. Returns the indices and
    a device array with the number of non-empty bins (if it is not larger than
    ``classes`` the indices are not meaningful).
    """
    prepare, stage, first, thresholds_kernel = _get_multiotsu_kernels()
    n_bins = counts.size
    counts = cp.ascontiguousarray(counts, dtype=cp.float32)
    L1 = n_bins + 1
    P = cp.empty(L1, dtype=cp.float64)
    S = cp.empty(L1, dtype=cp.float64)
    G_next = cp.empty(L1, dtype=cp.float64)
    n_nonzero = cp.empty(1, dtype=cp.int32)
    # back[k, u]: the end of the class k + 1 when it starts at bin u
    back = cp.empty((classes - 1, L1), dtype=cp.int32)
    # prefix sums and the last class: bins u, ..., n_bins - 1
    prepare(
        (1,),
        (512,),
        (counts, np.int64(n_bins), P, S, G_next, n_nonzero),
    )
    block = 256
    # the classes classes - 1, ..., 2 (class k has to leave one bin to each of
    # the classes after it)
    for k in range(classes - 1, 1, -1):
        remaining = classes - k  # the number of classes after class k
        u_begin = k - 1
        u_end = n_bins - remaining  # exclusive
        v_max = n_bins - remaining
        G_out = cp.empty(L1, dtype=cp.float64)
        n_rows = u_end - u_begin
        n_levels = int(n_rows).bit_length()
        for level in range(n_levels):
            step = 1 << (n_levels - 1 - level)
            n_level_rows = (n_rows // step + 1) // 2
            stage(
                (n_level_rows,),
                (block,),
                (
                    P,
                    S,
                    G_next,
                    G_out,
                    back[k - 1],
                    np.int64(u_begin),
                    np.int64(n_rows),
                    np.int64(step),
                    np.int64(v_max),
                ),
            )
        G_next = G_out
    first(
        (1,),
        (block,),
        (P, S, G_next, back[0], np.int64(n_bins - (classes - 1))),
    )
    thresholds = cp.empty(classes - 1, dtype=cp.int64)
    thresholds_kernel(
        (1,),
        (1,),
        (back, np.int64(L1), np.int32(classes - 1), thresholds),
    )
    return thresholds, n_nonzero
