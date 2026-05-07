/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define OP_PERCENTILE 0
#define OP_THRESHOLD 1
#define OP_MEAN 2
#define OP_SUM 3
#define OP_POP 4
#define OP_GRADIENT 5
#define OP_AUTOLEVEL 6
#define OP_ENTROPY 7
#define OP_ENHANCE_CONTRAST 8
#define OP_SUBTRACT_MEAN 9
#define OP_EQUALIZE 10
#define OP_BILATERAL_MEAN 11
#define OP_BILATERAL_POP 12
#define OP_BILATERAL_SUM 13

#ifndef HIST_COUNTER_T
#define HIST_COUNTER_T int
#endif

#ifndef RANK_HIST_OUTPUT_T
#define RANK_HIST_OUTPUT_T unsigned char
#endif

__device__ void histogramPrefixScan256(int* hist, int* scan) {
  int tx = threadIdx.x;
  if (tx < 256) {
    scan[tx] = hist[tx];
  }
  __syncthreads();

  for (int offset = 1; offset < 256; offset <<= 1) {
    int v = 0;
    if (tx >= offset && tx < 256) {
      v = scan[tx - offset];
    }
    __syncthreads();
    if (tx >= offset && tx < 256) {
      scan[tx] += v;
    }
    __syncthreads();
  }
}

__device__ void reduceSum256(int* values) {
  int tx = threadIdx.x;
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tx < stride) {
      values[tx] += values[tx + stride];
    }
    __syncthreads();
  }
}

__device__ void histogramWeightedPrefixScan256(int* hist, int* scan) {
  int tx = threadIdx.x;
  if (tx < 256) {
    scan[tx] = hist[tx] * tx;
  }
  __syncthreads();

  for (int offset = 1; offset < 256; offset <<= 1) {
    int v = 0;
    if (tx >= offset && tx < 256) {
      v = scan[tx - offset];
    }
    __syncthreads();
    if (tx >= offset && tx < 256) {
      scan[tx] += v;
    }
    __syncthreads();
  }
}

__device__ RANK_HIST_OUTPUT_T histogramRankValue(int* hist,
                                                 int* scan,
                                                 int* tmp0,
                                                 int* tmp1,
                                                 double* dtmp,
                                                 int op,
                                                 int window_size,
                                                 double p0,
                                                 double p1,
                                                 double s0,
                                                 double s1,
                                                 unsigned char center) {
  int tx = threadIdx.x;
  __shared__ int result;
  __shared__ int range_start;
  __shared__ int range_end;
#if RANK_HIST_OP == OP_MEAN || RANK_HIST_OP == OP_SUM || RANK_HIST_OP == OP_SUBTRACT_MEAN || RANK_HIST_OP == OP_BILATERAL_MEAN || RANK_HIST_OP == OP_BILATERAL_POP || RANK_HIST_OP == OP_BILATERAL_SUM
  __shared__ int range_start_sum;
  __shared__ int range_end_sum;
#endif

#if RANK_HIST_OP == OP_ENTROPY
  double ent = 0.0;
  if (tx < 256 && hist[tx] > 0) {
    double p = ((double)hist[tx]) / window_size;
    ent = -p * log(p) / 0.6931471805599453;
  }
  dtmp[tx] = ent;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tx < stride) {
      dtmp[tx] += dtmp[tx + stride];
    }
    __syncthreads();
  }
  return static_cast<RANK_HIST_OUTPUT_T>(dtmp[0]);
#else
  op = RANK_HIST_OP;
  histogramPrefixScan256(hist, scan);
  int pop = scan[255];

  if (tx == 0) {
    result = 0;
    int start = max(0, (int)ceil(p0 * pop / 100.0) - 1);
    int end = (int)(p1 * pop / 100.0);
    if (end <= start) {
      end = start + 1;
    }
    if (end > pop) {
      end = pop;
    }
    range_start = start;
    range_end = end;
  }
  __syncthreads();

  if (op == OP_PERCENTILE || op == OP_THRESHOLD) {
    int target;
    if (p0 == 100.0) {
      target = pop - 1;
    } else {
      target = (int)(p0 * pop / 100.0);
      if (target >= pop) {
        target = pop - 1;
      }
    }

    if (tx < 256 && hist[tx] > 0) {
      int bin_start = scan[tx] - hist[tx];
      if (bin_start <= target && scan[tx] > target) {
        result = tx;
      }
    }
    __syncthreads();

    if (op == OP_THRESHOLD) {
      return (center >= result) ? static_cast<RANK_HIST_OUTPUT_T>(255)
                                : static_cast<RANK_HIST_OUTPUT_T>(0);
    }
    return static_cast<RANK_HIST_OUTPUT_T>(result);
  }

#if RANK_HIST_OP == OP_EQUALIZE
  return static_cast<RANK_HIST_OUTPUT_T>(255.0 * ((double)scan[center]) / pop);
#endif

#if RANK_HIST_OP == OP_MEAN || RANK_HIST_OP == OP_SUM || RANK_HIST_OP == OP_SUBTRACT_MEAN || RANK_HIST_OP == OP_BILATERAL_MEAN || RANK_HIST_OP == OP_BILATERAL_POP || RANK_HIST_OP == OP_BILATERAL_SUM
  histogramWeightedPrefixScan256(hist, tmp1);
  if (tx == 0) {
    range_start_sum = 0;
    range_end_sum = 0;
  }
  __syncthreads();

#if RANK_HIST_OP == OP_BILATERAL_MEAN || RANK_HIST_OP == OP_BILATERAL_POP || RANK_HIST_OP == OP_BILATERAL_SUM
  if (tx == 0) {
    int start_bin = max(0, (int)floor((double)center - s1) + 1);
    int stop_bin = min(256, (int)ceil((double)center + s0));
    if (stop_bin <= start_bin) {
      range_start = 0;
      range_end = 0;
      range_start_sum = 0;
      range_end_sum = 0;
    } else {
      range_start = (start_bin > 0) ? scan[start_bin - 1] : 0;
      range_end = scan[stop_bin - 1];
      range_start_sum = (start_bin > 0) ? tmp1[start_bin - 1] : 0;
      range_end_sum = tmp1[stop_bin - 1];
    }
  }
  __syncthreads();
#else
  if (tx < 256 && hist[tx] > 0) {
    int bin_end = scan[tx];
    int bin_start = bin_end - hist[tx];
    int weighted_end = tmp1[tx];
    int weighted_start = weighted_end - hist[tx] * tx;

    if (range_start > 0 && bin_start < range_start &&
        bin_end >= range_start) {
      range_start_sum = weighted_start + (range_start - bin_start) * tx;
    }
    if (range_end > 0 && bin_start < range_end && bin_end >= range_end) {
      range_end_sum = weighted_start + (range_end - bin_start) * tx;
    }
  }
  __syncthreads();
#endif

  int selected_count_total = range_end - range_start;
  int selected_sum_total = range_end_sum - range_start_sum;
  if (op == OP_BILATERAL_POP) {
    return static_cast<RANK_HIST_OUTPUT_T>(selected_count_total);
  }
  if (selected_count_total <= 0) {
    return static_cast<RANK_HIST_OUTPUT_T>(0);
  }
  if (op == OP_BILATERAL_MEAN) {
    return static_cast<RANK_HIST_OUTPUT_T>(
        ((double)selected_sum_total) / selected_count_total);
  }
  if (op == OP_BILATERAL_SUM) {
    return static_cast<RANK_HIST_OUTPUT_T>(selected_sum_total);
  }
  if (op == OP_MEAN) {
    return static_cast<RANK_HIST_OUTPUT_T>(
        ((double)selected_sum_total) / selected_count_total);
  }
  if (op == OP_SUBTRACT_MEAN) {
    double mean = ((double)selected_sum_total) / selected_count_total;
    return static_cast<RANK_HIST_OUTPUT_T>(
        ((double)center - mean) * 0.5 + 128.0);
  }
  return static_cast<RANK_HIST_OUTPUT_T>(selected_sum_total);
#endif

  int selected_count = 0;
  int selected_sum = 0;
  if (tx < 256) {
    int bin_start = scan[tx] - hist[tx];
    int bin_end = scan[tx];
    selected_count = min(bin_end, range_end) - max(bin_start, range_start);
    if (selected_count < 0) {
      selected_count = 0;
    }
    selected_sum = selected_count * tx;
  }

  if (op == OP_POP) {
    double low = p0 * pop / 100.0;
    double high = p1 * pop / 100.0;
    int count = 0;
    if (tx < 256 && hist[tx] > 0 &&
        (double)scan[tx] >= low && (double)scan[tx] <= high) {
      count = hist[tx];
    }
    tmp0[tx] = count;
    __syncthreads();
    reduceSum256(tmp0);
    return static_cast<RANK_HIST_OUTPUT_T>(tmp0[0]);
  }

  if (op == OP_GRADIENT || op == OP_AUTOLEVEL ||
      op == OP_ENHANCE_CONTRAST) {
    tmp0[tx] = selected_count > 0 ? tx : 255;
    tmp1[tx] = selected_count > 0 ? tx : 0;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
      if (tx < stride) {
        tmp0[tx] = min(tmp0[tx], tmp0[tx + stride]);
        tmp1[tx] = max(tmp1[tx], tmp1[tx + stride]);
      }
      __syncthreads();
    }
    if (op == OP_GRADIENT) {
      return static_cast<RANK_HIST_OUTPUT_T>(tmp1[0] - tmp0[0]);
    }

    int min_val = tmp0[0];
    int max_val = tmp1[0];
    if (op == OP_ENHANCE_CONTRAST) {
      return (max_val - center < center - min_val)
                 ? static_cast<RANK_HIST_OUTPUT_T>(max_val)
                 : static_cast<RANK_HIST_OUTPUT_T>(min_val);
    }

    int clamped = min(max((int)center, min_val), max_val);
    int delta = max_val - min_val;
    if (delta > 0) {
      return static_cast<RANK_HIST_OUTPUT_T>(
          ((double)(clamped - min_val) / delta) * 255.0);
    }
    return static_cast<RANK_HIST_OUTPUT_T>(0);
  }

#if RANK_HIST_OP != OP_MEAN && RANK_HIST_OP != OP_SUM && RANK_HIST_OP != OP_SUBTRACT_MEAN && RANK_HIST_OP != OP_BILATERAL_MEAN && RANK_HIST_OP != OP_BILATERAL_POP && RANK_HIST_OP != OP_BILATERAL_SUM
  tmp0[tx] = selected_count;
  tmp1[tx] = selected_sum;
  __syncthreads();
  reduceSum256(tmp0);
  reduceSum256(tmp1);

  if (op == OP_MEAN) {
    return static_cast<RANK_HIST_OUTPUT_T>(((double)tmp1[0]) / tmp0[0]);
  }
  if (op == OP_SUBTRACT_MEAN) {
    double mean = ((double)tmp1[0]) / tmp0[0];
    return static_cast<RANK_HIST_OUTPUT_T>(
        ((double)center - mean) * 0.5 + 128.0);
  }
  return static_cast<RANK_HIST_OUTPUT_T>(tmp1[0]);
#endif
#endif
}

extern "C" __global__ void cuRankHistogram2DUint8(
    const unsigned char* src,
    RANK_HIST_OUTPUT_T* dest,
    HIST_COUNTER_T* histPar,
    int r0,
    int r1,
    double p0,
    double p1,
    double s0,
    double s1,
    int op,
    int window_size,
    int rows,
    int cols) {
  __shared__ int H[256];
  __shared__ int Hscan[256];
  __shared__ int tmp0[256];
  __shared__ int tmp1[256];
  __shared__ double dtmp[256];

  int tx = threadIdx.x;
  int out_rows = rows - 2 * r0;
  int rows_per_block = (out_rows + gridDim.x - 1) / gridDim.x;
  int start_out = blockIdx.x * rows_per_block;
  int stop_out = min(out_rows, start_out + rows_per_block);

  if (start_out >= stop_out) {
    return;
  }

  int start_row = r0 + start_out;
  int stop_row = r0 + stop_out;
  HIST_COUNTER_T* hist = histPar + blockIdx.x * cols * 256;

  for (int col = tx; col < cols; col += blockDim.x) {
    HIST_COUNTER_T* col_hist = hist + col * 256;
    for (int row = start_row - r0; row <= start_row + r0; row++) {
      col_hist[src[row * cols + col]]++;
    }
  }
  __syncthreads();

  for (int row = start_row; row < stop_row; row++) {
    if (tx < 256) {
      int total = 0;
      for (int col = 0; col <= 2 * r1; col++) {
        total += (int)hist[col * 256 + tx];
      }
      H[tx] = total;
    }
    __syncthreads();

    for (int col = r1; col < cols - r1; col++) {
      unsigned char center = src[row * cols + col];
      RANK_HIST_OUTPUT_T value = histogramRankValue(
          H, Hscan, tmp0, tmp1, dtmp, op, window_size, p0, p1, s0, s1, center);

      if (tx == 0) {
        dest[row * cols + col] = value;
      }
      __syncthreads();

      if (col < cols - r1 - 1 && tx < 256) {
        int sub_col = col - r1;
        int add_col = col + r1 + 1;
        H[tx] += (int)hist[add_col * 256 + tx] -
                 (int)hist[sub_col * 256 + tx];
      }
      __syncthreads();
    }

    if (row < stop_row - 1) {
      int sub_row = row - r0;
      int add_row = row + r0 + 1;
      for (int col = tx; col < cols; col += blockDim.x) {
        HIST_COUNTER_T* col_hist = hist + col * 256;
        col_hist[src[sub_row * cols + col]]--;
        col_hist[src[add_row * cols + col]]++;
      }
      __syncthreads();
    }
  }
}
