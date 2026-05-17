/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
Apache Software License 2.0

Copyright (c) 2020, Omar Elamin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


adapted from the following file:
https://github.com/rosalindfranklininstitute/cuda-slic/blob/master/src/cuda_slic/kernels/slic3d_template.cu

original license:
https://github.com/rosalindfranklininstitute/cuda-slic/blob/master/README.md


refactoring/update for cuCIM (c) 2025, Gregory Lee
- removed Jinja2 template code (prepend a string with any needed #define statements from Python)
- comment out unused init_clusters kernel
- add __force_inline__ to slic_distance
- minor stylistic/performance updates

*/

/*
Indexing:
idx = pixel/voxel index in cartesian coordinates
cidx = center index in cartesian coordinates

linear_idx = pixel/voxel index in flat array
linear_cidx = center index in flat array

Center Stride:
c_stride = number_of_features + image_dimention
center_addr = linear_cidx * c_stride

image has shape (z, y, x) with C-contiguous layout:
z_stride = image_shape.y * image_shape.x
y_stride = image_shape.x
x_stride = 1

Transformations 3D:
linear_idx = idx.z * z_stride + idx.y * y_stride + idx.x
pixel_addr = linear_idx * N_PIXEL_FEATURES

idx.z = linear_idx / z_stride
plane_idx = linear_idx % z_stride
idx.y = plane_idx / y_stride
idx.x = plane_idx % y_stride

Transformations 2D:
linear_idx = idx.y * y_stride + idx.x
pixel_addr = linear_idx * N_PIXEL_FEATURES

idx.y = linear_idx / y_stride
idx.x = linear_idx % y_stride

CuPy prepends the following defines in slic_superpixels.py:
#define N_FEATURES { n_features }
*/

#define __min(a, b) (((a) < (b)) ? (a) : (b))
#define __max(a, b) (((a) >= (b)) ? (a) : (b))

#ifndef N_PIXEL_FEATURES
#    define N_PIXEL_FEATURES 3 // number of features per pixel (e.g. 3 for RGB)
#endif

#ifndef FLOAT_DTYPE
#    define FLOAT_DTYPE double
#endif

#ifndef INTERNAL_FLOAT_DTYPE
#    define INTERNAL_FLOAT_DTYPE FLOAT_DTYPE
#endif

#ifndef SUM_DTYPE
#    define SUM_DTYPE FLOAT_DTYPE
#endif

#define CENTER_LIMIT ((FLOAT_DTYPE)1.0e30)
#define DIST_LIMIT ((INTERNAL_FLOAT_DTYPE)1.0e30)

#ifndef START_LABEL
#    define START_LABEL 1 // starting label (must be 0 or 1)
#endif

#ifndef PIXELS_PER_THREAD
#    define PIXELS_PER_THREAD 1
#endif

#ifndef SLIC_ZERO
#    define SLIC_ZERO 0
#endif

#define CENTER_STRIDE (N_PIXEL_FEATURES + 2)

__forceinline__ __device__ INTERNAL_FLOAT_DTYPE slic_distance(const int2 idx,
                                                              const FLOAT_DTYPE* __restrict__ pixel,
                                                              const long linear_cidx,
                                                              const long center_addr,
                                                              const FLOAT_DTYPE* __restrict__ centers,
                                                              const FLOAT_DTYPE* __restrict__ spacing,
                                                              INTERNAL_FLOAT_DTYPE inv_ss,
                                                              const FLOAT_DTYPE* __restrict__ max_dist_color)

{
    // Color diff
    INTERNAL_FLOAT_DTYPE color_diff = 0;
#if N_PIXEL_FEATURES <= 8
#    pragma unroll
#else
#    pragma unroll 8
#endif
    for (int w = 0; w < N_PIXEL_FEATURES; w++)
    {
        INTERNAL_FLOAT_DTYPE d = static_cast<INTERNAL_FLOAT_DTYPE>(pixel[w] - centers[center_addr + w]);
        color_diff += d * d;
    }
#if SLIC_ZERO
    color_diff /= static_cast<INTERNAL_FLOAT_DTYPE>(max_dist_color[linear_cidx]);
#endif

    // Position diff

    INTERNAL_FLOAT_DTYPE pd_y = static_cast<INTERNAL_FLOAT_DTYPE>(
        (static_cast<FLOAT_DTYPE>(idx.y) - centers[center_addr + N_PIXEL_FEATURES]) * spacing[0]);
    INTERNAL_FLOAT_DTYPE pd_x = static_cast<INTERNAL_FLOAT_DTYPE>(
        (static_cast<FLOAT_DTYPE>(idx.x) - centers[center_addr + N_PIXEL_FEATURES + 1]) * spacing[1]);

    INTERNAL_FLOAT_DTYPE position_diff = pd_y * pd_y + pd_x * pd_x;
    return color_diff + position_diff * inv_ss;
}

__global__ void expectation(const FLOAT_DTYPE* __restrict__ data,
                            const FLOAT_DTYPE* __restrict__ centers,
                            unsigned int* __restrict__ labels,
                            int im_shape_y,
                            int im_shape_x,
                            int sp_shape_y,
                            int sp_shape_x,
                            int sp_grid_y,
                            int sp_grid_x,
                            const FLOAT_DTYPE* __restrict__ spacing,
                            const FLOAT_DTYPE* __restrict__ ss,
                            const FLOAT_DTYPE* __restrict__ max_dist_color)

{
    int2 idx;
    idx.y = threadIdx.x + (blockIdx.x * blockDim.x);
    idx.x = threadIdx.y + (blockIdx.y * blockDim.y);

    if (idx.x >= im_shape_x || idx.y >= im_shape_y)
    {
        return;
    }

    long y_stride = im_shape_x;

    const long linear_idx = idx.y * y_stride + idx.x;
    const long pixel_addr = linear_idx * N_PIXEL_FEATURES;

    FLOAT_DTYPE pixel[N_PIXEL_FEATURES];
#if N_PIXEL_FEATURES <= 8
#    pragma unroll
#else
#    pragma unroll 8
#endif
    for (int w = 0; w < N_PIXEL_FEATURES; w++)
    {
        pixel[w] = data[pixel_addr + w];
    }
    INTERNAL_FLOAT_DTYPE inv_ss = static_cast<INTERNAL_FLOAT_DTYPE>(1) / static_cast<INTERNAL_FLOAT_DTYPE>(*ss);

    int2 cidx;
    long closest_linear_cidx = 0 - START_LABEL;

    // approx center grid position
    cidx.y = max(0, min(idx.y / sp_shape_y, sp_grid_y - 1));
    cidx.x = max(0, min(idx.x / sp_shape_x, sp_grid_x - 1));

    const int c_stride = N_PIXEL_FEATURES + 2;
    INTERNAL_FLOAT_DTYPE minimum_distance = DIST_LIMIT;
    const int R = 2;
    const int y_start = max(cidx.y - R, 0);
    const int y_end = min(cidx.y + R, sp_grid_y);
    const int x_start = max(cidx.x - R, 0);
    const int x_end = min(cidx.x + R, sp_grid_x);
    for (int j = y_start; j < y_end; j++)
    {
        long offset_y = j * sp_grid_x;
        for (int i = x_start; i < x_end; i++)
        {
            long iter_linear_cidx = offset_y + i;
            long iter_center_addr = iter_linear_cidx * c_stride;

            if (centers[iter_center_addr] == CENTER_LIMIT)
            {
                continue;
            }

            INTERNAL_FLOAT_DTYPE dist =
                slic_distance(idx, pixel, iter_linear_cidx, iter_center_addr, centers, spacing, inv_ss, max_dist_color);

            // Wrapup
            if (dist < minimum_distance)
            {
                minimum_distance = dist;
                closest_linear_cidx = iter_linear_cidx;
            }
        }
    }

    labels[linear_idx] = closest_linear_cidx + START_LABEL;
}

__global__ void update_max_dist_color(const FLOAT_DTYPE* __restrict__ data,
                                      const unsigned int* __restrict__ labels,
                                      const FLOAT_DTYPE* __restrict__ centers,
                                      int im_shape_y,
                                      int im_shape_x,
                                      long n_clusters,
                                      FLOAT_DTYPE* __restrict__ max_dist_color)
{
    const unsigned long thread_idx = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned long start_idx = thread_idx * PIXELS_PER_THREAD;
    const unsigned long n_pixels = (unsigned long)im_shape_y * im_shape_x;

    for (int k = 0; k < PIXELS_PER_THREAD; k++)
    {
        const unsigned long linear_idx = start_idx + k;
        if (linear_idx >= n_pixels)
        {
            break;
        }

        const long linear_cidx = (long)labels[linear_idx] - START_LABEL;
        if (linear_cidx < 0 || linear_cidx >= n_clusters)
        {
            continue;
        }

        const long center_addr = linear_cidx * CENTER_STRIDE;
        if (centers[center_addr] == CENTER_LIMIT)
        {
            continue;
        }

        const unsigned long pixel_addr = linear_idx * N_PIXEL_FEATURES;
        FLOAT_DTYPE dist_color = 0;
#if N_PIXEL_FEATURES <= 8
#    pragma unroll
#else
#    pragma unroll 8
#endif
        for (int w = 0; w < N_PIXEL_FEATURES; w++)
        {
            FLOAT_DTYPE d = data[pixel_addr + w] - centers[center_addr + w];
            dist_color += d * d;
        }
        atomicMax(&max_dist_color[linear_cidx], dist_color);
    }
}

__global__ void reset_accumulators(SUM_DTYPE* __restrict__ center_sums,
                                   unsigned int* __restrict__ center_counts,
                                   long n_clusters)
{
    const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (linear_cidx >= n_clusters)
    {
        return;
    }

    const long center_addr = linear_cidx * CENTER_STRIDE;
    for (int w = 0; w < CENTER_STRIDE; w++)
    {
        center_sums[center_addr + w] = 0;
    }
    center_counts[linear_cidx] = 0;
}

__global__ void accumulate_centers(const FLOAT_DTYPE* __restrict__ data,
                                   const unsigned int* __restrict__ labels,
                                   const FLOAT_DTYPE* __restrict__ centers,
                                   int im_shape_y,
                                   int im_shape_x,
                                   int sp_shape_y,
                                   int sp_shape_x,
                                   long n_clusters,
                                   SUM_DTYPE* __restrict__ center_sums,
                                   unsigned int* __restrict__ center_counts)
{
    const unsigned long thread_idx = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned long start_idx = thread_idx * PIXELS_PER_THREAD;
    const unsigned long n_pixels = (unsigned long)im_shape_y * im_shape_x;

    unsigned int encountered_labels[PIXELS_PER_THREAD];
    unsigned int local_counts[PIXELS_PER_THREAD];
    SUM_DTYPE local_sums[PIXELS_PER_THREAD * CENTER_STRIDE];

    for (int i = 0; i < PIXELS_PER_THREAD; i++)
    {
        encountered_labels[i] = 0;
        local_counts[i] = 0;
    }
    for (int i = 0; i < PIXELS_PER_THREAD * CENTER_STRIDE; i++)
    {
        local_sums[i] = 0;
    }

    int n_encountered = 0;
    for (int k = 0; k < PIXELS_PER_THREAD; k++)
    {
        const unsigned long linear_idx = start_idx + k;
        if (linear_idx >= n_pixels)
        {
            break;
        }

        const unsigned int label = labels[linear_idx];
        const long linear_cidx = (long)label - START_LABEL;
        if (linear_cidx < 0 || linear_cidx >= n_clusters)
        {
            continue;
        }

        int offset = -1;
        for (int j = 0; j < n_encountered; j++)
        {
            if (encountered_labels[j] == label)
            {
                offset = j;
                break;
            }
        }
        if (offset < 0)
        {
            offset = n_encountered;
            encountered_labels[offset] = label;
            n_encountered += 1;
        }

        const unsigned long pixel_addr = linear_idx * N_PIXEL_FEATURES;
        const long base = offset * CENTER_STRIDE;
#if N_PIXEL_FEATURES <= 8
#    pragma unroll
#else
#    pragma unroll 8
#endif
        for (int w = 0; w < N_PIXEL_FEATURES; w++)
        {
            local_sums[base + w] += data[pixel_addr + w];
        }

        const int y = linear_idx / im_shape_x;
        const int x = linear_idx - (unsigned long)y * im_shape_x;

        const long center_addr = linear_cidx * CENTER_STRIDE;
        const int center_y = (int)centers[center_addr + N_PIXEL_FEATURES];
        const int center_x = (int)centers[center_addr + N_PIXEL_FEATURES + 1];
        const float ratio = 2.0f;
        const int from_y = __max(center_y - sp_shape_y * ratio, 0);
        const int from_x = __max(center_x - sp_shape_x * ratio, 0);
        const int to_y = __min(center_y + sp_shape_y * ratio, im_shape_y);
        const int to_x = __min(center_x + sp_shape_x * ratio, im_shape_x);
        if (y < from_y || y >= to_y || x < from_x || x >= to_x)
        {
            continue;
        }

        local_sums[base + N_PIXEL_FEATURES] += y;
        local_sums[base + N_PIXEL_FEATURES + 1] += x;
        local_counts[offset] += 1;
    }

    for (int j = 0; j < n_encountered; j++)
    {
        const long linear_cidx = (long)encountered_labels[j] - START_LABEL;
        const long center_addr = linear_cidx * CENTER_STRIDE;
        const long base = j * CENTER_STRIDE;
        for (int w = 0; w < CENTER_STRIDE; w++)
        {
            atomicAdd(&center_sums[center_addr + w], local_sums[base + w]);
        }
        atomicAdd(&center_counts[linear_cidx], local_counts[j]);
    }
}

__global__ void normalize_centers(FLOAT_DTYPE* __restrict__ centers,
                                  const SUM_DTYPE* __restrict__ center_sums,
                                  const unsigned int* __restrict__ center_counts,
                                  long n_clusters)
{
    const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (linear_cidx >= n_clusters)
    {
        return;
    }

    const long center_addr = linear_cidx * CENTER_STRIDE;
    const unsigned int count = center_counts[linear_cidx];
    if (count > 0)
    {
        const FLOAT_DTYPE inv_count = static_cast<FLOAT_DTYPE>(1) / static_cast<FLOAT_DTYPE>(count);
        for (int w = 0; w < CENTER_STRIDE; w++)
        {
            centers[center_addr + w] = static_cast<FLOAT_DTYPE>(center_sums[center_addr + w]) * inv_count;
        }
    }
    else
    {
        centers[center_addr] = CENTER_LIMIT;
    }
}

__global__ void maximization(const FLOAT_DTYPE* __restrict__ data,
                             const unsigned int* __restrict__ labels,
                             FLOAT_DTYPE* __restrict__ centers,
                             int im_shape_y,
                             int im_shape_x,
                             int sp_shape_y,
                             int sp_shape_x,
                             long n_clusters)
{
    const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int c_stride = N_PIXEL_FEATURES + 2;
    const long center_addr = linear_cidx * c_stride;

    if (linear_cidx >= n_clusters)
    {
        return;
    }

    int2 cidx;
    cidx.y = (int)centers[center_addr + N_PIXEL_FEATURES];
    cidx.x = (int)centers[center_addr + N_PIXEL_FEATURES + 1];

    float ratio = 2.0f;

    int2 from;
    from.y = __max(cidx.y - sp_shape_y * ratio, 0);
    from.x = __max(cidx.x - sp_shape_x * ratio, 0);

    int2 to;
    to.y = __min(cidx.y + sp_shape_y * ratio, im_shape_y);
    to.x = __min(cidx.x + sp_shape_x * ratio, im_shape_x);

    FLOAT_DTYPE f[c_stride];
    for (int k = 0; k < c_stride; k++)
    {
        f[k] = 0;
    }

    long y_stride = im_shape_x;

    long count = 0;
    int2 p;
    for (p.y = from.y; p.y < to.y; p.y++)
    {
        long offset_y = p.y * y_stride;
        long linear_idx = offset_y + from.x;
        long pixel_addr = linear_idx * N_PIXEL_FEATURES;
        for (p.x = from.x; p.x < to.x; p.x++)
        {
            if (labels[linear_idx] == linear_cidx + START_LABEL)
            {
#if N_PIXEL_FEATURES <= 8
#    pragma unroll
#else
#    pragma unroll 8
#endif
                for (int w = 0; w < N_PIXEL_FEATURES; w++)
                {
                    f[w] += data[pixel_addr + w];
                }
                f[N_PIXEL_FEATURES] += p.y;
                f[N_PIXEL_FEATURES + 1] += p.x;
                count += 1;
            }
            linear_idx += 1;
            pixel_addr += N_PIXEL_FEATURES;
        }
    }

    if (count > 0)
    {
        for (int w = 0; w < c_stride; w++)
        {
            centers[center_addr + w] = f[w] / count;
        }
    }
    else
    {
        centers[center_addr] = CENTER_LIMIT;
    }
}
