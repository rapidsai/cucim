/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright (c) 2020, Omar Elamin
 * SPDX-License-Identifier: Apache-2.0
 */

/*
started out as an adaptation of:
https://github.com/rosalindfranklininstitute/cuda-slic/blob/master/src/cuda_slic/kernels/slic3d_template.cu

refactoring/update for cuCIM (c) 2026, Gregory Lee
- removed Jinja2 template code (prepend a string with any needed #define statements from Python)
- performance refactor (use float32 distance, use __restrict__, etc.)
- add SLIC0 implementation
- and maskedSLIC implementation

Note on indexing conventions used here:

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

CuPy prepends the following defines in slic_superpixels.py:

  #define N_PIXEL_FEATURES {n_features}
  #define START_LABEL {start_label}
  #define FLOAT_DTYPE {"double" if image.dtype == np.float64 else "float"}
  #define INTERNAL_FLOAT_DTYPE {"double" if image.dtype == np.float64 else "float"}
  #define PIXELS_PER_THREAD {maximization_pixels_per_thread}
  #define SLIC_ZERO {1 if slic_zero else 0}
  #define USE_MASK {1 if use_mask else 0}
  #define IGNORE_COLOR {1 if ignore_color else 0}
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

#ifndef USE_MASK
#    define USE_MASK 0
#endif

#ifndef IGNORE_COLOR
#    define IGNORE_COLOR 0
#endif

#define CENTER_STRIDE (N_PIXEL_FEATURES + 3)

__forceinline__ __device__ INTERNAL_FLOAT_DTYPE slic_distance(const int3 idx,
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
#if !IGNORE_COLOR
#    if N_PIXEL_FEATURES <= 8
#        pragma unroll
#    else
#        pragma unroll 8
#    endif
    for (int w = 0; w < N_PIXEL_FEATURES; w++)
    {
        INTERNAL_FLOAT_DTYPE d = static_cast<INTERNAL_FLOAT_DTYPE>(pixel[w] - centers[center_addr + w]);
        color_diff += d * d;
    }
#    if SLIC_ZERO
    color_diff /= static_cast<INTERNAL_FLOAT_DTYPE>(max_dist_color[linear_cidx]);
#    endif
#endif

    // Position diff
    INTERNAL_FLOAT_DTYPE pd_z = static_cast<INTERNAL_FLOAT_DTYPE>(
        (static_cast<FLOAT_DTYPE>(idx.z) - centers[center_addr + N_PIXEL_FEATURES]) * spacing[0]);
    INTERNAL_FLOAT_DTYPE pd_y = static_cast<INTERNAL_FLOAT_DTYPE>(
        (static_cast<FLOAT_DTYPE>(idx.y) - centers[center_addr + N_PIXEL_FEATURES + 1]) * spacing[1]);
    INTERNAL_FLOAT_DTYPE pd_x = static_cast<INTERNAL_FLOAT_DTYPE>(
        (static_cast<FLOAT_DTYPE>(idx.x) - centers[center_addr + N_PIXEL_FEATURES + 2]) * spacing[2]);

    INTERNAL_FLOAT_DTYPE position_diff = pd_z * pd_z + pd_y * pd_y + pd_x * pd_x;
    return color_diff + position_diff * inv_ss;
}

// __global__
// void init_clusters(float* centers,
//                    int sp_shape_z, int sp_shape_y, int sp_shape_x,
//                    int sp_grid_z, int sp_grid_y, int sp_grid_x,
//                    long n_clusters)
// {
//     const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);

//     if ( linear_cidx >= n_clusters ) {
//         return;
//     }

//     // calculating the (0,0,0) index of each superpixel block
//     // using linear to cartesian index transformation
//     int3 cidx;
//     int plane_size = sp_grid_y * sp_grid_x;
//     cidx.z = linear_cidx / plane_size;
//     int plane_idx = linear_cidx % plane_size;
//     cidx.y = plane_idx / sp_grid_x;
//     cidx.x = plane_idx % sp_grid_x;

//     // centering index into middle of superpixel block
//     cidx.z = cidx.z * sp_shape_z + sp_shape_z / 2;
//     cidx.y = cidx.y * sp_shape_y + sp_shape_y / 2;
//     cidx.x = cidx.x * sp_shape_x + sp_shape_x / 2;

//     //saving cluster center positions
//     // note: the color is not initialized, but is kept at zero.
//     const int c_stride = N_PIXEL_FEATURES + 3;
//     centers[linear_cidx * c_stride + N_PIXEL_FEATURES] = cidx.z;
//     centers[linear_cidx * c_stride + N_PIXEL_FEATURES + 1] = cidx.y;
//     centers[linear_cidx * c_stride + N_PIXEL_FEATURES + 2] = cidx.x;
// }

__global__ void expectation(const FLOAT_DTYPE* __restrict__ data,
                            const FLOAT_DTYPE* __restrict__ centers,
                            int* __restrict__ labels,
                            int im_shape_z,
                            int im_shape_y,
                            int im_shape_x,
                            int sp_shape_z,
                            int sp_shape_y,
                            int sp_shape_x,
                            int sp_grid_z,
                            int sp_grid_y,
                            int sp_grid_x,
                            const FLOAT_DTYPE* __restrict__ spacing,
                            const FLOAT_DTYPE* __restrict__ ss,
                            const FLOAT_DTYPE* __restrict__ max_dist_color)

{
    int3 idx;
    idx.z = threadIdx.x + (blockIdx.x * blockDim.x);
    idx.y = threadIdx.y + (blockIdx.y * blockDim.y);
    idx.x = threadIdx.z + (blockIdx.z * blockDim.z);

    if (idx.x >= im_shape_x || idx.y >= im_shape_y || idx.z >= im_shape_z)
    {
        return;
    }

    long z_stride = im_shape_x * im_shape_y;
    long y_stride = im_shape_x;

    const long linear_idx = idx.z * z_stride + idx.y * y_stride + idx.x;
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

    int3 cidx;
    long closest_linear_cidx = 0;

    // approx center grid position
    cidx.z = max(0, min(idx.z / sp_shape_z, sp_grid_z - 1));
    cidx.y = max(0, min(idx.y / sp_shape_y, sp_grid_y - 1));
    cidx.x = max(0, min(idx.x / sp_shape_x, sp_grid_x - 1));

    const int c_stride = N_PIXEL_FEATURES + 3;
    INTERNAL_FLOAT_DTYPE minimum_distance = DIST_LIMIT;
    const int R = 2;
    const int z_start = max(cidx.z - R, 0);
    const int z_end = min(cidx.z + R, sp_grid_z);
    const int y_start = max(cidx.y - R, 0);
    const int y_end = min(cidx.y + R, sp_grid_y);
    const int x_start = max(cidx.x - R, 0);
    const int x_end = min(cidx.x + R, sp_grid_x);
    for (int k = z_start; k < z_end; k++)
    {
        long offset_z = k * sp_grid_y * sp_grid_x;
        for (int j = y_start; j < y_end; j++)
        {
            long offset_y = offset_z + j * sp_grid_x;
            for (int i = x_start; i < x_end; i++)
            {
                long iter_linear_cidx = offset_y + i;
                long iter_center_addr = iter_linear_cidx * c_stride;

                if (centers[iter_center_addr] == CENTER_LIMIT)
                {
                    continue;
                }

                INTERNAL_FLOAT_DTYPE dist = slic_distance(
                    idx, pixel, iter_linear_cidx, iter_center_addr, centers, spacing, inv_ss, max_dist_color);

                // Wrapup
                if (dist < minimum_distance)
                {
                    minimum_distance = dist;
                    closest_linear_cidx = iter_linear_cidx;
                }
            }
        }
    }

    labels[linear_idx] = closest_linear_cidx + START_LABEL;
}

#if USE_MASK
__global__ void expectation_mask(const FLOAT_DTYPE* __restrict__ data,
                                 const FLOAT_DTYPE* __restrict__ centers,
                                 int* __restrict__ labels,
                                 int im_shape_z,
                                 int im_shape_y,
                                 int im_shape_x,
                                 int sp_shape_z,
                                 int sp_shape_y,
                                 int sp_shape_x,
                                 const FLOAT_DTYPE* __restrict__ spacing,
                                 const FLOAT_DTYPE* __restrict__ ss,
                                 const FLOAT_DTYPE* __restrict__ max_dist_color,
                                 const unsigned char* __restrict__ mask,
                                 long n_clusters)
{
    int3 idx;
    idx.z = threadIdx.x + (blockIdx.x * blockDim.x);
    idx.y = threadIdx.y + (blockIdx.y * blockDim.y);
    idx.x = threadIdx.z + (blockIdx.z * blockDim.z);

    if (idx.x >= im_shape_x || idx.y >= im_shape_y || idx.z >= im_shape_z)
    {
        return;
    }

    const long linear_idx = idx.z * im_shape_y * im_shape_x + idx.y * im_shape_x + idx.x;
    if (!mask[linear_idx])
    {
        labels[linear_idx] = START_LABEL - 1;
        return;
    }

    const long pixel_addr = linear_idx * N_PIXEL_FEATURES;
    FLOAT_DTYPE pixel[N_PIXEL_FEATURES];
#    if N_PIXEL_FEATURES <= 8
#        pragma unroll
#    else
#        pragma unroll 8
#    endif
    for (int w = 0; w < N_PIXEL_FEATURES; w++)
    {
        pixel[w] = data[pixel_addr + w];
    }
    INTERNAL_FLOAT_DTYPE inv_ss = static_cast<INTERNAL_FLOAT_DTYPE>(1) / static_cast<INTERNAL_FLOAT_DTYPE>(*ss);

    INTERNAL_FLOAT_DTYPE minimum_distance = DIST_LIMIT;
    long closest_linear_cidx = -1;
    for (long linear_cidx = 0; linear_cidx < n_clusters; linear_cidx++)
    {
        long center_addr = linear_cidx * CENTER_STRIDE;
        if (centers[center_addr] == CENTER_LIMIT)
        {
            continue;
        }
        const int center_z = (int)centers[center_addr + N_PIXEL_FEATURES];
        const int center_y = (int)centers[center_addr + N_PIXEL_FEATURES + 1];
        const int center_x = (int)centers[center_addr + N_PIXEL_FEATURES + 2];
        if (idx.z < center_z - 2 * sp_shape_z || idx.z > center_z + 2 * sp_shape_z || idx.y < center_y - 2 * sp_shape_y ||
            idx.y > center_y + 2 * sp_shape_y || idx.x < center_x - 2 * sp_shape_x || idx.x > center_x + 2 * sp_shape_x)
        {
            continue;
        }
        INTERNAL_FLOAT_DTYPE dist =
            slic_distance(idx, pixel, linear_cidx, center_addr, centers, spacing, inv_ss, max_dist_color);
        if (dist < minimum_distance)
        {
            minimum_distance = dist;
            closest_linear_cidx = linear_cidx;
        }
    }
    if (closest_linear_cidx < 0)
    {
        for (long linear_cidx = 0; linear_cidx < n_clusters; linear_cidx++)
        {
            long center_addr = linear_cidx * CENTER_STRIDE;
            if (centers[center_addr] == CENTER_LIMIT)
            {
                continue;
            }
            INTERNAL_FLOAT_DTYPE dist =
                slic_distance(idx, pixel, linear_cidx, center_addr, centers, spacing, inv_ss, max_dist_color);
            if (dist < minimum_distance)
            {
                minimum_distance = dist;
                closest_linear_cidx = linear_cidx;
            }
        }
    }
    labels[linear_idx] = closest_linear_cidx + START_LABEL;
}
#endif

__global__ void update_max_dist_color(const FLOAT_DTYPE* __restrict__ data,
                                      const int* __restrict__ labels,
                                      const FLOAT_DTYPE* __restrict__ centers,
                                      int im_shape_z,
                                      int im_shape_y,
                                      int im_shape_x,
                                      long n_clusters,
                                      FLOAT_DTYPE* __restrict__ max_dist_color
#if USE_MASK
                                      ,
                                      const unsigned char* __restrict__ mask
#endif
)
{
    const unsigned long thread_idx = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned long start_idx = thread_idx * PIXELS_PER_THREAD;
    const unsigned long n_pixels = (unsigned long)im_shape_z * im_shape_y * im_shape_x;

    for (int k = 0; k < PIXELS_PER_THREAD; k++)
    {
        const unsigned long linear_idx = start_idx + k;
        if (linear_idx >= n_pixels)
        {
            break;
        }
#if USE_MASK
        if (!mask[linear_idx])
        {
            continue;
        }
#endif

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
                                   const int* __restrict__ labels,
                                   const FLOAT_DTYPE* __restrict__ centers,
                                   int im_shape_z,
                                   int im_shape_y,
                                   int im_shape_x,
                                   int sp_shape_z,
                                   int sp_shape_y,
                                   int sp_shape_x,
                                   long n_clusters,
                                   SUM_DTYPE* __restrict__ center_sums,
                                   unsigned int* __restrict__ center_counts
#if USE_MASK
                                   ,
                                   const unsigned char* __restrict__ mask
#endif
)
{
    const unsigned long thread_idx = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned long start_idx = thread_idx * PIXELS_PER_THREAD;
    const unsigned long plane_size = (unsigned long)im_shape_y * im_shape_x;
    const unsigned long n_pixels = (unsigned long)im_shape_z * plane_size;

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
#if USE_MASK
        if (!mask[linear_idx])
        {
            continue;
        }
#endif

        const int label = labels[linear_idx];
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

        const int z = linear_idx / plane_size;
        const unsigned long plane_idx = linear_idx - (unsigned long)z * plane_size;
        const int y = plane_idx / im_shape_x;
        const int x = plane_idx - (unsigned long)y * im_shape_x;

#if !USE_MASK
        const long center_addr = linear_cidx * CENTER_STRIDE;
        const int center_z = (int)centers[center_addr + N_PIXEL_FEATURES];
        const int center_y = (int)centers[center_addr + N_PIXEL_FEATURES + 1];
        const int center_x = (int)centers[center_addr + N_PIXEL_FEATURES + 2];
        const float ratio = 2.0f;
        const int from_z = __max(center_z - sp_shape_z * ratio, 0);
        const int from_y = __max(center_y - sp_shape_y * ratio, 0);
        const int from_x = __max(center_x - sp_shape_x * ratio, 0);
        const int to_z = __min(center_z + sp_shape_z * ratio, im_shape_z);
        const int to_y = __min(center_y + sp_shape_y * ratio, im_shape_y);
        const int to_x = __min(center_x + sp_shape_x * ratio, im_shape_x);
        if (z < from_z || z >= to_z || y < from_y || y >= to_y || x < from_x || x >= to_x)
        {
            continue;
        }
#endif

        local_sums[base + N_PIXEL_FEATURES] += z;
        local_sums[base + N_PIXEL_FEATURES + 1] += y;
        local_sums[base + N_PIXEL_FEATURES + 2] += x;
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
                             const int* __restrict__ labels,
                             FLOAT_DTYPE* __restrict__ centers,
                             int im_shape_z,
                             int im_shape_y,
                             int im_shape_x,
                             int sp_shape_z,
                             int sp_shape_y,
                             int sp_shape_x,
                             long n_clusters
#if USE_MASK
                             ,
                             const unsigned char* __restrict__ mask
#endif
)

{
    const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int c_stride = N_PIXEL_FEATURES + 3;
    const long center_addr = linear_cidx * c_stride;

    if (linear_cidx >= n_clusters)
    {
        return;
    }

    int3 cidx;
    cidx.z = (int)centers[center_addr + N_PIXEL_FEATURES];
    cidx.y = (int)centers[center_addr + N_PIXEL_FEATURES + 1];
    cidx.x = (int)centers[center_addr + N_PIXEL_FEATURES + 2];

    float ratio = 2.0f;

    int3 from;
#if USE_MASK
    from.z = 0;
    from.y = 0;
    from.x = 0;
#else
    from.z = __max(cidx.z - sp_shape_z * ratio, 0);
    from.y = __max(cidx.y - sp_shape_y * ratio, 0);
    from.x = __max(cidx.x - sp_shape_x * ratio, 0);
#endif

    int3 to;
#if USE_MASK
    to.z = im_shape_z;
    to.y = im_shape_y;
    to.x = im_shape_x;
#else
    to.z = __min(cidx.z + sp_shape_z * ratio, im_shape_z);
    to.y = __min(cidx.y + sp_shape_y * ratio, im_shape_y);
    to.x = __min(cidx.x + sp_shape_x * ratio, im_shape_x);
#endif

    FLOAT_DTYPE f[c_stride];
    for (int k = 0; k < c_stride; k++)
    {
        f[k] = 0;
    }

    long z_stride = im_shape_x * im_shape_y;
    long y_stride = im_shape_x;

    long count = 0;
    int3 p;
    for (p.z = from.z; p.z < to.z; p.z++)
    {
        long offset_z = p.z * z_stride;
        for (p.y = from.y; p.y < to.y; p.y++)
        {
            long offset_y = offset_z + p.y * y_stride;
            long linear_idx = offset_y + from.x;
            long pixel_addr = linear_idx * N_PIXEL_FEATURES;
            for (p.x = from.x; p.x < to.x; p.x++)
            {
#if USE_MASK
                if (!mask[linear_idx])
                {
                    linear_idx += 1;
                    pixel_addr += N_PIXEL_FEATURES;
                    continue;
                }
#endif
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
                    f[N_PIXEL_FEATURES] += p.z;
                    f[N_PIXEL_FEATURES + 1] += p.y;
                    f[N_PIXEL_FEATURES + 2] += p.x;

                    count += 1;
                }
                linear_idx += 1;
                pixel_addr += N_PIXEL_FEATURES;
            }
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
