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

// Center sums are accumulated as 64-bit fixed-point integers, so that the
// result does not depend on the order of the atomic additions.
#ifndef SUM_DTYPE
#    define SUM_DTYPE long long
#endif

#ifndef FIXED_ROUND
#    define FIXED_ROUND llrintf
#endif

#define CENTER_LIMIT ((FLOAT_DTYPE)1.0e30)
#define DIST_LIMIT ((INTERNAL_FLOAT_DTYPE)1.0e30)

#ifndef EXTRA_CELLS
#    define EXTRA_CELLS 4 // extra grid cells searched for moved centers
#endif

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


#define FULL_MASK 0xffffffffu

// Segmented (by equal consecutive key) inclusive sum over a warp; the last
// lane of each segment ends up with the segment total.
#define SEG_SCAN_ADD(NAME, T)                                                \
__device__ __forceinline__ void NAME(T &v, bool f, int lane)                 \
{                                                                            \
    _Pragma("unroll")                                                        \
    for (int d = 1; d < 32; d <<= 1)                                         \
    {                                                                        \
        T vu = __shfl_up_sync(FULL_MASK, v, d);                              \
        bool fu = __shfl_up_sync(FULL_MASK, (int)f, d) != 0;                 \
        if (lane >= d)                                                       \
        {                                                                    \
            if (!f) v += vu;                                                 \
            f = f || fu;                                                     \
        }                                                                    \
    }                                                                        \
}
SEG_SCAN_ADD(seg_scan_sum, SUM_DTYPE)
SEG_SCAN_ADD(seg_scan_count, unsigned int)

// Appends `value` to `list` (in order of thread index) for every thread of the
// block whose `keep` is true. All threads of the block must call it. `list`
// needs room for blockDim.x * blockDim.y * blockDim.z entries beyond
// *list_size and `warp_counts` for 32 warps.
__device__ __forceinline__ void block_ordered_append(const bool keep,
                                                     const int value,
                                                     int* __restrict__ list,
                                                     int* __restrict__ warp_counts,
                                                     int* __restrict__ list_size)
{
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int n_warps = (blockDim.x * blockDim.y * blockDim.z + 31) >> 5;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const unsigned int ballot = __ballot_sync(FULL_MASK, keep);
    if (lane == 0)
    {
        warp_counts[warp] = __popc(ballot);
    }
    __syncthreads();
    int offset = *list_size;
    int total = 0;
    for (int w = 0; w < n_warps; w++)
    {
        if (w < warp)
        {
            offset += warp_counts[w];
        }
        total += warp_counts[w];
    }
    if (keep)
    {
        list[offset + __popc(ballot & ((1u << lane) - 1))] = value;
    }
    __syncthreads();
    if (tid == 0)
    {
        *list_size += total;
    }
    __syncthreads();
}

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
                            int win_shape_z,
                            int win_shape_y,
                            int win_shape_x,
                            const FLOAT_DTYPE* __restrict__ spacing,
                            const FLOAT_DTYPE* __restrict__ ss,
                            const FLOAT_DTYPE* __restrict__ max_dist_color)

{
    int3 idx;
    idx.z = threadIdx.x + (blockIdx.x * blockDim.x);
    idx.y = threadIdx.y + (blockIdx.y * blockDim.y);
    idx.x = threadIdx.z + (blockIdx.z * blockDim.z);
    const bool inside = idx.x < im_shape_x && idx.y < im_shape_y && idx.z < im_shape_z;

    const int c_stride = N_PIXEL_FEATURES + 3;
    const int n_threads = blockDim.x * blockDim.y * blockDim.z;
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    // Bounding box of the block's pixels, and the range of grid cells that
    // contains every center that any of those pixels searches.
    const int box_lo_z = blockIdx.x * blockDim.x;
    const int box_hi_z = min(box_lo_z + (int)blockDim.x, im_shape_z) - 1;
    const int box_lo_y = blockIdx.y * blockDim.y;
    const int box_hi_y = min(box_lo_y + (int)blockDim.y, im_shape_y) - 1;
    const int box_lo_x = blockIdx.z * blockDim.z;
    const int box_hi_x = min(box_lo_x + (int)blockDim.z, im_shape_x) - 1;
    const int R = 2 + EXTRA_CELLS;
    const int cell_lo_z = max(0, min(box_lo_z / sp_shape_z, sp_grid_z - 1));
    const int cell_hi_z = max(0, min(box_hi_z / sp_shape_z, sp_grid_z - 1));
    const int cell_lo_y = max(0, min(box_lo_y / sp_shape_y, sp_grid_y - 1));
    const int cell_hi_y = max(0, min(box_hi_y / sp_shape_y, sp_grid_y - 1));
    const int cell_lo_x = max(0, min(box_lo_x / sp_shape_x, sp_grid_x - 1));
    const int cell_hi_x = max(0, min(box_hi_x / sp_shape_x, sp_grid_x - 1));
    const int z_start = max(cell_lo_z - R, 0);
    const int z_end = min(cell_hi_z + R, sp_grid_z);
    const int y_start = max(cell_lo_y - R, 0);
    const int y_end = min(cell_hi_y + R, sp_grid_y);
    const int x_start = max(cell_lo_x - R, 0);
    const int x_end = min(cell_hi_x + R, sp_grid_x);
    const int n_x = max(x_end - x_start, 0);
    const int n_y = max(y_end - y_start, 0);
    const int n_cells = max(z_end - z_start, 0) * n_y * n_x;

    long linear_idx = 0;
    FLOAT_DTYPE pixel[N_PIXEL_FEATURES];
    INTERNAL_FLOAT_DTYPE inv_ss = 0;
    // a pixel that is in no window keeps its previous label, as in scikit-image
    long closest_linear_cidx = 0;
    INTERNAL_FLOAT_DTYPE minimum_distance = DIST_LIMIT;
    if (inside)
    {
        linear_idx = ((long)idx.z * im_shape_y + idx.y) * im_shape_x + idx.x;
        const long pixel_addr = linear_idx * N_PIXEL_FEATURES;
#if N_PIXEL_FEATURES <= 8
#    pragma unroll
#else
#    pragma unroll 8
#endif
        for (int w = 0; w < N_PIXEL_FEATURES; w++)
        {
            pixel[w] = data[pixel_addr + w];
        }
        inv_ss = static_cast<INTERNAL_FLOAT_DTYPE>(1) / static_cast<INTERNAL_FLOAT_DTYPE>(*ss);
        closest_linear_cidx = (long)labels[linear_idx] - START_LABEL;
    }

    // Centers whose window overlaps the block's bounding box are selected
    // cooperatively, n_threads cells at a time and in grid order (so ties are
    // broken as if every pixel looped over all cells). Each pixel then applies
    // its own window test to the selected centers.
    __shared__ int selected[1024];
    __shared__ int warp_counts[32];
    __shared__ int n_selected;
    for (int first = 0; first < n_cells; first += n_threads)
    {
        const int cell = first + tid;
        bool keep = false;
        int linear_cidx = 0;
        if (cell < n_cells)
        {
            linear_cidx = ((z_start + cell / (n_y * n_x)) * sp_grid_y + y_start + (cell / n_x) % n_y) * sp_grid_x +
                          x_start + cell % n_x;
            const long center_addr = (long)linear_cidx * c_stride;
            if (centers[center_addr] != CENTER_LIMIT)
            {
                const int center_z = (int)centers[center_addr + N_PIXEL_FEATURES];
                const int center_y = (int)centers[center_addr + N_PIXEL_FEATURES + 1];
                const int center_x = (int)centers[center_addr + N_PIXEL_FEATURES + 2];
                keep = box_hi_z >= center_z - 2 * win_shape_z && box_lo_z <= center_z + 2 * win_shape_z &&
                       box_hi_y >= center_y - 2 * win_shape_y && box_lo_y <= center_y + 2 * win_shape_y &&
                       box_hi_x >= center_x - 2 * win_shape_x && box_lo_x <= center_x + 2 * win_shape_x;
            }
        }
        if (tid == 0)
        {
            n_selected = 0;
        }
        __syncthreads();
        block_ordered_append(keep, linear_cidx, selected, warp_counts, &n_selected);
        if (inside)
        {
            for (int s = 0; s < n_selected; s++)
            {
                const long iter_linear_cidx = selected[s];
                const long iter_center_addr = iter_linear_cidx * c_stride;

                // The pixel must be inside the window (+- 2 steps) around the
                // current center, as in scikit-image.
                const int center_z = (int)centers[iter_center_addr + N_PIXEL_FEATURES];
                const int center_y = (int)centers[iter_center_addr + N_PIXEL_FEATURES + 1];
                const int center_x = (int)centers[iter_center_addr + N_PIXEL_FEATURES + 2];
                if (idx.z < center_z - 2 * win_shape_z || idx.z > center_z + 2 * win_shape_z || idx.y < center_y - 2 * win_shape_y || idx.y > center_y + 2 * win_shape_y ||
                    idx.x < center_x - 2 * win_shape_x || idx.x > center_x + 2 * win_shape_x)
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
        __syncthreads();
    }

    if (inside)
    {
        labels[linear_idx] = closest_linear_cidx + START_LABEL;
    }
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
    const bool inside = idx.x < im_shape_x && idx.y < im_shape_y && idx.z < im_shape_z;
    const int n_threads = blockDim.x * blockDim.y * blockDim.z;
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    // bounding box of the block's voxels
    const int box_lo[3] = {(int)(blockIdx.x * blockDim.x),
                           (int)(blockIdx.y * blockDim.y),
                           (int)(blockIdx.z * blockDim.z)};
    const int box_hi[3] = {min(box_lo[0] + (int)blockDim.x, im_shape_z) - 1,
                           min(box_lo[1] + (int)blockDim.y, im_shape_y) - 1,
                           min(box_lo[2] + (int)blockDim.z, im_shape_x) - 1};
    const int win[3] = {sp_shape_z, sp_shape_y, sp_shape_x};
    const int pos[3] = {idx.z, idx.y, idx.x};

    long linear_idx = 0;
    bool active = false;  // inside the image and the mask
    FLOAT_DTYPE pixel[N_PIXEL_FEATURES];
    INTERNAL_FLOAT_DTYPE inv_ss = 0;
    if (inside)
    {
        linear_idx = ((long)idx.z * im_shape_y + idx.y) * im_shape_x + idx.x;
        active = mask[linear_idx] != 0;
        if (active)
        {
            const long pixel_addr = linear_idx * N_PIXEL_FEATURES;
#    if N_PIXEL_FEATURES <= 8
#        pragma unroll
#    else
#        pragma unroll 8
#    endif
            for (int w = 0; w < N_PIXEL_FEATURES; w++)
            {
                pixel[w] = data[pixel_addr + w];
            }
            inv_ss = static_cast<INTERNAL_FLOAT_DTYPE>(1) / static_cast<INTERNAL_FLOAT_DTYPE>(*ss);
        }
        else
        {
            labels[linear_idx] = START_LABEL - 1;
        }
    }

    INTERNAL_FLOAT_DTYPE minimum_distance = DIST_LIMIT;
    long closest_linear_cidx = -1;

    // The centers whose window overlaps the block's bounding box are selected
    // cooperatively, n_threads centers at a time and in index order (so ties
    // are broken as if every pixel looped over all centers). Each pixel then
    // applies its own window test to the selected centers.
    __shared__ int selected[1024];
    __shared__ int warp_counts[32];
    __shared__ int n_selected;
    for (long first = 0; first < n_clusters; first += n_threads)
    {
        const long linear_cidx = first + tid;
        bool keep = false;
        if (linear_cidx < n_clusters)
        {
            const long center_addr = linear_cidx * CENTER_STRIDE;
            keep = centers[center_addr] != CENTER_LIMIT;
            for (int d = 0; d < 3; d++)
            {
                const int c = (int)centers[center_addr + N_PIXEL_FEATURES + d];
                keep = keep && box_hi[d] >= c - 2 * win[d] && box_lo[d] <= c + 2 * win[d];
            }
        }
        if (tid == 0)
        {
            n_selected = 0;
        }
        __syncthreads();
        block_ordered_append(keep, (int)linear_cidx, selected, warp_counts, &n_selected);
        if (active)
        {
            for (int s = 0; s < n_selected; s++)
            {
                const long selected_cidx = selected[s];
                const long center_addr = selected_cidx * CENTER_STRIDE;
                bool in_window = true;
                for (int d = 0; d < 3; d++)
                {
                    const int c = (int)centers[center_addr + N_PIXEL_FEATURES + d];
                    in_window = in_window && pos[d] >= c - 2 * win[d] && pos[d] <= c + 2 * win[d];
                }
                if (!in_window)
                {
                    continue;
                }
                INTERNAL_FLOAT_DTYPE dist =
                    slic_distance(idx, pixel, selected_cidx, center_addr, centers, spacing, inv_ss, max_dist_color);
                if (dist < minimum_distance)
                {
                    minimum_distance = dist;
                    closest_linear_cidx = selected_cidx;
                }
            }
        }
        __syncthreads();
    }

    if (!active)
    {
        return;
    }
    // a pixel that is in no window is assigned to the closest center overall
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
    const unsigned long thread_idx = threadIdx.x + ((unsigned long)blockIdx.x * blockDim.x);
    const unsigned long start_idx = thread_idx * PIXELS_PER_THREAD;
    const unsigned long n_pixels = (unsigned long)im_shape_z * im_shape_y * im_shape_x;
    const int lane = threadIdx.x & 31;

    // Same scheme as accumulate_centers: the running maximum of a run of equal
    // labels is flushed with atomicMax when the label changes, and the last run
    // is first combined across the warp.
    int run_label = -1;
    FLOAT_DTYPE run_max = 0;
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
        const long center_addr = linear_cidx * CENTER_STRIDE;
        if (centers[center_addr] == CENTER_LIMIT)
        {
            continue;
        }
        const unsigned long pixel_addr = linear_idx * N_PIXEL_FEATURES;
        FLOAT_DTYPE dist_color = 0;
#pragma unroll
        for (int w = 0; w < N_PIXEL_FEATURES; w++)
        {
            FLOAT_DTYPE d = data[pixel_addr + w] - centers[center_addr + w];
            dist_color += d * d;
        }
        if (label != run_label)
        {
            if (run_label >= 0)
            {
                atomicMax(&max_dist_color[run_label - START_LABEL], run_max);
            }
            run_label = label;
            run_max = dist_color;
        }
        else
        {
            run_max = __max(run_max, dist_color);
        }
    }
    const int prev_key = __shfl_up_sync(FULL_MASK, run_label, 1);
    bool f = (lane == 0) || (prev_key != run_label);
#pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        FLOAT_DTYPE vu = __shfl_up_sync(FULL_MASK, run_max, d);
        bool fu = __shfl_up_sync(FULL_MASK, (int)f, d) != 0;
        if (lane >= d)
        {
            if (!f) run_max = __max(run_max, vu);
            f = f || fu;
        }
    }
    const int next_key = __shfl_down_sync(FULL_MASK, run_label, 1);
    if (((lane == 31) || (next_key != run_label)) && run_label >= 0)
    {
        atomicMax(&max_dist_color[run_label - START_LABEL], run_max);
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
                                   unsigned int* __restrict__ center_counts,
                                   const FLOAT_DTYPE* __restrict__ fixed_scale
#if USE_MASK
                                   ,
                                   const unsigned char* __restrict__ mask
#endif
)
{
    const FLOAT_DTYPE scale = *fixed_scale;
    const unsigned long thread_idx = threadIdx.x + ((unsigned long)blockIdx.x * blockDim.x);
    const unsigned long start_idx = thread_idx * PIXELS_PER_THREAD;
    const unsigned long plane_size = (unsigned long)im_shape_y * im_shape_x;
    const unsigned long n_pixels = (unsigned long)im_shape_z * plane_size;
    const int lane = threadIdx.x & 31;

    // The current run of consecutive accepted pixels with one label. A run
    // that ends inside the thread is flushed with atomics (rare); the last
    // run is combined with equal-label neighbors across the warp first.
    int run_label = -1;
    unsigned int run_count = 0;
    SUM_DTYPE run_sums[CENTER_STRIDE];
    for (int w = 0; w < CENTER_STRIDE; w++) run_sums[w] = 0;

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
        const int z = linear_idx / plane_size;
        const unsigned long plane_idx = linear_idx - (unsigned long)z * plane_size;
        const int y = plane_idx / im_shape_x;
        const int x = plane_idx - (unsigned long)y * im_shape_x;
        const int pos[3] = {z, y, x};
#if !USE_MASK
        const long center_addr = linear_cidx * CENTER_STRIDE;
        const int center_z = (int)centers[center_addr + N_PIXEL_FEATURES];
        const int center_y = (int)centers[center_addr + N_PIXEL_FEATURES + 1];
        const int center_x = (int)centers[center_addr + N_PIXEL_FEATURES + 2];
        const float ratio = 2.0f;
        const int from_z = __max(center_z - sp_shape_z * ratio, 0);
        const int from_y = __max(center_y - sp_shape_y * ratio, 0);
        const int from_x = __max(center_x - sp_shape_x * ratio, 0);
        const int to_z = __min(center_z + 2 * sp_shape_z + 1, im_shape_z);
        const int to_y = __min(center_y + 2 * sp_shape_y + 1, im_shape_y);
        const int to_x = __min(center_x + 2 * sp_shape_x + 1, im_shape_x);
        if (z < from_z || z >= to_z || y < from_y || y >= to_y || x < from_x || x >= to_x)
        {
            continue;
        }
#endif
        if (label != run_label)
        {
            if (run_label >= 0)
            {
                const long a = (long)(run_label - START_LABEL) * CENTER_STRIDE;
                for (int w = 0; w < CENTER_STRIDE; w++)
                {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&center_sums[a + w]), static_cast<unsigned long long>(run_sums[w]));
                    run_sums[w] = 0;
                }
                atomicAdd(&center_counts[run_label - START_LABEL], run_count);
            }
            run_label = label;
            run_count = 0;
        }
        const unsigned long pixel_addr = linear_idx * N_PIXEL_FEATURES;
#if N_PIXEL_FEATURES <= 8
#    pragma unroll
#else
#    pragma unroll 8
#endif
        for (int w = 0; w < N_PIXEL_FEATURES; w++)
        {
            run_sums[w] += FIXED_ROUND(data[pixel_addr + w] * scale);
        }
        for (int a = 0; a < 3; a++)
        {
            run_sums[N_PIXEL_FEATURES + a] += pos[a];
        }
        run_count += 1;
    }

    // warp-level segmented reduction of the last run
    int prev_key = __shfl_up_sync(FULL_MASK, run_label, 1);
    bool head = (lane == 0) || (prev_key != run_label);
    for (int w = 0; w < CENTER_STRIDE; w++)
    {
        seg_scan_sum(run_sums[w], head, lane);
    }
    seg_scan_count(run_count, head, lane);
    int next_key = __shfl_down_sync(FULL_MASK, run_label, 1);
    const bool tail = (lane == 31) || (next_key != run_label);
    if (tail && run_label >= 0)
    {
        const long a = (long)(run_label - START_LABEL) * CENTER_STRIDE;
        for (int w = 0; w < CENTER_STRIDE; w++)
        {
            atomicAdd(reinterpret_cast<unsigned long long*>(&center_sums[a + w]), static_cast<unsigned long long>(run_sums[w]));
        }
        atomicAdd(&center_counts[run_label - START_LABEL], run_count);
    }
}

__global__ void normalize_centers(FLOAT_DTYPE* __restrict__ centers,
                                  const SUM_DTYPE* __restrict__ center_sums,
                                  const unsigned int* __restrict__ center_counts,
                                  const FLOAT_DTYPE* __restrict__ fixed_scale,
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
        // the scale is a power of two, so its inverse is exact
        const double inv_count = 1.0 / static_cast<double>(count);
        const double inv_scale = 1.0 / static_cast<double>(*fixed_scale);
        for (int w = 0; w < CENTER_STRIDE; w++)
        {
            // features are scaled fixed-point values, coordinates are exact integer sums
            const double unit = (w < N_PIXEL_FEATURES) ? inv_scale : 1.0;
            centers[center_addr + w] =
                static_cast<FLOAT_DTYPE>(static_cast<double>(center_sums[center_addr + w]) * unit * inv_count);
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
    to.z = __min(cidx.z + 2 * sp_shape_z + 1, im_shape_z);
    to.y = __min(cidx.y + 2 * sp_shape_y + 1, im_shape_y);
    to.x = __min(cidx.x + 2 * sp_shape_x + 1, im_shape_x);
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
