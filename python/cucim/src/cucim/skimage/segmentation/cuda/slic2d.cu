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

#define DLIMIT 1e308

#define __min(a, b) (((a) < (b)) ? (a) : (b))
#define __max(a, b) (((a) >= (b)) ? (a) : (b))

#ifndef N_PIXEL_FEATURES
#define N_PIXEL_FEATURES 3  // number of features per pixel (e.g. 3 for RGB)
#endif

#ifndef FLOAT_DTYPE
#define FLOAT_DTYPE double
#endif

#ifndef START_LABEL
#define START_LABEL 1  // starting label (must be 0 or 1)
#endif

__forceinline__ __device__ double slic_distance(const int2 idx, const FLOAT_DTYPE* pixel,
                                                const long center_addr, const FLOAT_DTYPE* centers,
                                                const FLOAT_DTYPE* spacing, FLOAT_DTYPE ss)

{
  // Color diff
  double color_diff = 0.;
  for (int w = 0; w < N_PIXEL_FEATURES; w++) {
    FLOAT_DTYPE d = pixel[w] - centers[center_addr + w];
    color_diff += d * d;
  }

  // Position diff

  FLOAT_DTYPE pd_y = (static_cast<FLOAT_DTYPE>(idx.y) - centers[center_addr + N_PIXEL_FEATURES]) * spacing[0];
  FLOAT_DTYPE pd_x = (static_cast<FLOAT_DTYPE>(idx.x) - centers[center_addr + N_PIXEL_FEATURES + 1]) * spacing[1];

  double position_diff = pd_y * pd_y + pd_x * pd_x;
  return color_diff + position_diff / ss;
}

__global__ void expectation(const FLOAT_DTYPE* data, const FLOAT_DTYPE* centers, unsigned int* labels,
                            int im_shape_y, int im_shape_x, int sp_shape_y, int sp_shape_x,
                            int sp_grid_y, int sp_grid_x, FLOAT_DTYPE* spacing, FLOAT_DTYPE* ss)

{
  int2 idx;
  idx.y = threadIdx.x + (blockIdx.x * blockDim.x);
  idx.x = threadIdx.y + (blockIdx.y * blockDim.y);

  if (idx.x >= im_shape_x || idx.y >= im_shape_y) { return; }

  long y_stride = im_shape_x;

  const long linear_idx = idx.y * y_stride + idx.x;
  const long pixel_addr = linear_idx * N_PIXEL_FEATURES;

  FLOAT_DTYPE pixel[N_PIXEL_FEATURES];
  for (int w = 0; w < N_PIXEL_FEATURES; w++) { pixel[w] = data[pixel_addr + w]; }

  int2 cidx;
  long closest_linear_cidx = 0 - START_LABEL;

  // approx center grid position
  cidx.y = max(0, min(idx.y / sp_shape_y, sp_grid_y - 1));
  cidx.x = max(0, min(idx.x / sp_shape_x, sp_grid_x - 1));

  const int c_stride = N_PIXEL_FEATURES + 2;
  double minimum_distance = DLIMIT;
  const int R = 2;
  const int y_start = max(cidx.y - R, 0);
  const int y_end = min(cidx.y + R, sp_grid_y);
  const int x_start = max(cidx.x - R, 0);
  const int x_end = min(cidx.x + R, sp_grid_x);
  for (int j = y_start; j < y_end; j++) {
    long offset_y = j * sp_grid_x;
    for (int i = x_start; i < x_end; i++) {
      long iter_linear_cidx = offset_y + i;
      long iter_center_addr = iter_linear_cidx * c_stride;

      if (centers[iter_center_addr] == DLIMIT) { continue; }

      double dist = slic_distance(idx, pixel, iter_center_addr, centers, spacing, *ss);

      // Wrapup
      if (dist < minimum_distance) {
        minimum_distance = dist;
        closest_linear_cidx = iter_linear_cidx;
      }
    }
  }

  labels[linear_idx] = closest_linear_cidx + START_LABEL;
}

__global__ void maximization(const FLOAT_DTYPE* data, const unsigned int* labels, FLOAT_DTYPE* centers,
                             int im_shape_y, int im_shape_x, int sp_shape_y, int sp_shape_x,
                             long n_clusters) {
  const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);
  const int c_stride = N_PIXEL_FEATURES + 2;
  const long center_addr = linear_cidx * c_stride;

  if (linear_cidx >= n_clusters) { return; }

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
  for (int k = 0; k < c_stride; k++) { f[k] = 0; }

  long y_stride = im_shape_x;

  long count = 0;
  int2 p;
  for (p.y = from.y; p.y < to.y; p.y++) {
    long offset_y = p.y * y_stride;
    long linear_idx = offset_y + from.x;
    long pixel_addr = linear_idx * N_PIXEL_FEATURES;
    for (p.x = from.x; p.x < to.x; p.x++) {
      if (labels[linear_idx] == linear_cidx + START_LABEL) {
        for (int w = 0; w < N_PIXEL_FEATURES; w++) { f[w] += data[pixel_addr + w]; }
        f[N_PIXEL_FEATURES] += p.y;
        f[N_PIXEL_FEATURES + 1] += p.x;
        count += 1;
      }
      linear_idx += 1;
      pixel_addr += N_PIXEL_FEATURES;
    }
  }

  if (count > 0) {
    for (int w = 0; w < c_stride; w++) { centers[center_addr + w] = f[w] / count; }
  } else {
    centers[center_addr] = DLIMIT;
  }
}
