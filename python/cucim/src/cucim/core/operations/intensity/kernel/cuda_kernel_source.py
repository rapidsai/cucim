# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cuda_kernel_code = r'''
extern "C" {
__global__ void normalize_data_by_range(float *in, float *out, \
                                        float norm_factor, \
                                        float min_value, \
                                        int total_size)
{
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  if( j < total_size ) {
    out[j] = norm_factor * (in[j] - min_value);
  }
}

__global__ void normalize_data_by_atan(float *in, float *out, \
                                       float norm_factor, \
                                       float min_value, \
                                       int total_size)
{
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  if( j < total_size ) {
    out[j] = norm_factor * atan(in[j] - min_value);
  }
}

__global__ void scaleVolume(float* image, float* output, \
                            float x, float y, float bmin, \
                            float bmax, int W)
{
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < W) {
    output[j] = fmaxf(fminf(image[j] * x - y, bmax), bmin);
  }
}

__global__ void zoom_in_kernel(float *input_tensor, float *output_tensor, \
                 int input_h, int input_w, int output_h, int output_w, \
                 int pitch, int out_h_start, int out_h_end, \
                 int out_w_start, int out_w_end) {
  extern __shared__ float staging_tile[];

  // H -> block Y, row
  // W -> block X, col
  int out_start_h = blockIdx.y * blockDim.y;
  int out_end_h   = (blockIdx.y + 1) * blockDim.y - 1;
  int out_start_w = blockIdx.x * blockDim.x;
  int out_end_w   = (blockIdx.x + 1) * blockDim.x - 1;

  int image_start_offset = blockIdx.z * pitch;

  // ideally should go in unified register
  int smem_load_h_start = floorf((out_start_h * input_h) / (float)output_h);
  int smem_load_h_end = ceilf(((out_end_h+1) * input_h) / (float)output_h);
  int smem_h_load_stretch = smem_load_h_end - smem_load_h_start;

  int smem_load_w_start = floorf((out_start_w * input_w) / (float)output_w);
  int smem_load_w_end = ceilf(((out_end_w+1) * input_w) / (float)output_w);
  int smem_w_load_stretch = smem_load_w_end - smem_load_w_start;

  for (int i = threadIdx.y; i < smem_h_load_stretch; i+=blockDim.y) {
    for (int j = threadIdx.x; j < smem_w_load_stretch; j+=blockDim.x) {

      if (((i+smem_load_h_start) < input_h) &&
          ((j+smem_load_w_start) < input_w)) {
          staging_tile[i * smem_w_load_stretch + j] = \
                      input_tensor[image_start_offset +
                                   (smem_load_h_start + i) * input_w +
                                   smem_load_w_start + j];
      } else {
        staging_tile[i * smem_w_load_stretch + j] = 0.0f;
      }
    }
  }
  __syncthreads();

  int out_pixel_h = blockIdx.y * blockDim.y + threadIdx.y;
  int out_pixel_w = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_pixel_h < output_h && out_pixel_w < output_w
      && out_pixel_h >= out_h_start && out_pixel_h < out_h_end
      && out_pixel_w >= out_w_start && out_pixel_w < out_w_end) {

    // compute pixels oh, ow span
    int start_h = floorf((out_pixel_h * input_h) / (float)output_h);
    int end_h = ceilf(((out_pixel_h+1) * input_h) / (float)output_h);

    int start_w = floorf((out_pixel_w * input_w) / (float)output_w);
    int end_w = ceilf(((out_pixel_w+1) * input_w) / (float)output_w);

    int del_h = end_h - start_h;
    int del_w = end_w - start_w;

    float sum_ = 0.0f;

    for (int i = 0; i < del_h; i++) {
      for (int j = 0; j < del_w; j++) {
        int smem_row = (start_h + i) - smem_load_h_start;
        int smem_col = (start_w + j) - smem_load_w_start;
        sum_ += staging_tile[smem_row * smem_w_load_stretch + smem_col];
      }
    }
    sum_ /= (float)del_h;
    sum_ /= (float)del_w;

    output_tensor[(blockIdx.z * pitch) +
                  ((out_pixel_h - out_h_start) * input_w) +
                  (out_pixel_w - out_w_start)] = sum_;
  }
}

__global__ void zoom_out_kernel(float *input_tensor, float *output_tensor,
                  int input_h, int input_w, int output_h, int output_w,
                  int pitch, int out_h_start, int out_h_end, int out_w_start,
                  int out_w_end) {
  extern __shared__ float staging_tile[];

  // H -> block Y, row
  // W -> block X, col
  int out_start_h = blockIdx.y * blockDim.y;
  int out_end_h   = (blockIdx.y + 1) * blockDim.y - 1;
  int out_start_w = blockIdx.x * blockDim.x;
  int out_end_w   = (blockIdx.x + 1) * blockDim.x - 1;

  int image_start_offset = blockIdx.z * pitch;

  // ideally should go in unified register
  int smem_load_h_start = floorf((out_start_h * input_h) / (float)output_h);
  int smem_load_h_end = ceilf(((out_end_h+1) * input_h) / (float)output_h);
  int smem_h_load_stretch = smem_load_h_end - smem_load_h_start;

  int smem_load_w_start = floorf((out_start_w * input_w) / (float)output_w);
  int smem_load_w_end = ceilf(((out_end_w+1) * input_w) / (float)output_w);
  int smem_w_load_stretch = smem_load_w_end - smem_load_w_start;

  for (int i = threadIdx.y; i < smem_h_load_stretch; i+=blockDim.y) {
    for (int j = threadIdx.x; j < smem_w_load_stretch; j+=blockDim.x) {

      if (((i+smem_load_h_start) < input_h) &&
          ((j+smem_load_w_start) < input_w)) {
          staging_tile[i * smem_w_load_stretch + j] = \
                    input_tensor[image_start_offset +
                                 (smem_load_h_start + i)*input_w +
                                 smem_load_w_start + j];
      } else {
        staging_tile[i * smem_w_load_stretch + j] = 0.0f;
      }
    }
  }
  __syncthreads();

  int out_pixel_h = blockIdx.y * blockDim.y + threadIdx.y;
  int out_pixel_w = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_pixel_h < output_h && out_pixel_w < output_w) {

    // compute pixels oh, ow span
    int start_h = floorf((out_pixel_h * input_h) / (float)output_h);
    int end_h = ceilf(((out_pixel_h+1) * input_h) / (float)output_h);

    int start_w = floorf((out_pixel_w * input_w) / (float)output_w);
    int end_w = ceilf(((out_pixel_w+1) * input_w) / (float)output_w);

    int del_h = end_h - start_h;
    int del_w = end_w - start_w;

    float sum_ = 0.0f;

    for (int i = 0; i < del_h; i++) {
      for (int j = 0; j < del_w; j++) {
        int smem_row = (start_h + i) - smem_load_h_start;
        int smem_col = (start_w + j) - smem_load_w_start;
        sum_ += staging_tile[smem_row * smem_w_load_stretch + smem_col];
      }
    }
    sum_ /= (float)del_h;
    sum_ /= (float)del_w;

    output_tensor[(blockIdx.z * pitch) +
                  ((out_pixel_h + out_h_start) * input_w) +
                  (out_pixel_w + out_w_start)] = sum_;
  }
}

__global__ void zoomout_edge_pad(float *output_tensor, int height, int width,
                                  int pitch, int no_padding_h_start,
                                  int no_padding_w_start,
                                  int no_padding_h_end, int no_padding_w_end) {
  // H -> block Y, row
  // W -> block X, col

  int out_pixel_h = blockIdx.y * blockDim.y + threadIdx.y;
  int out_pixel_w = blockIdx.x * blockDim.x + threadIdx.x;

  // no_padding_h_end, no_padding_w_end --> w_cropped+wstart, same for height
  int out_location = (blockIdx.z * pitch) + (out_pixel_h * width) + out_pixel_w;

  if (out_pixel_h < height && out_pixel_w < width) {
    if (out_pixel_h < no_padding_h_start && out_pixel_w >= no_padding_w_start
          && out_pixel_w < no_padding_w_end) {
      // top pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                    (no_padding_h_start * width) + out_pixel_w];
    } else if (out_pixel_h >= no_padding_h_end
                && out_pixel_w >= no_padding_w_start
                && out_pixel_w < no_padding_w_end) {
      // bottom pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                  ((no_padding_h_end-1) * width) + out_pixel_w];
    } else if (out_pixel_w < no_padding_w_start
                && out_pixel_h >= no_padding_h_start
                && out_pixel_h < no_padding_h_end) {
      // left pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                  (out_pixel_h * width) + no_padding_w_start];
    } else if (out_pixel_w >= no_padding_w_end
                && out_pixel_h >= no_padding_h_start
                && out_pixel_h < no_padding_h_end) {
      // right pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                  (out_pixel_h * width) + (no_padding_w_end-1)];
    } else if (out_pixel_h < no_padding_h_start
                && out_pixel_w < no_padding_w_start) {
      // top-left corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                                (no_padding_h_start * width) +
                                                no_padding_w_start];
    } else if (out_pixel_h < no_padding_h_start
                && out_pixel_w >= no_padding_w_end) {
      // top-right corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                                  (no_padding_h_start * width) +
                                                  (no_padding_w_end-1)];
    } else if (out_pixel_h >= no_padding_h_end
                && out_pixel_w < no_padding_w_start) {
      // bottom-left corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                    ((no_padding_h_end-1) * width) +
                                    no_padding_w_start];
    } else if (out_pixel_h >= no_padding_h_end
                && out_pixel_w >= no_padding_w_end) {
      // bottom-right corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
                                      ((no_padding_h_end-1) * width) +
                                      (no_padding_w_end-1)];
    }
  }
}
}'''
