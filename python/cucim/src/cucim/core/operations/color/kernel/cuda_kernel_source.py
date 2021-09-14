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
__global__ void brightnessjitter_kernel(unsigned char *input_rgb, unsigned char *output_rgb, int total_pixels, float brightness_factor) {
  // pitch is only WxH - not channels included
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int unvectorized_length = total_pixels;
  int vectorized_length = (unvectorized_length >> 2);

  if (idx < vectorized_length) {
    // 32-bit vectorized loads
    unsigned int input_val = reinterpret_cast<unsigned int*>(input_rgb)[idx];
    unsigned int out_vec = 0x0;

    #pragma unroll(4)
    for (unsigned int ik = 0; ik < 4; ik++) {
      unsigned char val = ((0xff << (ik*8)) & input_val) >> (ik*8);
      float pixel = __uint2float_rn((unsigned int)val);
      pixel *= brightness_factor;
      // clip
      pixel = (pixel <= 0.0f) ? 0.0f : ((pixel >= 255.0f) ? 255.0f : pixel);
      unsigned int tmp = __float2uint_rz(pixel);  // rz helps bitwise match but..
      out_vec = out_vec | (tmp << (ik*8));
    }
    reinterpret_cast<unsigned int*>(output_rgb)[idx] = out_vec;
  } else if (idx == vectorized_length) {
    // one 8-bit element at a time unroll

    int start_idx = idx << 2;
    while (start_idx < unvectorized_length) {
      float pixel = __uint2float_rn((unsigned int)input_rgb[start_idx]);
      pixel *= brightness_factor;
      // clip
      pixel = (pixel <= 0.0f) ? 0.0f : ((pixel >= 255.0f) ? 255.0f : pixel);
      output_rgb[start_idx] = (unsigned char)__float2uint_rz(pixel);
      start_idx++;
    }
  }
}

__global__ void rgb2l_kernel(unsigned char *input_rgb, unsigned int *output_L, int pitch) {
  // 1D grid, access RGB values with pitch'd access
  // pitch is WxH not the image array pitch used for storing surface data
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < pitch) {
    int lookup_idx = idx + (blockIdx.y * (pitch * 3));
    unsigned int ui_r = (unsigned int)input_rgb[lookup_idx];
    unsigned int ui_g = (unsigned int)input_rgb[lookup_idx+pitch];
    unsigned int ui_b = (unsigned int)input_rgb[lookup_idx+pitch*2];
    unsigned int L = ((ui_r * 19595 + ui_g * 38470 + ui_b * 7471) + 0x8000) >> 16;
    int out_idx = (blockIdx.y * pitch) + idx;
    output_L[out_idx] = L;
  }
}

__global__ void blendconstant_kernel(unsigned char *input_rgb, unsigned char *output_rgb, int pitch, float* blend_constant, float blend_factor) {
  // 1D grid, access RGB values with pitch'd access
  // pitch is WxH not the image array pitch used for storing surface data
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int blend_constant_i = int(blend_constant[blockIdx.y] + 0.5);
  float blend_constant_f = (float)blend_constant_i;

  if (idx < pitch) {
    idx += (blockIdx.y * (pitch * 3));
    float r = __uint2float_rn((unsigned int)input_rgb[idx]);
    float g = __uint2float_rn((unsigned int)input_rgb[idx+pitch]);
    float b = __uint2float_rn((unsigned int)input_rgb[idx+pitch*2]);

    // jit_contrast = float(L_round) + contrast * (input_arr.astype(cp.float32) -  float(L_round))

    r = blend_constant_f + blend_factor * (r - blend_constant_f);
    g = blend_constant_f + blend_factor * (g - blend_constant_f);
    b = blend_constant_f + blend_factor * (b - blend_constant_f);

    r = (r <= 0.0f) ? 0.0f : ((r >= 255.0f) ? 255.0f : r);
    g = (g <= 0.0f) ? 0.0f : ((g >= 255.0f) ? 255.0f : g);
    b = (b <= 0.0f) ? 0.0f : ((b >= 255.0f) ? 255.0f : b);

    output_rgb[idx]         = (unsigned char)__float2uint_rz(r);
    output_rgb[idx+pitch]   = (unsigned char)__float2uint_rz(g);
    output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rz(b);
  }
}

__global__ void saturationjitter_kernel(unsigned char *input_rgb, unsigned char *output_rgb, int pitch, float saturation_factor) {
  // 1D grid, access RGB values with pitch'd access
  // pitch is WxH not the image array pitch used for storing surface data
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < pitch) {
    idx += (blockIdx.y * (pitch * 3));
    unsigned int ui_r = (unsigned int)input_rgb[idx];
    unsigned int ui_g = (unsigned int)input_rgb[idx+pitch];
    unsigned int ui_b = (unsigned int)input_rgb[idx+pitch*2];

    // output_L = ((input_arr[0,:,:] * 19595 + input_arr[1,:,:] * 38470 + input_arr[2,:,:] * 7471) + 0x8000) >> 16

    unsigned int L = ((ui_r * 19595 + ui_g * 38470 + ui_b * 7471) + 0x8000) >> 16;
    // jit_saturation = L_saturation + saturation * (input_arr.astype(cp.float32) - L_saturation)

    float sat_L = __uint2float_rn(L);
    float f_r = __uint2float_rn(ui_r);
    float f_g = __uint2float_rn(ui_g);
    float f_b = __uint2float_rn(ui_b);

    float r = sat_L + saturation_factor * (f_r - sat_L);
    float g = sat_L + saturation_factor * (f_g - sat_L);
    float b = sat_L + saturation_factor * (f_b - sat_L);
    r = (r <= 0.0f) ? 0.0f : ((r >= 255.0f) ? 255.0f : r);
    g = (g <= 0.0f) ? 0.0f : ((g >= 255.0f) ? 255.0f : g);
    b = (b <= 0.0f) ? 0.0f : ((b >= 255.0f) ? 255.0f : b);

    output_rgb[idx]         = (unsigned char)__float2uint_rz(r);
    output_rgb[idx+pitch]   = (unsigned char)__float2uint_rz(g);
    output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rz(b);
  }
}

__global__ void huejitter_kernel(unsigned char *input_rgb, unsigned char *output_rgb, int pitch, float hue_factor) {
  // 1D grid, access RGB values with pitch'd access
  // pitch is WxH not the image array pitch used for storing surface data
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // convert to HSV
  // change hue value
  // convert HSV to RGB
  if (idx < pitch) {
    idx += (blockIdx.y * (pitch * 3));
    float r = __uint2float_rn((unsigned int)input_rgb[idx]);
    float g = __uint2float_rn((unsigned int)input_rgb[idx+pitch]);
    float b = __uint2float_rn((unsigned int)input_rgb[idx+pitch*2]);

    float maxc = fmaxf(r, fmaxf(g,b));
    float minc = fminf(r, fminf(g,b));

    float uv = maxc;
    float uh = 0;
    float us = 0;
    if (maxc == minc) {
      uh = 0;
      us = 0;
    } else {
      float cr = maxc - minc;
      float s = cr / maxc;
      float rc = (maxc - r) / cr;
      float gc = (maxc - g) / cr;
      float bc = (maxc - b) / cr;

      float h;
      if (r == maxc) {
        h = bc - gc;
      } else if (g == maxc) {
        h = 2.0f + rc - bc;
      } else {
        h = 4.0f + gc - rc;
      }
      h = (h / 6.0f) + 1.0f;
      h = fmodf(h, 1.0f);
      uh = h * 255.0f;
      uh = (uh <= 0.0f) ? 0.0f : ((uh >= 255.0f) ? 255.0f : uh);
      us = s * 255.0f;
      us = (us <= 0.0f) ? 0.0f : ((us >= 255.0f) ? 255.0f : us);
    }

    // jitter hue
    unsigned char h_char = (unsigned char)__float2uint_rn(uh);
    unsigned char s_char = (unsigned char)__float2uint_rn(us);
    unsigned char v_char = (unsigned char)__float2uint_rn(uv);

    h_char += (hue_factor * 255);

    float h = __uint2float_rn((unsigned int)h_char);
    float s = __uint2float_rn((unsigned int)s_char);
    float v = __uint2float_rn((unsigned int)v_char);

    if (s == 0) {
      // write zero and out
      output_rgb[idx] = 0;
      output_rgb[idx+pitch] = 0;
      output_rgb[idx+pitch*2] = 0;
    } else {
      float i = (h * 6.0f) / 255.0f;
      float f = i - floorf(i);
      float fs = s / 255.0f;
      i = floorf(i);

      float p = roundf(v * (1.0f - fs));
      float q = roundf(v * (1.0f - (fs * f)));
      float t = roundf(v * (1.0f - (fs * (1.0f - f))));

      float up = (p <= 0.0f) ? 0.0f : ((p >= 255.0f) ? 255.0f : p);
      float uq = (q <= 0.0f) ? 0.0f : ((q >= 255.0f) ? 255.0f : q);
      float ut = (t <= 0.0f) ? 0.0f : ((t >= 255.0f) ? 255.0f : t);

      // todo: make atleast 16-bit stores
      switch ((int)i % 6) {
        case 0:
          output_rgb[idx]         = (unsigned char)__float2uint_rn(v);
          output_rgb[idx+pitch]   = (unsigned char)__float2uint_rn(ut);
          output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rn(up);
          break;
        case 1:
          output_rgb[idx]         = (unsigned char)__float2uint_rn(uq);
          output_rgb[idx+pitch]   = (unsigned char)__float2uint_rn(v);
          output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rn(up);
          break;
        case 2:
          output_rgb[idx]         = (unsigned char)__float2uint_rn(up);
          output_rgb[idx+pitch]   = (unsigned char)__float2uint_rn(v);
          output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rn(ut);
          break;
        case 3:
          output_rgb[idx]         = (unsigned char)__float2uint_rn(up);
          output_rgb[idx+pitch]   = (unsigned char)__float2uint_rn(uq);
          output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rn(v);
          break;
        case 4:
          output_rgb[idx]         = (unsigned char)__float2uint_rn(ut);
          output_rgb[idx+pitch]   = (unsigned char)__float2uint_rn(up);
          output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rn(v);
          break;
        case 5:
          output_rgb[idx]         = (unsigned char)__float2uint_rn(v);
          output_rgb[idx+pitch]   = (unsigned char)__float2uint_rn(up);
          output_rgb[idx+pitch*2] = (unsigned char)__float2uint_rn(uq);
          break;
      }
    }
  }
}
}'''
