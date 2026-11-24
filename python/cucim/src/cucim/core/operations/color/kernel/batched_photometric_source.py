# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The arithmetic reproduces Pillow's uint8 operations, including the rounding
# between successive operations. Compile with --fmad=false: contracting the
# multiply-adds into FMAs changes which way values round.
batched_photometric_code = r"""
// Per-image parameters, int32[NUM_PARAMS]:
//   0-3  jitter order: 0 brightness, 1 contrast, 2 saturation, 3 hue;
//        -1 ends the sequence
//   4    horizontal flip
//   5    grayscale
//   6    solarize
//   7    index of the image among the blurred images, or -1 for no blur
//   8    hue shift in [0, 255]
// Per-image factors, float32[4]: brightness, contrast, saturation, hue.
#define NUM_PARAMS 9

__device__ unsigned char byte_clip(float x) {
    return (unsigned char)fminf(255.f, fmaxf(0.f, x));
}

__device__ unsigned char luminance(unsigned char r, unsigned char g,
                                   unsigned char b) {
    return (19595 * (int)r + 38470 * (int)g + 7471 * (int)b + 32768) >> 16;
}

__device__ void blend(unsigned char &r, unsigned char &g, unsigned char &b,
                      float base, float factor) {
    r = byte_clip(base + factor * ((float)r - base));
    g = byte_clip(base + factor * ((float)g - base));
    b = byte_clip(base + factor * ((float)b - base));
}

// Pillow's RGB -> HSV, uint8 hue shift, HSV -> RGB round trip.
__device__ void hue_shift(unsigned char &r, unsigned char &g, unsigned char &b,
                          int shift) {
    float hi = fmaxf(r, fmaxf(g, b)), lo = fminf(r, fminf(g, b));
    float delta = hi - lo, divisor = fmaxf(delta, 1.f);
    float rd = (hi - r) / divisor, gd = (hi - g) / divisor,
          bd = (hi - b) / divisor;
    float h = r == hi ? bd - gd
            : (g == hi ? (float)(2.0 + (double)rd - (double)bd)
                       : (float)(4.0 + (double)gd - (double)rd));
    double h0 = (double)h / 6.0 + 1.0;
    h = (float)(h0 - floor(h0));
    int hb = delta == 0 ? 0 : (int)floor((double)h * 255.0);
    int sb = (int)floor((double)(delta / fmaxf(hi, 1.f)) * 255.0);
    hb = (hb + shift) % 256;
    double position = (double)hb * 6.0 / 255.0;
    int sector = (int)floor(position);
    float fraction = (float)(position - sector);
    float saturation = (float)((double)sb / 255.0);
    float p = (float)floor((double)hi * (1.0 - (double)saturation) + 0.5);
    float q = (float)floor(
        (double)hi * (1.0 - (double)(saturation * fraction)) + 0.5);
    float t = (float)floor(
        (double)hi * (1.0 - (double)saturation * (1.0 - (double)fraction))
        + 0.5);
    switch (sector % 6) {
        case 0: r = hi; g = t; b = p; break;
        case 1: r = q; g = hi; b = p; break;
        case 2: r = p; g = hi; b = t; break;
        case 3: r = p; g = q; b = hi; break;
        case 4: r = t; g = p; b = hi; break;
        default: r = hi; g = p; b = q;
    }
}

// Every jitter operation except contrast, which needs the image mean.
__device__ void pointwise(unsigned char &r, unsigned char &g, unsigned char &b,
                          int op, const float *factors, int shift) {
    if (op == 0) blend(r, g, b, 0.f, factors[0]);
    else if (op == 2) blend(r, g, b, luminance(r, g, b), factors[2]);
    else if (op == 3) hue_shift(r, g, b, shift);
}

__device__ float normalized(int value, int channel, bool solarize,
                            int threshold, const float *mean_std) {
    if (solarize && value >= threshold) value = 255 - value;
    return ((float)value / 255.f - mean_std[channel]) / mean_std[channel + 3];
}

__device__ int reflect(int index, int length) {
    return index < 0 ? -index
                     : (index >= length ? 2 * length - 2 - index : index);
}

// One thread per pixel. Reads image n through its own pointer, flips, and
// applies the jitter operations that precede contrast. Writes interleaved RGB
// and, when contrast is used, the luminance for the per-image mean.
extern "C" __global__ void photometric_prefix(
        const unsigned long long *images, long long pixel_stride,
        long long channel_stride, const int *params, const float *factors,
        unsigned char *out, unsigned char *gray, int height, int width,
        long long total) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long area = (long long)height * width;
    int n = (int)(idx / area);
    long long pos = idx - n * area;
    int y = (int)(pos / width), x = (int)(pos - (long long)y * width);
    const int *p = params + NUM_PARAMS * n;
    const unsigned char *in = (const unsigned char *)images[n];
    long long src =
        ((long long)y * width + (p[4] ? width - 1 - x : x)) * pixel_stride;
    unsigned char r = in[src], g = in[src + channel_stride],
                  b = in[src + 2 * channel_stride];
    for (int j = 0; j < 4 && p[j] >= 0 && p[j] != 1; j++)
        pointwise(r, g, b, p[j], factors + 4 * n, p[8]);
    out[idx * 3] = r;
    out[idx * 3 + 1] = g;
    out[idx * 3 + 2] = b;
    if (gray) gray[idx] = luminance(r, g, b);
}

// One thread per pixel. Applies contrast, the jitter operations after it and
// grayscale. Images to blur go to blur_in; others are finished here, either
// as uint8 in the caller's layout (final == NULL) or normalized NCHW float.
extern "C" __global__ void photometric_suffix(
        const unsigned char *in, const float *means, const int *params,
        const float *factors, unsigned char *out, long long pixel_stride,
        long long channel_stride, unsigned char *blur_in, float *final,
        const float *mean_std, int threshold, long long area,
        long long total) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int n = (int)(idx / area);
    long long pos = idx - n * area;
    const int *p = params + NUM_PARAMS * n;
    unsigned char r = in[idx * 3], g = in[idx * 3 + 1], b = in[idx * 3 + 2];
    bool after_contrast = false;
    for (int j = 0; j < 4 && p[j] >= 0; j++) {
        if (p[j] == 1) {
            blend(r, g, b, means[n], factors[4 * n + 1]);
            after_contrast = true;
        } else if (after_contrast) {
            pointwise(r, g, b, p[j], factors + 4 * n, p[8]);
        }
    }
    if (p[5]) r = g = b = luminance(r, g, b);
    if (p[7] >= 0) {
        long long target = ((long long)p[7] * area + pos) * 3;
        blur_in[target] = r;
        blur_in[target + 1] = g;
        blur_in[target + 2] = b;
    } else if (final) {
        float *target = final + n * 3 * area + pos;
        target[0] = normalized(r, 0, p[6], threshold, mean_std);
        target[area] = normalized(g, 1, p[6], threshold, mean_std);
        target[2 * area] = normalized(b, 2, p[6], threshold, mean_std);
    } else {
        long long target = n * 3 * area + pos * pixel_stride;
        out[target] = r;
        out[target + channel_stride] = g;
        out[target + 2 * channel_stride] = b;
    }
}

// One thread per channel value of each blurred image: horizontal pass of the
// separable Gaussian, with reflected borders.
extern "C" __global__ void photometric_blur_rows(
        const unsigned char *in, float *out, const float *weights, int radius,
        int height, int width, long long total) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long pixel = idx / 3, area = (long long)height * width;
    int c = (int)(idx - pixel * 3), s = (int)(pixel / area);
    long long pos = pixel - s * area;
    int y = (int)(pos / width), x = (int)(pos - (long long)y * width);
    const float *w = weights + (long long)s * (2 * radius + 1);
    const unsigned char *row = in + (s * area + (long long)y * width) * 3 + c;
    float sum = 0.f;
    for (int k = 0; k <= 2 * radius; k++)
        sum += (float)row[reflect(x + k - radius, width) * 3] * w[k];
    out[idx] = sum;
}

// Vertical pass, then rounding, solarization, normalization and NCHW output.
extern "C" __global__ void photometric_blur_columns(
        const float *in, float *final, const int *blur_images,
        const int *params, const float *weights, const float *mean_std,
        int threshold, int radius, int height, int width, long long total) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long pixel = idx / 3, area = (long long)height * width;
    int c = (int)(idx - pixel * 3), s = (int)(pixel / area);
    long long pos = pixel - s * area;
    int y = (int)(pos / width), x = (int)(pos - (long long)y * width);
    const float *w = weights + (long long)s * (2 * radius + 1);
    const float *column = in + s * area * 3 + (long long)x * 3 + c;
    float sum = 0.f;
    for (int k = 0; k <= 2 * radius; k++)
        sum += column[(long long)reflect(y + k - radius, height) * width * 3]
               * w[k];
    int value = max(0, min(255, __float2int_rn(sum)));
    int n = blur_images[s];
    final[((long long)n * 3 + c) * area + pos] =
        normalized(value, c, params[NUM_PARAMS * n + 6], threshold, mean_std);
}
"""
