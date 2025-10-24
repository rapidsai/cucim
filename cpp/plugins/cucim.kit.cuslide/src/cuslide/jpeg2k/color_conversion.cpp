/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Code below is derived from the openjpeg's code which is under BSD-2-Clause License.
 * Please see LICENSE-3rdparty.md for the detail.
 * - https://github.com/uclouvain/openjpeg/blob/v2.4.0/src/bin/common/color.c#L403
 */

#include "color_conversion.h"

#include <cucim/profiler/nvtx3.h>

#include "color_table.h"

namespace cuslide::jpeg2k
{

static inline uint8_t clamp(int32_t x)
{
    return (x < 0) ? 0 : ((x > 255) ? 255 : x);
}

void fast_sycc422_to_rgb(opj_image_t* image, uint8_t* dest)
{
    PROF_SCOPED_RANGE(PROF_EVENT(jpeg2k_fast_sycc422_to_rgb));
    const opj_image_comp_t* comps = image->comps;
    const size_t maxw = (size_t)comps[0].w;
    const size_t maxh = (size_t)comps[0].h;
    const int* y = image->comps[0].data;
    const int* cb = image->comps[1].data;
    const int* cr = image->comps[2].data;

    /* if image->x0 is odd, then first column shall use Cb/Cr = 0 */
    size_t offx = image->x0 & 1U;
    size_t loopmaxw = maxw - offx;
    size_t j_max = (loopmaxw & ~(size_t)1U);

    uint8_t c0, c1, c2;
    int16_t R, G, B;
    size_t i, j;

    for (i = 0U; i < maxh; ++i)
    {
        if (offx > 0U)
        {
            c0 = *y;
            c1 = 0;
            c2 = 0;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;
        }

        for (j = 0U; j < j_max; j += 2U)
        {
            c0 = *y;
            c1 = *cb;
            c2 = *cr;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;

            c0 = *y;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;
            ++cb;
            ++cr;
        }
        if (j < loopmaxw)
        {
            c0 = *y;
            c1 = *cb;
            c2 = *cr;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;
            ++cb;
            ++cr;
        }
    }
}

void fast_sycc420_to_rgb(opj_image_t* image, uint8_t* dest)
{
    PROF_SCOPED_RANGE(PROF_EVENT(jpeg2k_fast_sycc420_to_rgb));
    const opj_image_comp_t* comps = image->comps;
    const size_t maxw = (size_t)comps[0].w;
    const size_t maxh = (size_t)comps[0].h;
    const int* y = image->comps[0].data;
    const int* cb = image->comps[1].data;
    const int* cr = image->comps[2].data;

    /* if image->x0 is odd, then first column shall use Cb/Cr = 0 */
    size_t offx = image->x0 & 1U;
    size_t loopmaxw = maxw - offx;
    size_t j_max = (loopmaxw & ~(size_t)1U);
    /* if image->y0 is odd, then first line shall use Cb/Cr = 0 */
    size_t offy = image->y0 & 1U;
    size_t loopmaxh = maxh - offy;
    size_t i_max = (loopmaxh & ~(size_t)1U);

    size_t width_nbytes = maxw * 3;

    uint8_t c0, c1, c2;
    int16_t R, G, B;
    uint8_t* ndest;
    const int* ny;
    size_t i, j;

    if (offy > 0U)
    {
        for (j = 0; j < maxw; ++j)
        {
            c0 = *y;
            c1 = 0;
            c2 = 0;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;
        }
    }

    for (i = 0U; i < i_max; i += 2U)
    {
        ny = y + maxw;
        ndest = dest + width_nbytes;

        if (offx > 0U)
        {
            c0 = *y;
            c1 = 0;
            c2 = 0;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;

            c0 = *ny;
            c1 = *cb;
            c2 = *cr;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *ndest++ = R;
            *ndest++ = G;
            *ndest++ = B;
            ++ny;
        }

        for (j = 0; j < j_max; j += 2U)
        {
            c0 = *y;
            c1 = *cb;
            c2 = *cr;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;

            c0 = *y;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;

            c0 = *ny;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *ndest++ = R;
            *ndest++ = G;
            *ndest++ = B;
            ++ny;

            c0 = *ny;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *ndest++ = R;
            *ndest++ = G;
            *ndest++ = B;
            ++ny;
            ++cb;
            ++cr;
        }
        if (j < loopmaxw)
        {
            c0 = *y;
            c1 = *cb;
            c2 = *cr;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;

            c0 = *ny;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *ndest++ = R;
            *ndest++ = G;
            *ndest++ = B;
            ++ny;
            ++cb;
            ++cr;
        }
        y += maxw;
        dest += width_nbytes;
    }
    if (i < loopmaxh)
    {
        for (j = 0U; j < j_max; j += 2U)
        {
            c0 = *y;
            c1 = *cb;
            c2 = *cr;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;

            c0 = *y;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
            ++y;
            ++cb;
            ++cr;
        }
        if (j < maxw)
        {
            c0 = *y;
            c1 = *cb;
            c2 = *cr;

            R = clamp(c0 + R_Cr[c2]);
            G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
            B = clamp(c0 + B_Cb[c1]);
            *dest++ = R;
            *dest++ = G;
            *dest++ = B;
        }
    }
}

void fast_sycc444_to_rgb(opj_image_t* image, uint8_t* dest)
{
    PROF_SCOPED_RANGE(PROF_EVENT(jpeg2k_fast_sycc444_to_rgb));
    const opj_image_comp_t* comps = image->comps;
    const size_t maxw = (size_t)comps[0].w;
    const size_t maxh = (size_t)comps[0].h;
    const size_t max = maxw * maxh;
    const int* y = image->comps[0].data;
    const int* cb = image->comps[1].data;
    const int* cr = image->comps[2].data;

    uint8_t c0, c1, c2;
    int16_t R, G, B;
    size_t i;
    for (i = 0U; i < max; ++i)
    {
        c0 = *y;
        c1 = *cb;
        c2 = *cr;

        R = clamp(c0 + R_Cr[c2]);
        G = clamp(c0 + ((G_Cb[c1] + G_Cr[c2]) >> 16));
        B = clamp(c0 + B_Cb[c1]);
        *dest++ = R;
        *dest++ = G;
        *dest++ = B;
        ++y;
        ++cb;
        ++cr;
    }
}

void fast_image_to_rgb(opj_image_t* image, uint8_t* dest)
{
    PROF_SCOPED_RANGE(PROF_EVENT(jpeg2k_fast_image_to_rgb));
    opj_image_comp_t* comps = image->comps;
    uint32_t width = comps[0].w;
    uint32_t height = comps[0].h;
    uint32_t items = width * height;

    uint8_t* buf = dest;
    int32_t* comp0 = comps[0].data;
    int32_t* comp1 = comps[1].data;
    int32_t* comp2 = comps[2].data;
    for (uint32_t i = 0; i < items; ++i)
    {
        *(buf++) = comp0[i];
        *(buf++) = comp1[i];
        *(buf++) = comp2[i];
    }
}

} // namespace cuslide::jpeg2k
