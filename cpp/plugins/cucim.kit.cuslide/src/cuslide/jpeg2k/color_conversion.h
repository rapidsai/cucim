/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_JPEG2K_COLOR_CONVERSION_H
#define CUSLIDE_JPEG2K_COLOR_CONVERSION_H

#include <cstdint>

#include <openjpeg.h>

namespace cuslide::jpeg2k
{

void fast_sycc420_to_rgb(opj_image_t* image, uint8_t* dest);
void fast_sycc422_to_rgb(opj_image_t* image, uint8_t* dest);
void fast_sycc444_to_rgb(opj_image_t* image, uint8_t* dest);
void fast_image_to_rgb(opj_image_t* image, uint8_t* dest);

} // namespace cuslide::jpeg2k

#endif // CUSLIDE_JPEG2K_COLOR_CONVERSION_H
