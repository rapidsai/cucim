/*
 * Apache License, Version 2.0
 * Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
