/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#ifndef PYCUCIM_CACHE_PYDOC_H
#define PYCUCIM_CACHE_PYDOC_H

#include "../macros.h"

namespace cucim::cache::doc
{

// py::int_ py_preferred_memory_capacity(const py::object& img,
//                                       const std::optional<const std::vector<uint32_t>>& image_size,
//                                       const std::optional<const std::vector<uint32_t>>& tile_size,
//                                       const std::optional<const std::vector<uint32_t>>& patch_size,
//                                       uint32_t bytes_per_pixel);
PYDOC(preferred_memory_capacity, R"doc(
Returns a good cache memory capacity value in MiB for the given conditions.

Please see how the value is calculated: https://godbolt.org/z/8vxnPfKM5

Args:
    img: A `CuImage` object that can provide `image_size`, `tile_size`, `bytes_per_pixel` information.
        If this argument is provided, only `patch_size` from the arguments is used for the calculation.
    image_size: A list of values that presents the image size (width, height).
    tile_size: A list of values that presents the tile size (width, height). The default value is (256, 256).
    patch_size: A list of values that presents the patch size (width, height). The default value is (256, 256).
    bytes_per_pixel: The number of bytes that each pixel in the 2D image takes place. The default value is 3.

Returns:
    The suggested memory capacity in MiB.
)doc")

} // namespace cucim::cache::doc
#endif // PYCUCIM_CACHE_PYDOC_H
