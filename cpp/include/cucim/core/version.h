/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#ifndef CUCIM_VERSION_H
#define CUCIM_VERSION_H

#include <cstdint>

#ifndef CUCIM_VERSION_MAJOR
#    error "CUCIM_VERSION_MAJOR is not defined"
#endif

#ifndef CUCIM_VERSION_MINOR
#    error "CUCIM_VERSION_MINOR is not defined"
#endif

#ifndef CUCIM_VERSION_PATCH
#    error "CUCIM_VERSION_PATCH is not defined"
#endif

#ifndef CUCIM_VERSION_BUILD
#    error "CUCIM_VERSION_BUILD is not defined"
#endif

namespace cucim
{

struct InterfaceVersion
{
    uint32_t major;
    uint32_t minor;
};

struct Version
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

} // namespace cucim
#endif // CUCIM_VERSION_H
