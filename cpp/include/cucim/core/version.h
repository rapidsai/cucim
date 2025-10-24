/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
