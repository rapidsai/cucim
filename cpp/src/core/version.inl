/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_VERSION_INL
#define CUCIM_VERSION_INL

#include "cucim/core/version.h"

namespace cucim
{

constexpr bool operator<(const InterfaceVersion& lhs, const InterfaceVersion& rhs)
{
    if (lhs.major == rhs.major)
    {
        return lhs.minor < rhs.minor;
    }
    return lhs.major < rhs.major;
}

constexpr bool operator==(const InterfaceVersion& lhs, const InterfaceVersion& rhs)
{
    return lhs.major == rhs.major && lhs.minor == rhs.minor;
}

constexpr bool is_version_semantically_compatible(const InterfaceVersion& minimum, const InterfaceVersion& candidate)
{
    if (minimum.major != candidate.major)
    {
        return false;
    }
    else
    {
        // Need to special case when major is equal but zero, then any difference in minor makes them
        // incompatible. See http://semver.org for details.
        if (minimum.major == 0 && minimum.minor != candidate.minor)
        {
            return false;
        }
    }

    if (minimum.minor > candidate.minor)
    {
        return false;
    }
    return true;
}
} // namespace cucim


#endif // CUCIM_VERSION_INL
