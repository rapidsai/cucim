/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/codec/base64.h"

#include "cucim/memory/memory_manager.h"

#include <absl/strings/escaping.h>

namespace cucim::codec::base64
{

bool encode(const char* src, int src_count, char** out_dst, int* out_count)
{
    if (src == nullptr)
    {
        return 1;
    }

    absl::string_view sv(src, src_count);
    std::string output;
    absl::Base64Escape(sv, &output);

    int count = output.size();

    if (out_dst == nullptr)
    {
        *out_dst = static_cast<char*>(cucim_malloc(count + 1));
    }
    memcpy(*out_dst, output.c_str(), count);
    *out_dst[count] = '\0';

    if (out_count != nullptr)
    {
        *out_count = count;
    }

    return 0;
}

bool decode(const char* src, int src_count, char** out_dst, int* out_count)
{
    if (src == nullptr)
    {
        return 1;
    }

    absl::string_view sv(src, src_count);
    std::string output;
    if (absl::Base64Unescape(sv, &output))
    {
        int count = output.size();

        if (out_dst == nullptr)
        {
            *out_dst = static_cast<char*>(cucim_malloc(count + 1));
        }
        memcpy(*out_dst, output.c_str(), count);
        *out_dst[count] = '\0';

        if (out_count != nullptr)
        {
            *out_count = count;
        }
    }
    return 0;
}
} // namespace cucim::codec::base64
