/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//
#ifndef CUCIM_UTIL_CUDA_H
#define CUCIM_UTIL_CUDA_H


#if CUCIM_SUPPORT_CUDA
#    include <cuda_runtime.h>
#endif

#define CUDA_TRY(stmt)                                                                                                 \
    {                                                                                                                  \
        cuda_status = stmt;                                                                                            \
        if (cudaSuccess != cuda_status)                                                                                \
        {                                                                                                              \
            fmt::print(stderr, "[Error] CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).\n", #stmt,   \
                       __LINE__, __FILE__, cudaGetErrorString(cuda_status), static_cast<int>(cuda_status));            \
        }                                                                                                              \
    }

#define CUDA_ERROR(stmt)                                                                                               \
    {                                                                                                                  \
        cuda_status = stmt;                                                                                            \
        if (cudaSuccess != cuda_status)                                                                                \
        {                                                                                                              \
            throw std::runtime_error(                                                                                  \
                fmt::format("[Error] CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).\n", #stmt,      \
                            __LINE__, __FILE__, cudaGetErrorString(cuda_status), static_cast<int>(cuda_status)));      \
        }                                                                                                              \
    }

#define NVJPEG_TRY(stmt)                                                                                               \
    {                                                                                                                  \
        nvjpegStatus_t _nvjpeg_status = stmt;                                                                          \
        if (_nvjpeg_status != NVJPEG_STATUS_SUCCESS)                                                                   \
        {                                                                                                              \
            fmt::print("[Error] NVJPEG call {} in line {} of file {} failed with the error code {}.\n", #stmt,         \
                __LINE__, __FILE__, static_cast<int>(_nvjpeg_status));                                                 \
        }                                                                                                              \
    }

#define NVJPEG_ERROR(stmt)                                                                                             \
    {                                                                                                                  \
        nvjpegStatus_t _nvjpeg_status = stmt;                                                                          \
        if (_nvjpeg_status != NVJPEG_STATUS_SUCCESS)                                                                   \
        {                                                                                                              \
            throw std::runtime_error(                                                                                  \
                fmt::format("[Error] NVJPEG call {} in line {} of file {} failed with the error code {}.\n", #stmt,    \
                            __LINE__, __FILE__, static_cast<int>(_nvjpeg_status)));                                    \
        }                                                                                                              \
    }

namespace cucim::util
{

} // namespace cucim::util

#endif // CUCIM_UTIL_CUDA_H
