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
