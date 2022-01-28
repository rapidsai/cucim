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

#ifndef CUCIM_FILE_HANDLE_H
#define CUCIM_FILE_HANDLE_H


#include "../macros/defines.h"
#include <cstdio>
#include <cstdint>
#include <memory>

#include <fmt/format.h>

#include "cucim/memory/memory_manager.h"

typedef void* CUfileHandle_t;
typedef void* CuCIMFileHandle_share;
typedef void* CuCIMFileHandle_ptr;
typedef bool (*CuCIMFileHandleDeleter)(CuCIMFileHandle_ptr);

enum class FileHandleType: uint16_t
{
    kUnknown = 0,
    kPosix = 1,
    kPosixODirect = 1 << 1,
    kMemoryMapped = 1 << 2,
    kGPUDirect = 1 << 3,
};


#if CUCIM_PLATFORM_LINUX

struct EXPORT_VISIBLE CuCIMFileHandle : public std::enable_shared_from_this<CuCIMFileHandle>
{
    CuCIMFileHandle();
    CuCIMFileHandle(int fd, CUfileHandle_t cufile, FileHandleType type, char* path, void* client_data);
    CuCIMFileHandle(int fd,
                    CUfileHandle_t cufile,
                    FileHandleType type,
                    char* path,
                    void* client_data,
                    uint64_t dev,
                    uint64_t ino,
                    int64_t mtime);

    ~CuCIMFileHandle()
    {
        if (path && path[0] != '\0')
        {
            cucim_free(path);
            path = nullptr;
        }

        if (deleter)
        {
            deleter(this);
            deleter = nullptr;
        }
    }

    CuCIMFileHandleDeleter set_deleter(CuCIMFileHandleDeleter deleter)
    {
        return this->deleter = deleter;
    }

    int fd = -1;
    CUfileHandle_t cufile = nullptr;
    FileHandleType type = FileHandleType::kUnknown; /// 1: POSIX, 2: POSIX+ODIRECT, 4: MemoryMapped, 8: GPUDirect
    char* path = nullptr;
    void* client_data = nullptr;
    uint64_t hash_value = 0;
    uint64_t dev = 0;
    uint64_t ino = 0;
    int64_t mtime = 0;
    CuCIMFileHandleDeleter deleter = nullptr;
};
#else
#    error "This platform is not supported!"
#endif

#endif // CUCIM_FILE_HANDLE_H
