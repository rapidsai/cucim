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

typedef void* CUfileHandle_t;

enum class FileHandleType: uint16_t
{
    kUnknown = 0,
    kPosix = 1,
    kPosixODirect = 1 << 1,
    kMemoryMapped = 1 << 2,
    kGPUDirect = 1 << 3,
};


#if CUCIM_PLATFORM_LINUX
struct CuCIMFileHandle
{
#    ifdef __cplusplus
    EXPORT_VISIBLE CuCIMFileHandle();
    EXPORT_VISIBLE CuCIMFileHandle(int fd, CUfileHandle_t cufile, FileHandleType type, char* path, void* client_data);
    EXPORT_VISIBLE CuCIMFileHandle(int fd,
                                   CUfileHandle_t cufile,
                                   FileHandleType type,
                                   char* path,
                                   void* client_data,
                                   uint64_t dev,
                                   uint64_t ino,
                                   int64_t mtime);
#    endif
    int fd;
    CUfileHandle_t cufile;
    FileHandleType type; /// 1: POSIX, 2: POSIX+ODIRECT, 4: MemoryMapped, 8: GPUDirect
    char* path;
    void* client_data;
    uint64_t hash_value;
    uint64_t dev;
    uint64_t ino;
    int64_t mtime;
};
#else
#    error "This platform is not supported!"
#endif

#endif // CUCIM_FILE_HANDLE_H
