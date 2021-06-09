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

#include "cucim/filesystem/file_handle.h"

#include <sys/stat.h>

#include "cucim/codec/hash_function.h"

CuCIMFileHandle::CuCIMFileHandle()
    : fd(0),
      cufile(nullptr),
      type(FileHandleType::kUnknown),
      path(nullptr),
      client_data(nullptr),
      hash_value(0),
      dev(0),
      ino(0),
      mtime(0)
{
}

CuCIMFileHandle::CuCIMFileHandle(int fd, CUfileHandle_t cufile, FileHandleType type, char* path, void* client_data)
    : fd(fd), cufile(cufile), type(type), path(path), client_data(client_data)

{
    struct stat st;
    fstat(fd, &st);

    dev = static_cast<uint64_t>(st.st_dev);
    ino = static_cast<uint64_t>(st.st_ino);
    mtime = static_cast<uint64_t>(st.st_mtim.tv_nsec);
    hash_value = cucim::codec::splitmix64_3(dev, ino, mtime);
}

CuCIMFileHandle::CuCIMFileHandle(int fd,
                                 CUfileHandle_t cufile,
                                 FileHandleType type,
                                 char* path,
                                 void* client_data,
                                 uint64_t dev,
                                 uint64_t ino,
                                 int64_t mtime)
    : fd(fd), cufile(cufile), type(type), path(path), client_data(client_data), dev(dev), ino(ino), mtime(mtime)
{
    hash_value = cucim::codec::splitmix64_3(dev, ino, mtime);
}
