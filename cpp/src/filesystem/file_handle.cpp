/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/filesystem/file_handle.h"

#include <sys/stat.h>

#include "cucim/codec/hash_function.h"

CuCIMFileHandle::CuCIMFileHandle()
    : fd(-1),
      cufile(nullptr),
      type(FileHandleType::kUnknown),
      path(nullptr),
      client_data(nullptr),
      hash_value(0),
      dev(0),
      ino(0),
      mtime(0),
      own_fd(false)
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
                                 int64_t mtime,
                                 bool own_fd)
    : fd(fd), cufile(cufile), type(type), path(path), client_data(client_data), dev(dev), ino(ino), mtime(mtime), own_fd(own_fd)
{
    hash_value = cucim::codec::splitmix64_3(dev, ino, mtime);
}
