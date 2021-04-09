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

#ifndef CUCIM_DYNAMIC_LIBRARY_H
#define CUCIM_DYNAMIC_LIBRARY_H

// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include "../macros/defines.h"

#include <string>

#if CUCIM_PLATFORM_LINUX
#    include <dlfcn.h>
#else
#    error "This platform is not supported!"
#endif

namespace cucim
{
namespace dynlib
{

#if CUCIM_PLATFORM_LINUX
using LibraryHandle = void*;
#else
#    error "This platform is not supported!"
#endif

template <typename T>
T get_library_symbol(LibraryHandle libHandle, const char* name)
{
#if CUCIM_PLATFORM_LINUX
    return reinterpret_cast<T>(::dlsym(libHandle, name));
#else
#    error "This platform is not supported!"
#endif
}

inline LibraryHandle load_library(const char* library_name)
{
#if CUCIM_PLATFORM_LINUX
    LibraryHandle handle = dlopen(library_name, RTLD_LAZY);
#else
#    error "This platform is not supported!"
#endif
    return handle;
}

inline std::string get_last_load_library_error()
{
#if CUCIM_PLATFORM_LINUX
    return dlerror();
#else
#    error "This platform is not supported!"
#endif
}

inline void unload_library(LibraryHandle library_handle)
{
    if (library_handle)
    {
#if CUCIM_PLATFORM_LINUX
        ::dlclose(library_handle);
#else
#    error "This platform is not supported!"
#endif
    }
}

} // namespace dynlib
} // namespace cucim

#endif // CUCIM_DYNAMIC_LIBRARY_H
