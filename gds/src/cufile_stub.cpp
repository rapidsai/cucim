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

#include "cufile_stub.h"
#include "cucim/dynlib/helper.h"
#include "cucim/util/platform.h"

#include <cstdio>

#define IMPORT_FUNCTION(handle, name) impl_##name = cucim::dynlib::get_library_symbol<t_##name>(handle, #name);

typedef CUfileError_t (*t_cuFileHandleRegister)(CUfileHandle_t* fh, CUfileDescr_t* descr);
typedef void (*t_cuFileHandleDeregister)(CUfileHandle_t fh);
typedef CUfileError_t (*t_cuFileBufRegister)(const void* devPtr_base, size_t length, int flags);
typedef CUfileError_t (*t_cuFileBufDeregister)(const void* devPtr_base);
typedef ssize_t (*t_cuFileRead)(CUfileHandle_t fh, void* devPtr_base, size_t size, off_t file_offset, off_t devPtr_offset);
typedef ssize_t (*t_cuFileWrite)(
    CUfileHandle_t fh, const void* devPtr_base, size_t size, off_t file_offset, off_t devPtr_offset);
typedef CUfileError_t (*t_cuFileDriverOpen)(void);
typedef CUfileError_t (*t_cuFileDriverClose)(void);
typedef CUfileError_t (*t_cuFileDriverGetProperties)(CUfileDrvProps_t* props);
typedef CUfileError_t (*t_cuFileDriverSetPollMode)(bool poll, size_t poll_threshold_size);
typedef CUfileError_t (*t_cuFileDriverSetMaxDirectIOSize)(size_t max_direct_io_size);
typedef CUfileError_t (*t_cuFileDriverSetMaxCacheSize)(size_t max_cache_size);
typedef CUfileError_t (*t_cuFileDriverSetMaxPinnedMemSize)(size_t max_pinned_size);

static t_cuFileHandleRegister impl_cuFileHandleRegister = nullptr;
static t_cuFileHandleDeregister impl_cuFileHandleDeregister = nullptr;
static t_cuFileBufRegister impl_cuFileBufRegister = nullptr;
static t_cuFileBufDeregister impl_cuFileBufDeregister = nullptr;
static t_cuFileRead impl_cuFileRead = nullptr;
static t_cuFileWrite impl_cuFileWrite = nullptr;
static t_cuFileDriverOpen impl_cuFileDriverOpen = nullptr;
static t_cuFileDriverClose impl_cuFileDriverClose = nullptr;
static t_cuFileDriverGetProperties impl_cuFileDriverGetProperties = nullptr;
static t_cuFileDriverSetPollMode impl_cuFileDriverSetPollMode = nullptr;
static t_cuFileDriverSetMaxDirectIOSize impl_cuFileDriverSetMaxDirectIOSize = nullptr;
static t_cuFileDriverSetMaxCacheSize impl_cuFileDriverSetMaxCacheSize = nullptr;
static t_cuFileDriverSetMaxPinnedMemSize impl_cuFileDriverSetMaxPinnedMemSize = nullptr;


class CuFileStub
{
public:
    void load()
    {
#if !CUCIM_SUPPORT_GDS
        return;
#endif

#if !CUCIM_STATIC_GDS
        fprintf(stderr, "##load()\n");
        if (handle_ == nullptr)
        {
            fprintf(stderr, "##load(): handle_ == nullptr\n");
            handle_ = cucim::dynlib::load_library("libcufile.so");
            if (handle_ == nullptr)
            {
                fprintf(stderr, "##load(): load_library == nullptr\n");
                return;
            }
            fprintf(stderr, "##load(): load_library != nullptr\n");
            IMPORT_FUNCTION(handle_, cuFileDriverOpen);
            IMPORT_FUNCTION(handle_, cuFileHandleRegister);
            IMPORT_FUNCTION(handle_, cuFileHandleDeregister);
            IMPORT_FUNCTION(handle_, cuFileBufRegister);
            IMPORT_FUNCTION(handle_, cuFileBufDeregister);
            IMPORT_FUNCTION(handle_, cuFileRead);
            IMPORT_FUNCTION(handle_, cuFileWrite);
            IMPORT_FUNCTION(handle_, cuFileDriverOpen);
            IMPORT_FUNCTION(handle_, cuFileDriverClose);
            IMPORT_FUNCTION(handle_, cuFileDriverGetProperties);
            IMPORT_FUNCTION(handle_, cuFileDriverSetPollMode);
            IMPORT_FUNCTION(handle_, cuFileDriverSetMaxDirectIOSize);
            IMPORT_FUNCTION(handle_, cuFileDriverSetMaxCacheSize);
            IMPORT_FUNCTION(handle_, cuFileDriverSetMaxPinnedMemSize);
        }
#endif
    }
    void unload()
    {
#if !CUCIM_SUPPORT_GDS
        return;
#endif

#if !CUCIM_STATIC_GDS
        fprintf(stderr, "##unload()\n");
        if (handle_)
        {
            fprintf(stderr, "##Unloading libcufile.so in unload()\n");
            cucim::dynlib::unload_library(handle_);
            handle_ = nullptr;

            impl_cuFileDriverOpen = nullptr;
            impl_cuFileHandleRegister = nullptr;
            impl_cuFileHandleDeregister = nullptr;
            impl_cuFileBufRegister = nullptr;
            impl_cuFileBufDeregister = nullptr;
            impl_cuFileRead = nullptr;
            impl_cuFileWrite = nullptr;
            impl_cuFileDriverOpen = nullptr;
            impl_cuFileDriverClose = nullptr;
            impl_cuFileDriverGetProperties = nullptr;
            impl_cuFileDriverSetPollMode = nullptr;
            impl_cuFileDriverSetMaxDirectIOSize = nullptr;
            impl_cuFileDriverSetMaxCacheSize = nullptr;
            impl_cuFileDriverSetMaxPinnedMemSize = nullptr;
        }
        else
        {
            fprintf(stderr, "##Unloading libcufile.so in unload(): other path\n");
        }
#endif
    }
    ~CuFileStub()
    {
        fprintf(stderr, "##Destructor CuFileStub for g_cufile_stub\n");
        // Note: unload() would be called explicitly by CuFileDriverInitializer to unload the shared library after
        // calling cuFileDriverClose() in CuFileDriverInitializer::~CuFileDriverInitializer()
        //        unload();
    }

private:
    cucim::dynlib::LibraryHandle handle_ = nullptr;
};

static CuFileStub g_cufile_stub;

#if __cplusplus
extern "C"
{
#endif

    void open_cufile_stub()
    {
        g_cufile_stub.load();
    }
    void close_cufile_stub()
    {
        g_cufile_stub.unload();
    }

#if !CUCIM_STATIC_GDS
    CUfileError_t cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr)
    {
        if (impl_cuFileHandleRegister)
        {
            fprintf(stderr, "##cuFileHandleRegister() called\n");
            return impl_cuFileHandleRegister(fh, descr);
        }

        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    void cuFileHandleDeregister(CUfileHandle_t fh)
    {
        if (impl_cuFileHandleDeregister)
        {
            fprintf(stderr, "##impl_cuFileHandleDeregister() called\n");
            impl_cuFileHandleDeregister(fh);
        }
    }

    CUfileError_t cuFileBufRegister(const void* devPtr_base, size_t length, int flags)
    {
        if (impl_cuFileBufRegister)
        {
            fprintf(stderr, "##impl_cuFileBufRegister() called\n");
            return impl_cuFileBufRegister(devPtr_base, length, flags);
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    CUfileError_t cuFileBufDeregister(const void* devPtr_base)
    {
        if (impl_cuFileBufDeregister)
        {
            fprintf(stderr, "##impl_cuFileBufDeregister() called\n");
            return impl_cuFileBufDeregister(devPtr_base);
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    ssize_t cuFileRead(CUfileHandle_t fh, void* devPtr_base, size_t size, off_t file_offset, off_t devPtr_offset)
    {
        if (impl_cuFileRead)
        {
            fprintf(stderr, "##impl_cuFileRead() called\n");
            return impl_cuFileRead(fh, devPtr_base, size, file_offset, devPtr_offset);
        }
        return -1;
    }

    ssize_t cuFileWrite(CUfileHandle_t fh, const void* devPtr_base, size_t size, off_t file_offset, off_t devPtr_offset)
    {
        if (impl_cuFileWrite)
        {
            fprintf(stderr, "##impl_cuFileWrite() called\n");
            return impl_cuFileWrite(fh, devPtr_base, size, file_offset, devPtr_offset);
        }
        return -1;
    }

    CUfileError_t cuFileDriverOpen(void)
    {
        // GDS v1.0.0 does not support WSL and executing this can cause the following error:
        //    Assertion failure, file index :cufio-udev  line :143
        // So we do not call impl_cuFileDriverOpen() here if the current platform is WSL.
        if (impl_cuFileDriverOpen)
        {
            // If not in WSL, call impl_cuFileDriverOpen()
            if (!cucim::util::is_in_wsl())
            {
                return impl_cuFileDriverOpen();
            }
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    CUfileError_t cuFileDriverClose(void)
    {
        if (impl_cuFileDriverClose)
        {
            fprintf(stderr, "##impl_cuFileDriverClose() called\n");
            return impl_cuFileDriverClose();
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t* props)
    {
        if (impl_cuFileDriverGetProperties)
        {
            fprintf(stderr, "##impl_cuFileDriverGetProperties() called\n");
            return impl_cuFileDriverGetProperties(props);
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    CUfileError_t cuFileDriverSetPollMode(bool poll, size_t poll_threshold_size)
    {
        if (impl_cuFileDriverSetPollMode)
        {
            fprintf(stderr, "##impl_cuFileDriverSetPollMode() called\n");
            return impl_cuFileDriverSetPollMode(poll, poll_threshold_size);
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)
    {
        if (impl_cuFileDriverSetMaxDirectIOSize)
        {
            fprintf(stderr, "##impl_cuFileDriverSetMaxDirectIOSize() called\n");
            return impl_cuFileDriverSetMaxDirectIOSize(max_direct_io_size);
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    CUfileError_t cuFileDriverSetMaxCacheSize(size_t max_cache_size)
    {
        if (impl_cuFileDriverSetMaxCacheSize)
        {
            fprintf(stderr, "##impl_cuFileDriverSetMaxCacheSize() called\n");
            return impl_cuFileDriverSetMaxCacheSize(max_cache_size);
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }

    CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)
    {
        if (impl_cuFileDriverSetMaxPinnedMemSize)
        {
            fprintf(stderr, "##impl_cuFileDriverSetMaxPinnedMemSize() called\n");
            return impl_cuFileDriverSetMaxPinnedMemSize(max_pinned_size);
        }
        return CUfileError_t{ CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS };
    }
#endif

#if __cplusplus
}
#endif
