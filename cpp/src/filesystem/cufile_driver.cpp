/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "cucim/filesystem/cufile_driver.h"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <chrono>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "cucim/util/cuda.h"
#include "cucim/util/platform.h"
#include "cufile_stub.h"

#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))
#define ALIGN_DOWN(x, align_to) ((uint64_t)(x) & ~((uint64_t)(align_to)-1))


namespace cucim::filesystem
{
static constexpr unsigned int PAGE_SIZE = 4096;
static constexpr uint64_t DEFAULT_MAX_CACHE_SIZE = 128 << 20; // 128MiB
static CuFileStub s_cufile_stub;
static CuFileDriverInitializer s_cufile_initializer;
thread_local static CuFileDriverCache s_cufile_cache;
Mutex CuFileDriver::driver_mutex_;


static std::string get_fd_path(int fd)
{
    pid_t pid = getpid();
    ssize_t file_path_len = 0;

    char real_path[PATH_MAX];

    std::string src_path = fmt::format("/proc/{}/fd/{}", pid, fd);

    if ((file_path_len = readlink(src_path.c_str(), real_path, PATH_MAX - 1)) > 0)
    {
        real_path[file_path_len] = '\0';
    }
    else
    {
        throw std::runtime_error(fmt::format("Cannot get the real path from process entries ({})", strerror(errno)));
    }

    return std::string(real_path);
}

static int get_file_flags(const char* flags)
{
    int file_flags = -1;
    if (flags == nullptr || flags[0] == '\0')
    {
        return -1;
    }
    switch (flags[0])
    {
    case 'r':
        file_flags = O_RDONLY;
        if (flags[1] == '+')
        {
            file_flags = O_RDWR;
        }
        break;
    case 'w':
        file_flags = O_RDWR | O_CREAT | O_TRUNC;
        break;
    case 'a':
        file_flags = O_RDWR | O_CREAT;
        break;
    default:
        return -1;
    }


    file_flags |= O_CLOEXEC;

    return file_flags;
}

bool is_gds_available()
{
    return static_cast<bool>(s_cufile_initializer) && !cucim::util::is_in_wsl();
}

std::shared_ptr<CuFileDriver> open(const char* file_path, const char* flags, mode_t mode)
{
    bool use_o_direct = true;
    bool no_gds = false;
    bool use_mmap = false;
    int file_flags = get_file_flags(flags);

    for (const char* ch = (flags[1] == '+' ? &flags[2] : &flags[1]); *ch; ch++)
        switch (*ch)
        {
        case 'n':
            use_o_direct = false;
            break;
        case 'p':
            no_gds = true;
            break;
        case 'm':
            use_mmap = true;
            break;
        }
    if (use_o_direct)
    {
        file_flags |= O_DIRECT;
    }

    if (file_flags < 0)
    {
        return std::shared_ptr<CuFileDriver>();
    }

    FileHandleType file_type = (file_flags & O_DIRECT ? FileHandleType::kPosixODirect : FileHandleType::kPosix);

    int fd = ::open(file_path, file_flags, mode);
    if (fd < 0)
    {
        if (errno == ENOENT)
        {
            throw std::invalid_argument(fmt::format("File '{}' doesn't exist!", file_path));
        }
        if (file_type == FileHandleType::kPosix)
        {
            throw std::invalid_argument(fmt::format("File '{}' cannot be open!", file_path));
        }
        else // if kFileHandlePosixODirect
        {
            file_flags &= ~O_DIRECT;
            fd = ::open(file_path, file_flags, mode);
            fmt::print(
                stderr, "The file {} doesn't support O_DIRECT. Trying to open the file without O_DIRECT\n", file_path);
            if (fd < 0)
            {
                throw std::invalid_argument(fmt::format("File '{}' cannot be open!", file_path));
            }
            file_type = FileHandleType::kPosix; // POSIX
        }
    }

    return std::make_shared<CuFileDriver>(fd, no_gds, use_mmap, file_path);
}

std::shared_ptr<CuFileDriver> open(int fd, bool no_gds, bool use_mmap)
{
    return std::make_shared<CuFileDriver>(fd, no_gds, use_mmap, nullptr);
}

CuFileDriver::CuFileDriver(int fd, bool no_gds, bool use_mmap, const char* file_path)
{
    if (file_path == nullptr || *file_path == '\0')
    {
        file_path_ = get_fd_path(fd);
    }
    else
    {
        file_path_ = file_path;
    }

    struct stat st;
    fstat(fd, &st);
    file_size_ = st.st_size;

    int flags;
    // Note: the following method cannot detect flags such as O_EXCL and O_TRUNC.
    flags = fcntl(fd, F_GETFL);
    if (flags < 0)
    {
        throw std::runtime_error(fmt::format("[Error] fcntl failed for fd {} ({})", fd, std::strerror(errno)));
    }
    file_flags_ = flags;

    FileHandleType file_type = (flags & O_DIRECT) ? FileHandleType::kPosixODirect : FileHandleType::kPosix;
    // Copy file path (Allocated memory would be freed at close() method.)
    char* file_path_cstr = static_cast<char*>(cucim_malloc(file_path_.size() + 1));
    memcpy(file_path_cstr, file_path_.c_str(), file_path_.size());
    file_path_cstr[file_path_.size()] = '\0';
    handle_ = std::make_shared<CuCIMFileHandle>(fd, nullptr, file_type, const_cast<char*>(file_path_cstr), this,
                                                static_cast<uint64_t>(st.st_dev), static_cast<uint64_t>(st.st_ino),
                                                static_cast<int64_t>(st.st_mtim.tv_nsec));

    CUfileError_t status;
    CUfileDescr_t cf_descr{}; // It is important to set zero!

    if ((file_type == FileHandleType::kPosixODirect || file_type == FileHandleType::kGPUDirect) && !no_gds &&
        !use_mmap && s_cufile_initializer)
    {
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&handle_->cufile, &cf_descr);
        if (status.err == CU_FILE_SUCCESS)
        {
            handle_->type = FileHandleType::kGPUDirect;
        }
        else
        {
            fmt::print(
                stderr,
                "[Error] cuFileHandleRegister fd: {} ({}), status: {}. Would work with cuCIM's compatibility mode.\n",
                fd, file_path_, cufileop_status_error(status.err));
        }
    }
    else if (use_mmap)
    {
        if (flags & (O_RDWR || O_WRONLY))
        {
            throw std::runtime_error(
                fmt::format("[Error] Memory-mapped IO for writable file descriptor is not supported!"));
        }

        mmap_ptr_ = mmap((void*)0, file_size_, PROT_READ, MAP_SHARED, fd, 0);
        if (mmap_ptr_ != MAP_FAILED)
        {
            handle_->type = FileHandleType::kMemoryMapped;
        }
        else
        {
            mmap_ptr_ = nullptr;
            throw std::runtime_error(fmt::format("[Error] failed to call mmap ({})", std::strerror(errno)));
        }
    }
}

bool close(const std::shared_ptr<CuFileDriver>& fd)
{
    return fd->close();
}
ssize_t pread(const std::shared_ptr<CuFileDriver>& fd, void* buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (fd != nullptr)
    {
        return fd->pread(buf, count, file_offset, buf_offset);
    }
    else
    {
        fmt::print(stderr, "fd (CuFileDriver) is null!\n");
        return -1;
    }
}
ssize_t pwrite(const std::shared_ptr<CuFileDriver>& fd, const void* buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (fd != nullptr)
    {
        return fd->pwrite(buf, count, file_offset, buf_offset);
    }
    else
    {
        fmt::print(stderr, "fd (CuFileDriver) is null!\n");
        return -1;
    }
}

bool discard_page_cache(const char* file_path)
{
    int fd = ::open(file_path, O_RDONLY);
    if (fd < 0)
    {
        return false;
    }
    if (::fdatasync(fd) < 0)
    {
        return false;
    }
    if (::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) < 0)
    {
        return false;
    }
    if (::close(fd) < 0)
    {
        return false;
    }
    return true;
}


CuFileDriverInitializer::CuFileDriverInitializer()
{
    // Initialize libcufile library
    s_cufile_stub.load();

    CUfileError_t status = cuFileDriverOpen();
    if (status.err == CU_FILE_SUCCESS)
    {
        is_available_ = true;
        CUfileDrvProps_t props;

        status = cuFileDriverGetProperties(&props);
        if (status.err == CU_FILE_SUCCESS)
        {
            // kb -> bytes
            max_device_cache_size_ = static_cast<uint64_t>(props.max_device_cache_size) << 10;
            max_host_cache_size_ = static_cast<uint64_t>(props.max_device_cache_size) << 10;
        }
        else
        {
            fmt::print(stderr, "cuFileDriverGetProperties() failed!\n");
        }
        // fmt::print(stderr, "CuFileDriver opened!\n");
    }
    else
    {
        is_available_ = false;
        max_device_cache_size_ = DEFAULT_MAX_CACHE_SIZE;
        max_host_cache_size_ = DEFAULT_MAX_CACHE_SIZE;

        // fmt::print(stderr, "[Warning] CuFileDriver cannot be open. Falling back to use POSIX file IO APIs.\n");
    }
}
CuFileDriverInitializer::~CuFileDriverInitializer()
{
    if (is_available_)
    {
        CUfileError_t status = cuFileDriverClose();
        if (status.err != CU_FILE_SUCCESS)
        {
            fmt::print(stderr, "Unable to close cuFileDriver ({})\n", cufileop_status_error(status.err));
        }
        else
        {
            // fmt::print(stderr, "CuFileDriver closed!\n");
        }
        is_available_ = false;
    }

    // Close cufile stub
    s_cufile_stub.unload();
}

CuFileDriverCache::CuFileDriverCache()
{
}
void* CuFileDriverCache::device_cache()
{
    if (device_cache_)
    {
        return device_cache_aligned_;
    }
    else
    {
        cudaError_t cuda_status;
        unsigned int cache_size = s_cufile_initializer.max_device_cache_size();
        CUDA_TRY(cudaMalloc(&device_cache_, PAGE_SIZE + cache_size));
        if (cuda_status)
        {
            throw std::bad_alloc();
        }
        device_cache_aligned_ = reinterpret_cast<void*>(ALIGN_UP(device_cache_, PAGE_SIZE));
        CUfileError_t status = cuFileBufRegister(device_cache_aligned_, cache_size, 0);
        if (status.err != CU_FILE_SUCCESS)
        {
            CUDA_TRY(cudaFree(device_cache_));
            device_cache_ = nullptr;
            device_cache_aligned_ = nullptr;
            if (cuda_status)
            {
                throw std::bad_alloc();
            }
            throw std::runtime_error("Failed to call cuFileBufRegister()!");
        }

        return device_cache_aligned_;
    }
}
void* CuFileDriverCache::host_cache()
{
    if (host_cache_)
    {
        return host_cache_aligned_;
    }
    else
    {
        if (posix_memalign(&host_cache_, PAGE_SIZE, s_cufile_initializer.max_host_cache_size()))
        {
            throw std::bad_alloc();
        }
        host_cache_aligned_ = host_cache_;

        return host_cache_aligned_;
    }
}
CuFileDriverCache::~CuFileDriverCache()
{

    if (device_cache_)
    {
        cudaError_t cuda_status;
        CUfileError_t status = cuFileBufDeregister(device_cache_aligned_);
        if (status.err != CU_FILE_SUCCESS)
        {
            fmt::print(stderr, "Failed on cuFileBufDeregister()! (status: {})\n", cufileop_status_error(status.err));
        }
        CUDA_TRY(cudaFree(device_cache_));
        if (cuda_status)
        {
            fmt::print(stderr, "Failed on cudaFree()!\n");
        }
        device_cache_ = nullptr;
        device_cache_aligned_ = nullptr;
    }
    if (host_cache_)
    {
        free(host_cache_);
        host_cache_ = nullptr;
        host_cache_aligned_ = nullptr;
    }
}
ssize_t CuFileDriver::pread(void* buf, size_t count, off_t file_offset, off_t buf_offset) const
{
    if (file_flags_ == -1)
    {
        fmt::print(stderr, "File is not open yet.\n");
        return -1;
    }
    if ((file_flags_ & O_ACCMODE) == O_WRONLY)
    {
        fmt::print(stderr, "The file is open with write-only mode!\n");
        return -1;
    }

    cudaError_t cuda_status;
    ssize_t total_read_cnt = 0;

    cudaPointerAttributes attributes;
    cudaMemoryType memory_type;

    FileHandleType file_type = handle_->type;

    CUDA_TRY(cudaPointerGetAttributes(&attributes, buf));
    if (cuda_status)
    {
        //        if (cuda_status == cudaErrorInvalidValue)
        //        {
        //            attributes.type = cudaMemoryTypeDevice;
        //        }
        //        else
        //        {
        return -1;
        //        }
    }
    memory_type = attributes.type;

    if (file_type == FileHandleType::kPosix)
    {
        if (memory_type != cudaMemoryTypeUnregistered)
        {
            uint64_t cache_size = s_cufile_initializer.max_host_cache_size();
            uint64_t remaining_size = count;
            ssize_t read_cnt;
            uint8_t* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());
            uint8_t* output_buf = static_cast<uint8_t*>(buf) + buf_offset;
            off_t read_offset = file_offset;
            while (true)
            {
                size_t bytes_to_copy = std::min(cache_size, remaining_size);

                if (bytes_to_copy == 0)
                {
                    break;
                }
                read_cnt = ::pread(handle_->fd, cache_buf, bytes_to_copy, read_offset);
                CUDA_TRY(cudaMemcpy(output_buf, cache_buf, bytes_to_copy, cudaMemcpyHostToDevice));
                if (cuda_status)
                {
                    return -1;
                }
                read_offset += read_cnt;
                output_buf += read_cnt;
                remaining_size -= read_cnt;

                total_read_cnt += bytes_to_copy;
            }
        }
        else
        {
            total_read_cnt = ::pread(handle_->fd, reinterpret_cast<char*>(buf) + buf_offset, count, file_offset);
        }
    }
    else if (file_type == FileHandleType::kMemoryMapped)
    {
        if (memory_type != cudaMemoryTypeUnregistered)
        {
            CUDA_TRY(cudaMemcpy(reinterpret_cast<char*>(buf) + buf_offset,
                                reinterpret_cast<char*>(mmap_ptr_) + file_offset, count, cudaMemcpyHostToDevice));
            if (cuda_status)
            {
                return -1;
            }
        }
        else
        {
            memcpy(reinterpret_cast<char*>(buf) + buf_offset, reinterpret_cast<char*>(mmap_ptr_) + file_offset, count);
        }
        total_read_cnt = count;
    }
    else if (memory_type == cudaMemoryTypeUnregistered || handle_->type == FileHandleType::kPosixODirect)
    {
        uint64_t buf_align = (reinterpret_cast<uint64_t>(buf) + buf_offset) % PAGE_SIZE;
        bool is_aligned = (buf_align == 0) && ((file_offset % PAGE_SIZE) == 0);

        if (is_aligned)
        {
            ssize_t read_cnt;
            size_t block_read_size = ALIGN_DOWN(count, PAGE_SIZE);
            if (block_read_size > 0)
            {
                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    read_cnt =
                        ::pread(handle_->fd, reinterpret_cast<char*>(buf) + buf_offset, block_read_size, file_offset);
                    total_read_cnt += read_cnt;
                }
                else
                {
                    uint64_t cache_size = s_cufile_initializer.max_host_cache_size();
                    uint64_t remaining_size = block_read_size;
                    ssize_t read_cnt;
                    uint8_t* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());
                    uint8_t* input_buf = static_cast<uint8_t*>(buf) + buf_offset;
                    off_t read_offset = file_offset;
                    while (true)
                    {
                        size_t bytes_to_copy = std::min(cache_size, remaining_size);

                        if (bytes_to_copy == 0)
                        {
                            break;
                        }

                        read_cnt = ::pread(handle_->fd, cache_buf, bytes_to_copy, read_offset);
                        CUDA_TRY(cudaMemcpy(input_buf, cache_buf, bytes_to_copy, cudaMemcpyHostToDevice));
                        if (cuda_status)
                        {
                            return -1;
                        }
                        read_offset += read_cnt;
                        input_buf += read_cnt;
                        remaining_size -= read_cnt;

                        total_read_cnt += bytes_to_copy;
                    }
                }
            }

            size_t remaining = count - block_read_size;
            if (remaining)
            {
                uint8_t internal_buf[PAGE_SIZE * 2]; // no need to initialize for pread()
                uint8_t* buf_pos = reinterpret_cast<uint8_t*>(ALIGN_UP(static_cast<uint8_t*>(internal_buf), PAGE_SIZE));

                // Read the remaining block (size of PAGE_SIZE)
                ssize_t read_cnt;
                read_cnt = ::pread(handle_->fd, buf_pos, PAGE_SIZE, block_read_size);
                if (read_cnt < 0)
                {
                    fmt::print(stderr, "Cannot read the remaining file content block! ({})\n", std::strerror(errno));
                    return -1;
                }
                // Copy a buffer to read, from the intermediate remaining block (buf_pos)
                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    memcpy(reinterpret_cast<uint8_t*>(buf) + buf_offset + block_read_size, buf_pos, remaining);
                }
                else
                {
                    CUDA_TRY(cudaMemcpy(reinterpret_cast<uint8_t*>(buf) + buf_offset + block_read_size, buf_pos,
                                        remaining, cudaMemcpyHostToDevice));
                    if (cuda_status)
                    {
                        return -1;
                    }
                }

                total_read_cnt += remaining;
            }
        }
        else
        {
            uint64_t cache_size = s_cufile_initializer.max_host_cache_size();
            uint8_t* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());

            off_t file_start_offset = ALIGN_DOWN(file_offset, PAGE_SIZE);
            off_t end_offset = count + file_offset;
            off_t end_boundary_offset = ALIGN_UP(end_offset, PAGE_SIZE);
            size_t large_block_size = end_boundary_offset - file_start_offset;
            off_t page_offset = file_offset - file_start_offset;
            uint8_t* output_buf = static_cast<uint8_t*>(buf) + buf_offset;

            if (large_block_size <= cache_size) // Optimize if bytes to load is less than cache_size
            {
                ssize_t read_cnt = ::pread(handle_->fd, cache_buf, large_block_size, file_start_offset);
                if (read_cnt < 0)
                {
                    fmt::print(stderr, "Cannot read the file content block! ({})\n", std::strerror(errno));
                    return -1;
                }

                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    memcpy(output_buf, cache_buf + page_offset, count);
                }
                else
                {
                    CUDA_TRY(cudaMemcpy(output_buf, cache_buf + page_offset, count, cudaMemcpyHostToDevice));
                    if (cuda_status)
                    {
                        return -1;
                    }
                }
                total_read_cnt += std::min(static_cast<size_t>(read_cnt - page_offset), count);
            }
            else
            {
                off_t overflow_offset = page_offset + count;
                size_t header_size = (overflow_offset > PAGE_SIZE) ? PAGE_SIZE - page_offset : count;
                size_t tail_size = (overflow_offset > PAGE_SIZE) ? end_offset - ALIGN_DOWN(end_offset, PAGE_SIZE) : 0;
                uint64_t body_remaining_size = count - header_size - tail_size;
                off_t read_offset = file_start_offset;

                size_t bytes_to_copy;
                ssize_t read_cnt;

                uint8_t internal_buf[PAGE_SIZE * 2]; // no need to initialize for pread()
                uint8_t* internal_buf_pos =
                    reinterpret_cast<uint8_t*>(ALIGN_UP(static_cast<uint8_t*>(internal_buf), PAGE_SIZE));


                // Handle the head part of the file content
                if (header_size)
                {
                    read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, read_offset);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the head part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }

                    bytes_to_copy = header_size;
                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(output_buf, internal_buf_pos + page_offset, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(
                            output_buf, internal_buf_pos + page_offset, bytes_to_copy, cudaMemcpyHostToDevice));
                        if (cuda_status)
                        {
                            return -1;
                        }
                    }

                    output_buf += bytes_to_copy;
                    read_offset += read_cnt;

                    total_read_cnt += bytes_to_copy;
                }

                // Copy n * PAGE_SIZE bytes
                while (true)
                {
                    size_t bytes_to_copy = std::min(cache_size, body_remaining_size);

                    if (bytes_to_copy == 0)
                    {
                        break;
                    }

                    read_cnt = ::pread(handle_->fd, cache_buf, bytes_to_copy, read_offset);
                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(output_buf, cache_buf, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(output_buf, cache_buf, bytes_to_copy, cudaMemcpyHostToDevice));
                        if (cuda_status)
                        {
                            return -1;
                        }
                    }
                    read_offset += read_cnt;
                    output_buf += read_cnt;
                    body_remaining_size -= read_cnt;

                    total_read_cnt += bytes_to_copy;
                }

                // Handle the tail part of the file content
                if (tail_size)
                {
                    //                memset(internal_buf_pos, 0, PAGE_SIZE); // no need to initialize for pread()
                    read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, read_offset);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the tail part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                    // Copy the region
                    bytes_to_copy = tail_size;
                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(output_buf, internal_buf_pos, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(output_buf, internal_buf_pos, bytes_to_copy, cudaMemcpyHostToDevice));
                        if (cuda_status)
                        {
                            return -1;
                        }
                    }
                    total_read_cnt += tail_size;
                }
            }
        }
    }
    else if (file_type == FileHandleType::kGPUDirect)
    {
        (void*)s_cufile_cache.device_cache(); // Lazy initialization

        ssize_t read_cnt = cuFileRead(handle_->cufile, reinterpret_cast<char*>(buf) + buf_offset, count, file_offset, 0);
        total_read_cnt += read_cnt;
        if (read_cnt < 0)
        {
            fmt::print(stderr, "Failed to read file with cuFileRead().\n");
            return -1;
        }
    }

    return total_read_cnt;
}
ssize_t CuFileDriver::pwrite(const void* buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (file_flags_ == -1)
    {
        fmt::print(stderr, "File is not open yet.\n");
        return -1;
    }
    if ((file_flags_ & O_ACCMODE) == O_RDONLY)
    {
        fmt::print(stderr, "The file is open with read-only mode!\n");
        return -1;
    }

    cudaError_t cuda_status;
    ssize_t total_write_cnt = 0;

    cudaPointerAttributes attributes;
    cudaMemoryType memory_type;

    FileHandleType file_type = handle_->type;

    CUDA_TRY(cudaPointerGetAttributes(&attributes, buf));
    if (cuda_status)
    {
        return -1;
    }
    memory_type = attributes.type;

    if (file_type == FileHandleType::kPosix)
    {
        if (memory_type != cudaMemoryTypeUnregistered)
        {
            uint64_t cache_size = s_cufile_initializer.max_host_cache_size();
            uint64_t remaining_size = count;
            ssize_t write_cnt;
            uint8_t* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());
            const uint8_t* input_buf = static_cast<const uint8_t*>(buf) + buf_offset;
            off_t write_offset = file_offset;
            while (true)
            {
                size_t bytes_to_copy = std::min(cache_size, remaining_size);

                if (bytes_to_copy == 0)
                {
                    break;
                }

                CUDA_TRY(cudaMemcpy(cache_buf, input_buf, bytes_to_copy, cudaMemcpyDeviceToHost));
                if (cuda_status)
                {
                    return -1;
                }
                write_cnt = ::pwrite(handle_->fd, cache_buf, bytes_to_copy, write_offset);
                write_offset += write_cnt;
                input_buf += write_cnt;
                remaining_size -= write_cnt;

                total_write_cnt += bytes_to_copy;
            }
        }
        else
        {
            total_write_cnt = ::pwrite(handle_->fd, reinterpret_cast<const char*>(buf) + buf_offset, count, file_offset);
        }
    }
    else if (file_type == FileHandleType::kMemoryMapped)
    {
        fmt::print(stderr, "[Error] pwrite() is not supported for Memory-mapped IO file type!\n");
        return -1;
    }
    else if (memory_type == cudaMemoryTypeUnregistered || handle_->type == FileHandleType::kPosixODirect)
    {
        uint64_t buf_align = (reinterpret_cast<uint64_t>(buf) + buf_offset) % PAGE_SIZE;
        bool is_aligned = (buf_align == 0) && ((file_offset % PAGE_SIZE) == 0);

        if (is_aligned)
        {
            ssize_t write_cnt;
            size_t block_write_size = ALIGN_DOWN(count, PAGE_SIZE);

            if (block_write_size > 0)
            {
                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    write_cnt = ::pwrite(
                        handle_->fd, reinterpret_cast<const char*>(buf) + buf_offset, block_write_size, file_offset);
                    total_write_cnt += write_cnt;
                }
                else
                {
                    uint64_t cache_size = s_cufile_initializer.max_host_cache_size();
                    uint64_t remaining_size = block_write_size;
                    ssize_t write_cnt;
                    uint8_t* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());
                    const uint8_t* input_buf = static_cast<const uint8_t*>(buf) + buf_offset;
                    off_t write_offset = file_offset;
                    while (true)
                    {
                        size_t bytes_to_copy = std::min(cache_size, remaining_size);

                        if (bytes_to_copy == 0)
                        {
                            break;
                        }

                        CUDA_TRY(cudaMemcpy(cache_buf, input_buf, bytes_to_copy, cudaMemcpyDeviceToHost));
                        if (cuda_status)
                        {
                            return -1;
                        }
                        write_cnt = ::pwrite(handle_->fd, cache_buf, bytes_to_copy, write_offset);
                        write_offset += write_cnt;
                        input_buf += write_cnt;
                        remaining_size -= write_cnt;

                        total_write_cnt += bytes_to_copy;
                    }
                }
            }

            size_t remaining = count - block_write_size;
            if (remaining)
            {
                uint8_t internal_buf[PAGE_SIZE * 2]{};
                uint8_t* internal_buf_pos =
                    reinterpret_cast<uint8_t*>(ALIGN_UP(static_cast<uint8_t*>(internal_buf), PAGE_SIZE));

                // Read the remaining block (size of PAGE_SIZE)
                ssize_t read_cnt;
                read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, block_write_size);
                if (read_cnt < 0)
                {
                    fmt::print(stderr, "Cannot read the remaining file content block! ({})\n", std::strerror(errno));
                    return -1;
                }
                // Overwrite a buffer to write, to the intermediate remaining block (internal_buf_pos)
                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    memcpy(internal_buf_pos, reinterpret_cast<const uint8_t*>(buf) + buf_offset + block_write_size,
                           remaining);
                }
                else
                {
                    CUDA_TRY(cudaMemcpy(internal_buf_pos,
                                        reinterpret_cast<const uint8_t*>(buf) + buf_offset + block_write_size,
                                        remaining, cudaMemcpyDeviceToHost));
                    if (cuda_status)
                    {
                        return -1;
                    }
                }
                // Write the constructed block
                write_cnt = ::pwrite(handle_->fd, internal_buf_pos, PAGE_SIZE, block_write_size);
                if (write_cnt < 0)
                {
                    fmt::print(stderr, "Cannot write the remaining file content! ({})\n", std::strerror(errno));
                    return -1;
                }

                total_write_cnt += remaining;
            }
        }
        else
        {
            uint64_t cache_size = s_cufile_initializer.max_host_cache_size();
            uint8_t* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());

            off_t file_start_offset = ALIGN_DOWN(file_offset, PAGE_SIZE);
            off_t end_offset = count + file_offset;
            off_t end_boundary_offset = ALIGN_UP(end_offset, PAGE_SIZE);
            size_t large_block_size = end_boundary_offset - file_start_offset;
            off_t page_offset = file_offset - file_start_offset;
            const uint8_t* input_buf = static_cast<const uint8_t*>(buf) + buf_offset;

            if (large_block_size <= cache_size) // Optimize if bytes to write is less than cache_size
            {
                memset(cache_buf, 0, PAGE_SIZE);
                ssize_t read_cnt = ::pread(handle_->fd, cache_buf, PAGE_SIZE, file_start_offset);
                if (read_cnt < 0)
                {
                    fmt::print(
                        stderr, "Cannot read the head part of the file content block! ({})\n", std::strerror(errno));
                    return -1;
                }
                if (large_block_size > PAGE_SIZE)
                {
                    read_cnt = ::pread(handle_->fd, cache_buf + large_block_size - PAGE_SIZE, PAGE_SIZE,
                                       end_boundary_offset - PAGE_SIZE);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the tail part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                }

                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    memcpy(cache_buf + page_offset, input_buf, count);
                }
                else
                {
                    CUDA_TRY(cudaMemcpy(cache_buf + page_offset, input_buf, count, cudaMemcpyDeviceToHost));
                    if (cuda_status)
                    {
                        return -1;
                    }
                }

                // Write the constructed block
                ssize_t write_cnt = ::pwrite(handle_->fd, cache_buf, large_block_size, file_start_offset);
                if (write_cnt < 0)
                {
                    fmt::print(stderr, "Cannot write the file content block! ({})\n", std::strerror(errno));
                    return -1;
                }

                total_write_cnt += std::min(static_cast<size_t>(write_cnt - page_offset), count);
            }
            else
            {
                off_t overflow_offset = page_offset + count;
                size_t header_size = (overflow_offset > PAGE_SIZE) ? PAGE_SIZE - page_offset : count;
                size_t tail_size = (overflow_offset > PAGE_SIZE) ? end_offset - ALIGN_DOWN(end_offset, PAGE_SIZE) : 0;
                uint64_t body_remaining_size = count - header_size - tail_size;
                off_t write_offset = file_start_offset;

                size_t bytes_to_copy;
                ssize_t read_cnt;
                ssize_t write_cnt;

                uint8_t internal_buf[PAGE_SIZE * 2]{};
                uint8_t* internal_buf_pos =
                    reinterpret_cast<uint8_t*>(ALIGN_UP(static_cast<uint8_t*>(internal_buf), PAGE_SIZE));
                // Handle the head part of the file content
                if (header_size)
                {
                    read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the head part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                    // Overwrite the region to write
                    bytes_to_copy = header_size;
                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(internal_buf_pos + page_offset, input_buf, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(
                            internal_buf_pos + page_offset, input_buf, bytes_to_copy, cudaMemcpyDeviceToHost));
                        if (cuda_status)
                        {
                            return -1;
                        }
                    }

                    // Write the constructed block
                    write_cnt = ::pwrite(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (write_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot write the head part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                    input_buf += bytes_to_copy;
                    write_offset += write_cnt;

                    total_write_cnt += bytes_to_copy;
                }

                // Copy n * PAGE_SIZE bytes
                while (true)
                {
                    size_t bytes_to_copy = std::min(cache_size, body_remaining_size);

                    if (bytes_to_copy == 0)
                    {
                        break;
                    }

                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(cache_buf, input_buf, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(cache_buf, input_buf, bytes_to_copy, cudaMemcpyDeviceToHost));
                        if (cuda_status)
                        {
                            return -1;
                        }
                    }
                    write_cnt = ::pwrite(handle_->fd, cache_buf, bytes_to_copy, write_offset);
                    write_offset += write_cnt;
                    input_buf += write_cnt;
                    body_remaining_size -= write_cnt;

                    total_write_cnt += bytes_to_copy;
                }

                // Handle the tail part of the file content
                if (tail_size)
                {
                    memset(internal_buf_pos, 0, PAGE_SIZE);
                    read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the tail part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                    // Overwrite the region to write
                    bytes_to_copy = tail_size;
                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(internal_buf_pos, input_buf, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(internal_buf_pos, input_buf, bytes_to_copy, cudaMemcpyDeviceToHost));
                        if (cuda_status)
                        {
                            return -1;
                        }
                    }

                    // Write the constructed block
                    write_cnt = ::pwrite(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (write_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot write the tail part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                    total_write_cnt += tail_size;
                }
            }
        }
    }
    else if (file_type == FileHandleType::kGPUDirect)
    {
        (void*)s_cufile_cache.device_cache(); // Lazy initialization

        ssize_t write_cnt =
            cuFileWrite(handle_->cufile, reinterpret_cast<const char*>(buf) + buf_offset, count, file_offset, 0);
        if (write_cnt < 0)
        {
            fmt::print(stderr, "[cuFile Error] {}\n", CUFILE_ERRSTR(write_cnt));
            return -1;
        }
        total_write_cnt += write_cnt;
    }
    // Update file size
    if (total_write_cnt > 0)
    {
        file_size_ = std::max(file_size_, file_offset + static_cast<size_t>(total_write_cnt));
    }

    return total_write_cnt;
}
bool CuFileDriver::close()
{
    if (handle_->cufile)
    {
        cuFileHandleDeregister(handle_->cufile);

    }
    if (mmap_ptr_)
    {
        int err = munmap(mmap_ptr_, file_size_);
        if (err < 0)
        {
            fmt::print(stderr, "[Error] Cannot call munmap() ({})\n", std::strerror(errno));
        }
        mmap_ptr_ = nullptr;
    }
    if (handle_->fd != -1)
    {
        // If block write was used
        if ((file_flags_ & O_RDWR) &&
            (handle_->type == FileHandleType::kGPUDirect || handle_->type == FileHandleType::kPosixODirect))
        {
            // Truncate file assuming that `file_size_` is up to date during pwrite() calls
            int err = ::ftruncate(handle_->fd, file_size_);
            if (err < 0)
            {
                fmt::print(stderr, "[Error] Cannot resize the file {} to {} ({})\n", handle_->path, file_size_,
                           std::strerror(errno));
            }
        }
        handle_ = nullptr;
    }
    file_path_.clear();
    file_size_ = 0;
    file_flags_ = -1;
    return true;
}

filesystem::Path CuFileDriver::path() const
{
    return file_path_;
}

CuFileDriver::~CuFileDriver()
{
    close();
}

} // namespace cucim::filesystem
