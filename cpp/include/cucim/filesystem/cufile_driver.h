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
#ifndef CUCIM_CUFILE_DRIVER_H
#define CUCIM_CUFILE_DRIVER_H

#include "file_handle.h"
#include "file_path.h"
#include <memory>
#include <mutex>

namespace cucim::filesystem
{

using Mutex = std::mutex;
using ScopedLock = std::scoped_lock<Mutex>;


// Forward declaration.
class CuFileDriver;


/**
 * @brief Check if the GDS is available in the system.
 *
 * @return true if libcufile.so is loaded and cuFileDriverOpen() API call succeeds.
 */
bool EXPORT_VISIBLE is_gds_available();

/**
 * Open file with specific flags and mode.
 *
 * 'flags' can be one of the following flag string:
 * - "r": O_RDONLY
 * - "r+": O_RDWR
 * - "w": O_RDWR | O_CREAT | O_TRUNC
 * - "a": O_RDWR | O_CREAT
 * In addition to above flags, the method append O_CLOEXEC and O_DIRECT by default.
 *
 * The following is optional flags that can be added to above string:
 * - 'p': Use POSIX APIs only (first try to open with O_DIRECT). It does not use GDS.
 * - 'n': Do not add O_DIRECT flag.
 * - 'm': Use memory-mapped file. This flag is supported only for the read-only file descriptor.
 *
 * When 'm' is used, `PROT_READ` and `MAP_SHARED` are used for the parameter of mmap() function.
 *
 * @param file_path A file path to open.
 * @param flags File flags in string. Default value is "r".
 * @param mode A file mode. Default value is '0644'.
 * @return a std::shared_ptr object of CuFileDriver.
 */
std::shared_ptr<CuFileDriver> EXPORT_VISIBLE open(const char* file_path, const char* flags = "r", mode_t mode = 0644);

/**
 * Open file with existing file descriptor.
 *
 * @param fd A file descriptor. To use GDS, fd needs to be opened with O_DIRECT flag.
 * @param no_gds true if you do not want to use GDS. Default value is `false`.
 * @param use_mmap true if you want to use memory-mapped IO. This flag is supported only for the read-only file descriptor. Default value is `false`.
 * @return A std::shared_ptr object of CuFileDriver.
 */
std::shared_ptr<CuFileDriver> EXPORT_VISIBLE open(int fd, bool no_gds = false, bool use_mmap = false);

/**
 * Close the given file driver.
 *
 * @param fd A std::shared_ptr object of CuFileDriver.
 * @return true if succeed, false otherwise.
 */
bool EXPORT_VISIBLE close(const std::shared_ptr<CuFileDriver>& fd);

/**
 * Read up to `count` bytes from file driver `fd` at offset `file_offset` (from the start of the file) into the buffer
 * `buf` at offset `buf_offset`. The file offset is not changed.
 *
 * @param fd A std::shared_ptr object of CuFileDriver.
 * @param buf A buffer where read bytes are stored. Buffer can be either in CPU memory or (CUDA) GPU memory.
 * @param count The number of bytes to read.
 * @param file_offset An offset from the start of the file.
 * @param buf_offset An offset from the start of the buffer. Default value is 0.
 * @return The number of bytes read if succeed, -1 otherwise.
 */
ssize_t EXPORT_VISIBLE pread(const std::shared_ptr<CuFileDriver>& fd, void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);

/**
 * Write up to `count` bytes from the buffer `buf` at offset `buf_offset` to the file driver `fd` at offset
 * `file_offset` (from the start of the file). The file offset is not changed.
 *
 *
 * @param fd A std::shared_ptr object of CuFileDriver.
 * @param buf A buffer where write bytes come from. Buffer can be either in CPU memory or (CUDA) GPU memory.
 * @param count The number of bytes to write.
 * @param file_offset An offset from the start of the file.
 * @param buf_offset An offset from the start of the buffer. Default value is 0.
 * @return The number of bytes written if succeed, -1 otherwise.
 */
ssize_t EXPORT_VISIBLE pwrite(const std::shared_ptr<CuFileDriver>& fd, const void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);

/**
 * Discard a system (page) cache for the given file path.
 * @param file_path A file path to drop system cache.
 * @return true if succeed, false otherwise.
 */
bool EXPORT_VISIBLE discard_page_cache(const char* file_path);

class CuFileDriverInitializer
{
public:
    CuFileDriverInitializer();

    inline operator bool() const
    {
        return is_available_;
    }
    inline uint64_t max_device_cache_size() const
    {
        return max_device_cache_size_;
    }
    inline uint64_t max_host_cache_size() const
    {
        return max_host_cache_size_;
    }

    ~CuFileDriverInitializer();

private:
    bool is_available_ = false;
    uint64_t max_device_cache_size_ = 0;
    uint64_t max_host_cache_size_ = 0;
};

class CuFileDriverCache
{
public:
    CuFileDriverCache();

    void* device_cache();
    void* host_cache();

    inline bool is_device_cache_available()
    {
        return !!device_cache_;
    }
    inline bool is_host_cache_available()
    {
        return !!host_cache_;
    }

    ~CuFileDriverCache();

private:
    void* device_cache_ = nullptr;
    void* device_cache_aligned_ = nullptr;
    void* host_cache_ = nullptr;
    void* host_cache_aligned_ = nullptr;
};

class EXPORT_VISIBLE CuFileDriver : public std::enable_shared_from_this<CuFileDriver>
{
public:
    CuFileDriver() = delete;

    CuFileDriver(int fd, bool no_gds = false, bool use_mmap = false, const char* file_path = nullptr);

    ssize_t pread(void* buf, size_t count, off_t file_offset, off_t buf_offset = 0) const;
    ssize_t pwrite(const void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);

    bool close();

    filesystem::Path path() const;

    ~CuFileDriver();

private:
    static Mutex driver_mutex_; // TODO: not used yet.

    std::string file_path_;
    size_t file_size_ = 0;
    int file_flags_ = -1;
    void* mmap_ptr_ = nullptr;
    std::shared_ptr<CuCIMFileHandle> handle_;
};

} // namespace cucim::filesystem

#endif // CUCIM_CUFILE_DRIVER_H
