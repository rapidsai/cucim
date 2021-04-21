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
#ifndef PYCUCIM_FILESYSTEM_PYDOC_H
#define PYCUCIM_FILESYSTEM_PYDOC_H

#include "../macros.h"

namespace cucim::filesystem::doc
{

// bool is_gds_available();
PYDOC(is_gds_available, R"doc(
Check if the GDS is available in the system.

Returns:
    True if libcufile.so is loaded and cuFileDriverOpen() API call succeeds.
)doc")


// std::shared_ptr<CuFileDriver> open(const char* file_path, const char* flags = "r", mode_t mode = 0644);
PYDOC(open, R"doc(
Open file with specific flags and mode.

'flags' can be one of the following flag string:

- "r": os.O_RDONLY

- "r+": os.O_RDWR

- "w": os.O_RDWR | os.O_CREAT | os.O_TRUNC

- "a": os.O_RDWR | os.O_CREAT

In addition to above flags, the method append os.O_CLOEXEC and os.O_DIRECT by default.

The following is optional flags that can be added to above string:

- 'p': Use POSIX APIs only (first try to open with O_DIRECT). It does not use GDS.

- 'n': Do not add O_DIRECT flag.

- 'm': Use memory-mapped file. This flag is supported only for the read-only file descriptor.

When 'm' is used, `PROT_READ` and `MAP_SHARED` are used for the parameter of mmap() function.

Args:
    file_path: A file path to open.
    flags: File flags in string. Default value is "r".
    mode: A file mode. Default value is '0o644'.
Returns:
    An object of CuFileDriver.
)doc")


// bool close(const std::shared_ptr<CuFileDriver>& fd);
PYDOC(close, R"doc(
Closes the given file driver.

Args:
    fd: An CuFileDriver object.
Returns:
    True if succeed, False otherwise.
)doc")

// ssize_t pread(void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);
PYDOC(pread, R"doc(
Reads up to `count` bytes from file driver `fd` at offset `offset` (from the start of the file) into the buffer
`buf` starting at offset `buf_offset`. The file offset is not changed.

Args:
    fd: An object of CuFileDriver.
    buf: A buffer where read bytes are stored. Buffer can be either in CPU memory or (CUDA) GPU memory.
    count: The number of bytes to read.
    file_offset: An offset from the start of the file.
    buf_offset: An offset from the start of the buffer. Default value is 0.
Returns:
    The number of bytes read if succeed, -1 otherwise.
)doc")

// ssize_t pread(const std::shared_ptr<CuFileDriver>& fd, const void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);
PYDOC(pwrite, R"doc(
Write up to `count` bytes from the buffer `buf` starting at offset `buf_offset` to the file driver `fd` at offset
`offset` (from the start of the file). The file offset is not changed.

Args:
    fd: An object of CuFileDriver.
    buf: A buffer where write bytes come from. Buffer can be either in CPU memory or (CUDA) GPU memory.
    count: The number of bytes to write.
    file_offset: An offset from the start of the file.
    buf_offset: An offset from the start of the buffer. Default value is 0.
Returns:
    The number of bytes written if succeed, -1 otherwise.
)doc")

// bool discard_page_cache(const char* file_path);
PYDOC(discard_page_cache, R"doc(
Discards a system (page) cache for the given file path.

Args:
    file_path: A file path to drop system cache.
Returns:
    True if succeed, False otherwise.
)doc")
} // namespace cucim::filesystem::doc
#endif // PYCUCIM_FILESYSTEM_PYDOC_H
