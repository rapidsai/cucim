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
#ifndef PYCUCIM_CUFILE_PYDOC_H
#define PYCUCIM_CUFILE_PYDOC_H

#include "../macros.h"

namespace cucim::filesystem::doc::CuFileDriver
{

// CuFileDriver(int fd, bool no_gds = false, bool use_mmap = false, const char* file_path = "");
PYDOC(CuFileDriver, R"doc(
Constructor of CuFileDriver.

Args:
    fd: A file descriptor (in `int` type) which is available through `os.open()` method.
    no_gds: If True, use POSIX APIs only even when GDS can be supported for the file.
    use_mmap: If True, use memory-mapped IO. This flag is supported only for the read-only file descriptor. Default value is `False`.
    file_path: A file path for the file descriptor. It would retrieve the absolute file path of the file descriptor if not specified.
)doc")

// ssize_t pread(void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);
PYDOC(pread, R"doc(
Reads up to `count` bytes from the file driver at offset `file_offset` (from the start of the file) into the buffer
`buf` starting at offset `buf_offset`. The file offset is not changed.

Args:
    buf: A buffer where read bytes are stored. Buffer can be either in CPU memory or (CUDA) GPU memory.
    count: The number of bytes to read.
    file_offset: An offset from the start of the file.
    buf_offset: An offset from the start of the buffer. Default value is 0.
Returns:
    The number of bytes read if succeed, -1 otherwise.
)doc")

// ssize_t pwrite(const void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);
PYDOC(pwrite, R"doc(
Reads up to `count` bytes from the file driver at offset `file_offset` (from the start of the file) into the buffer
`buf` starting at offset `buf_offset`. The file offset is not changed.

Args:
    buf: A buffer where write bytes come from. Buffer can be either in CPU memory or (CUDA) GPU memory.
    count: The number of bytes to write.
    file_offset: An offset from the start of the file.
    buf_offset: An offset from the start of the buffer. Default value is 0.
Returns:
    The number of bytes written if succeed, -1 otherwise.
)doc")

// bool close();
PYDOC(close, R"doc(
Closes opened file if not closed.
)doc")

} // namespace cucim::filesystem::doc::CuFileDriver

#endif // PYCUCIM_CUFILE_PYDOC_H
