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

#include "cucim/logger/timer.h"
#include "config.h"

#include <catch2/catch.hpp>
#include <fmt/format.h>
#include <cucim/filesystem/cufile_driver.h>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <string_view>
// Test
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))

#define CUDA_ERROR(stmt)                                                                                               \
    {                                                                                                                  \
        cuda_status = stmt;                                                                                            \
        if (cudaSuccess != cuda_status)                                                                                \
        {                                                                                                              \
            INFO(fmt::format("Error message: {}", cudaGetErrorString(cuda_status)));                                   \
            REQUIRE(cudaSuccess == cuda_status);                                                                       \
        }                                                                                                              \
    }

#define POSIX_ERROR(stmt)                                                                                              \
    {                                                                                                                  \
        err = stmt;                                                                                                    \
        if (err < 0)                                                                                                   \
        {                                                                                                              \
            INFO(fmt::format("Error message: {}", std::strerror(errno)));                                              \
            REQUIRE(err >= 0);                                                                                         \
        }                                                                                                              \
    }

static void create_test_file(const char* file_name, int size)
{
    int fd = open(file_name, O_RDWR | O_CREAT | O_TRUNC, 0666);
    char test_data[size];
    ::srand(0);
    for (int i = 0; i < size; i++)
    {
        test_data[i] = ::rand() % 256; // or i % 256;
    }
    ssize_t write_cnt = write(fd, test_data, size);
    (void)write_cnt;
    assert(write_cnt == size);
    close(fd);
}

TEST_CASE("Verify libcufile usage", "[test_cufile.cpp]")
{
    cudaError_t cuda_status;
    int err;
    constexpr int BLOCK_SECTOR_SIZE = 4096;
    constexpr char const* test_w_flags[] = { "wpn", "wp", "wn", "w" };
    constexpr char const* test_flags_desc[] = { "regular file", "o_direct", "gds with no O_DIRECT", "gds with O_DIRECT" };
    constexpr int W_FLAG_LEN = sizeof(test_w_flags) / sizeof(test_w_flags[0]);
    constexpr char const* test_r_flags[] = { "rpn", "rp", "rn", "r" };
    // clang-format off
    constexpr int test_buf_offsets[] =  { 0,   0,   0,    0,  0,     0,    0,    0,    0,            0 };
    constexpr int test_file_offsets[] = { 0, 500,   0,    0, 400,  400, 4000, 4500, 4500, 4096 * 2 - 1 };
    constexpr int test_counts[] =       { 0,   0, 500, 4097, 500, 4097,  500,  500, 4097,          500 };
    // clang-format on
    constexpr int TEST_PARAM_LEN = sizeof(test_counts) / sizeof(test_counts[0]);
    uint8_t test_data[BLOCK_SECTOR_SIZE * 3];

    std::string output_file = fmt::format("{}/test_cufile.raw", g_config.temp_folder);

    ::srand(777);
    for (int i = 0; i < BLOCK_SECTOR_SIZE * 3; i++)
    {
        test_data[i] = ::rand() % 256; // or (BLOCK_SECTOR_SIZE * 3 - i) % 256;
    }

    std::hash<std::string_view> str_hash;

    for (int test_param_index = 0; test_param_index < TEST_PARAM_LEN; ++test_param_index)
    {
        int test_buf_offset = test_buf_offsets[test_param_index];
        int test_file_offset = test_file_offsets[test_param_index];
        int test_count = test_counts[test_param_index];

        // Allocate memory
        uint8_t* unaligned_host = static_cast<uint8_t*>(malloc(test_count + test_buf_offset));
        uint8_t* aligned_host;
        POSIX_ERROR(posix_memalign(reinterpret_cast<void**>(&aligned_host), 512, test_count + test_buf_offset));

        uint8_t* unaligned_device;
        CUDA_ERROR(cudaMalloc(&unaligned_device, test_count + test_buf_offset + BLOCK_SECTOR_SIZE));
        uint8_t* aligned_device = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_device, BLOCK_SECTOR_SIZE));

        uint8_t* unaligned_device_host;
        CUDA_ERROR(cudaMallocHost(&unaligned_device_host, test_count + test_buf_offset + BLOCK_SECTOR_SIZE));
        uint8_t* aligned_device_host = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_device_host, BLOCK_SECTOR_SIZE));

        uint8_t* unaligned_device_managed;
        CUDA_ERROR(cudaMallocManaged(&unaligned_device_managed, test_count + test_buf_offset + BLOCK_SECTOR_SIZE));
        uint8_t* aligned_device_managed =
            reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_device_managed, BLOCK_SECTOR_SIZE));

        SECTION(fmt::format("Write Test with different parameters (offset:{}, count:{})", test_file_offset, test_count))
        {
            {
                INFO("# unaligned_host");
                size_t reference_whole_hash = 0;
                size_t reference_hash = 0;
                ssize_t reference_write_cnt = 0;
                ssize_t reference_read_cnt = 0;
                for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                {
                    INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                     test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                    {
                        INFO(fmt::format("memory: unaligned_host \n"));
                        create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                        auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                        memcpy(unaligned_host + test_buf_offset, test_data, test_count);

                        ssize_t write_cnt = fd->pwrite(unaligned_host, test_count, test_file_offset, test_buf_offset);
                        if (flag_idx == 0)
                        {
                            reference_write_cnt = write_cnt;
                        }
                        else
                        {
                            REQUIRE(write_cnt == reference_write_cnt);
                        }

                        fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                        memset(unaligned_host, 0, test_count + test_buf_offset);

                        ssize_t read_cnt = fd->pread(unaligned_host, test_count, test_file_offset, test_buf_offset);
                        if (flag_idx == 0)
                        {
                            reference_read_cnt = read_cnt;
                        }
                        else
                        {
                            REQUIRE(read_cnt == reference_read_cnt);
                        }

                        size_t posix_hash = str_hash(std::string_view((char*)unaligned_host + test_buf_offset, read_cnt));

                        char file_data[BLOCK_SECTOR_SIZE * 4]{};
                        int fd2 = open(output_file.c_str(), O_RDONLY);
                        read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                        size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                        if (flag_idx == 0)
                        {
                            reference_hash = posix_hash;
                            reference_whole_hash = file_hash;
                        }
                        else
                        {
                            REQUIRE(reference_hash == posix_hash);
                            REQUIRE(reference_whole_hash == file_hash);
                        }
                    }
                }
            }
            {
                INFO("# aligned_host");
                size_t reference_whole_hash = 0;
                size_t reference_hash = 0;
                ssize_t reference_write_cnt = 0;
                ssize_t reference_read_cnt = 0;
                for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                {
                    INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                     test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                    {
                        INFO(fmt::format("memory: aligned_host \n"));
                        create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                        auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                        memcpy(aligned_host + test_buf_offset, test_data, test_count + test_buf_offset);
                        ssize_t write_cnt = fd->pwrite(aligned_host, test_count, test_file_offset, test_buf_offset);
                        if (flag_idx == 0)
                        {
                            reference_write_cnt = write_cnt;
                        }
                        else
                        {
                            REQUIRE(write_cnt == reference_write_cnt);
                        }

                        fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                        memset(aligned_host, 0, test_count + test_buf_offset);
                        ssize_t read_cnt = fd->pread(aligned_host, test_count, test_file_offset, test_buf_offset);
                        if (flag_idx == 0)
                        {
                            reference_read_cnt = read_cnt;
                        }
                        else
                        {
                            REQUIRE(read_cnt == reference_read_cnt);
                        }

                        size_t posix_hash = str_hash(std::string_view((char*)aligned_host + test_buf_offset, read_cnt));

                        char file_data[BLOCK_SECTOR_SIZE * 4]{};
                        int fd2 = open(output_file.c_str(), O_RDONLY);
                        read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                        size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                        if (flag_idx == 0)
                        {
                            reference_hash = posix_hash;
                            reference_whole_hash = file_hash;
                        }
                        else
                        {
                            REQUIRE(reference_hash == posix_hash);
                            REQUIRE(reference_whole_hash == file_hash);
                        }
                    }
                }
            }
            {
                // Device Memory
                {
                    INFO("# unaligned_device");
                    size_t reference_whole_hash = 0;
                    size_t reference_hash = 0;
                    ssize_t reference_write_cnt = 0;
                    ssize_t reference_read_cnt = 0;
                    for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                    {
                        INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                         test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                        {
                            INFO(fmt::format("memory: unaligned_device \n"));
                            create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                            auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                            cudaMemcpy(unaligned_device + test_buf_offset, test_data, test_count, cudaMemcpyHostToDevice);
                            ssize_t write_cnt =
                                fd->pwrite(unaligned_device, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_write_cnt = write_cnt;
                            }
                            else
                            {
                                REQUIRE(write_cnt == reference_write_cnt);
                            }

                            fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                            cudaMemset(unaligned_device, 0, test_count + test_buf_offset);
                            ssize_t read_cnt = fd->pread(unaligned_device, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_read_cnt = read_cnt;
                            }
                            else
                            {
                                REQUIRE(read_cnt == reference_read_cnt);
                            }

                            cudaMemcpy(unaligned_host, unaligned_device, test_count, cudaMemcpyDeviceToHost);
                            size_t posix_hash =
                                str_hash(std::string_view((char*)unaligned_host + test_buf_offset, read_cnt));

                            char file_data[BLOCK_SECTOR_SIZE * 4]{};
                            int fd2 = open(output_file.c_str(), O_RDONLY);
                            read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                            size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                            if (flag_idx == 0)
                            {
                                reference_hash = posix_hash;
                                reference_whole_hash = file_hash;
                            }
                            else
                            {
                                REQUIRE(reference_hash == posix_hash);
                                REQUIRE(reference_whole_hash == file_hash);
                            }
                        }
                    }
                }
                {
                    INFO("# aligned_device");
                    size_t reference_whole_hash = 0;
                    size_t reference_hash = 0;
                    ssize_t reference_write_cnt = 0;
                    ssize_t reference_read_cnt = 0;
                    for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                    {
                        INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                         test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                        {
                            INFO(fmt::format("memory: aligned_device \n"));
                            create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                            auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                            cudaMemcpy(aligned_device + test_buf_offset, test_data, test_count, cudaMemcpyHostToDevice);
                            ssize_t write_cnt = fd->pwrite(aligned_device, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_write_cnt = write_cnt;
                            }
                            else
                            {
                                REQUIRE(write_cnt == reference_write_cnt);
                            }

                            fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                            cudaMemset(aligned_device, 0, test_count + test_buf_offset);
                            ssize_t read_cnt = fd->pread(aligned_device, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_read_cnt = read_cnt;
                            }
                            else
                            {
                                REQUIRE(read_cnt == reference_read_cnt);
                            }

                            cudaMemcpy(aligned_host, aligned_device, test_count, cudaMemcpyDeviceToHost);
                            size_t posix_hash =
                                str_hash(std::string_view((char*)aligned_host + test_buf_offset, read_cnt));

                            char file_data[BLOCK_SECTOR_SIZE * 4]{};
                            int fd2 = open(output_file.c_str(), O_RDONLY);
                            read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                            size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                            if (flag_idx == 0)
                            {
                                reference_hash = posix_hash;
                                reference_whole_hash = file_hash;
                            }
                            else
                            {
                                REQUIRE(reference_hash == posix_hash);
                                REQUIRE(reference_whole_hash == file_hash);
                            }
                        }
                    }
                }

                // Pinned Host Memory
                {
                    INFO("# unaligned_device (pinned host)");
                    size_t reference_whole_hash = 0;
                    size_t reference_hash = 0;
                    ssize_t reference_write_cnt = 0;
                    ssize_t reference_read_cnt = 0;
                    for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                    {
                        INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                         test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                        {
                            INFO(fmt::format("memory: unaligned_device (pinned host)\n"));
                            create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                            auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                            cudaMemcpy(
                                unaligned_device_host + test_buf_offset, test_data, test_count, cudaMemcpyHostToDevice);
                            ssize_t write_cnt =
                                fd->pwrite(unaligned_device_host, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_write_cnt = write_cnt;
                            }
                            else
                            {
                                REQUIRE(write_cnt == reference_write_cnt);
                            }

                            fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                            cudaMemset(unaligned_device_host, 0, test_count + test_buf_offset);
                            ssize_t read_cnt =
                                fd->pread(unaligned_device_host, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_read_cnt = read_cnt;
                            }
                            else
                            {
                                REQUIRE(read_cnt == reference_read_cnt);
                            }

                            cudaMemcpy(unaligned_host, unaligned_device_host, test_count, cudaMemcpyDeviceToHost);
                            size_t posix_hash =
                                str_hash(std::string_view((char*)unaligned_host + test_buf_offset, read_cnt));

                            char file_data[BLOCK_SECTOR_SIZE * 4]{};
                            int fd2 = open(output_file.c_str(), O_RDONLY);
                            read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                            size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                            if (flag_idx == 0)
                            {
                                reference_hash = posix_hash;
                                reference_whole_hash = file_hash;
                            }
                            else
                            {
                                REQUIRE(reference_hash == posix_hash);
                                REQUIRE(reference_whole_hash == file_hash);
                            }
                        }
                    }
                }
                {
                    INFO("# aligned_device (pinned host)");
                    size_t reference_whole_hash = 0;
                    size_t reference_hash = 0;
                    ssize_t reference_write_cnt = 0;
                    ssize_t reference_read_cnt = 0;
                    for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                    {
                        INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                         test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                        {
                            INFO(fmt::format("memory: aligned_device (pinned host)\n"));
                            create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                            auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                            cudaMemcpy(
                                aligned_device_host + test_buf_offset, test_data, test_count, cudaMemcpyHostToDevice);
                            ssize_t write_cnt =
                                fd->pwrite(aligned_device_host, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_write_cnt = write_cnt;
                            }
                            else
                            {
                                REQUIRE(write_cnt == reference_write_cnt);
                            }

                            fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                            cudaMemset(aligned_device_host, 0, test_count + test_buf_offset);
                            ssize_t read_cnt =
                                fd->pread(aligned_device_host, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_read_cnt = read_cnt;
                            }
                            else
                            {
                                REQUIRE(read_cnt == reference_read_cnt);
                            }

                            cudaMemcpy(aligned_host, aligned_device_host, test_count, cudaMemcpyDeviceToHost);
                            size_t posix_hash =
                                str_hash(std::string_view((char*)aligned_host + test_buf_offset, read_cnt));

                            char file_data[BLOCK_SECTOR_SIZE * 4]{};
                            int fd2 = open(output_file.c_str(), O_RDONLY);
                            read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                            size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                            if (flag_idx == 0)
                            {
                                reference_hash = posix_hash;
                                reference_whole_hash = file_hash;
                            }
                            else
                            {
                                REQUIRE(reference_hash == posix_hash);
                                REQUIRE(reference_whole_hash == file_hash);
                            }
                        }
                    }
                }

                // ManageDevice Memory
                {
                    INFO("# unaligned_device (managed)");
                    size_t reference_whole_hash = 0;
                    size_t reference_hash = 0;
                    ssize_t reference_write_cnt = 0;
                    ssize_t reference_read_cnt = 0;
                    for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                    {
                        INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                         test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                        {
                            INFO(fmt::format("memory: unaligned_device (managed)\n"));
                            create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                            auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                            cudaMemcpy(unaligned_device_managed + test_buf_offset, test_data, test_count,
                                       cudaMemcpyHostToDevice);
                            ssize_t write_cnt =
                                fd->pwrite(unaligned_device_managed, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_write_cnt = write_cnt;
                            }
                            else
                            {
                                REQUIRE(write_cnt == reference_write_cnt);
                            }

                            fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                            cudaMemset(unaligned_device_managed, 0, test_count + test_buf_offset);
                            ssize_t read_cnt =
                                fd->pread(unaligned_device_managed, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_read_cnt = read_cnt;
                            }
                            else
                            {
                                REQUIRE(read_cnt == reference_read_cnt);
                            }

                            cudaMemcpy(unaligned_host, unaligned_device_managed, test_count, cudaMemcpyDeviceToHost);
                            size_t posix_hash =
                                str_hash(std::string_view((char*)unaligned_host + test_buf_offset, read_cnt));

                            char file_data[BLOCK_SECTOR_SIZE * 4]{};
                            int fd2 = open(output_file.c_str(), O_RDONLY);
                            read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                            size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                            if (flag_idx == 0)
                            {
                                reference_hash = posix_hash;
                                reference_whole_hash = file_hash;
                            }
                            else
                            {
                                REQUIRE(reference_hash == posix_hash);
                                REQUIRE(reference_whole_hash == file_hash);
                            }
                        }
                    }
                }

                {
                    INFO("# aligned_device (managed)");
                    size_t reference_whole_hash = 0;
                    size_t reference_hash = 0;
                    ssize_t reference_write_cnt = 0;
                    ssize_t reference_read_cnt = 0;
                    for (int flag_idx = 0; flag_idx < W_FLAG_LEN; ++flag_idx)
                    {
                        INFO(fmt::format("flag_index: {} ({})\n  count: {}\n  file_offset: {}  buf_offset: {}\n", flag_idx,
                                         test_flags_desc[flag_idx], test_count, test_file_offset, test_buf_offset));
                        {
                            INFO(fmt::format("memory: aligned_device (managed)\n"));
                            create_test_file(output_file.c_str(), BLOCK_SECTOR_SIZE * 3);

                            auto fd = cucim::filesystem::open(output_file.c_str(), test_w_flags[flag_idx]);
                            cudaMemcpy(aligned_device_managed + test_buf_offset, test_data, test_count,
                                       cudaMemcpyHostToDevice);
                            ssize_t write_cnt =
                                fd->pwrite(aligned_device_managed, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_write_cnt = write_cnt;
                            }
                            else
                            {
                                REQUIRE(write_cnt == reference_write_cnt);
                            }

                            fd = cucim::filesystem::open(output_file.c_str(), test_r_flags[flag_idx]);
                            cudaMemset(aligned_device_managed, 0, test_count + test_buf_offset);
                            ssize_t read_cnt =
                                fd->pread(aligned_device_managed, test_count, test_file_offset, test_buf_offset);
                            if (flag_idx == 0)
                            {
                                reference_read_cnt = read_cnt;
                            }
                            else
                            {
                                REQUIRE(read_cnt == reference_read_cnt);
                            }

                            cudaMemcpy(aligned_host, aligned_device_managed, test_count, cudaMemcpyDeviceToHost);
                            size_t posix_hash =
                                str_hash(std::string_view((char*)aligned_host + test_buf_offset, read_cnt));

                            char file_data[BLOCK_SECTOR_SIZE * 4]{};
                            int fd2 = open(output_file.c_str(), O_RDONLY);
                            read_cnt = read(fd2, file_data, BLOCK_SECTOR_SIZE * 4);

                            size_t file_hash = str_hash(std::string_view(file_data, read_cnt));
                            if (flag_idx == 0)
                            {
                                reference_hash = posix_hash;
                                reference_whole_hash = file_hash;
                            }
                            else
                            {
                                REQUIRE(reference_hash == posix_hash);
                                REQUIRE(reference_whole_hash == file_hash);
                            }
                        }
                    }
                }
            }
        }

        CUDA_ERROR(cudaFree(unaligned_device));
        CUDA_ERROR(cudaFreeHost(unaligned_device_host));
        CUDA_ERROR(cudaFree(unaligned_device_managed));
        free(aligned_host);
        free(unaligned_host);
    }
}
