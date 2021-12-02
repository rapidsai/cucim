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

#include <openslide/openslide.h>
#include "cuslide/tiff/tiff.h"
#include "config.h"

#include <cuda_runtime.h>
#include <catch2/catch.hpp>
#include <fmt/format.h>
#include <cucim/filesystem/cufile_driver.h>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <string_view>
#include <cucim/logger/timer.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/mman.h>

#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))
#define ALIGN_DOWN(x, align_to) ((uint64_t)(x) & ~((uint64_t)(align_to)-1))

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

static void shuffle_offsets(uint32_t count, uint64_t* offsets, uint64_t* bytecounts)
{
    // Fisher-Yates shuffle
    for (uint32_t i = 0; i < count; ++i)
    {
        int j = (std::rand() % (count - i)) + i;
        std::swap(offsets[i], offsets[j]);
        std::swap(bytecounts[i], bytecounts[j]);
    }
}

TEST_CASE("Verify raw tiff read", "[test_read_rawtiff.cpp]")
{
//    cudaError_t cuda_status;
//    int err;
    constexpr int BLOCK_SECTOR_SIZE = 4096;
    constexpr bool SHUFFLE_LIST = true;
    //    constexpr int iter_max = 32;
    //    constexpr int skip_count = 2;
    constexpr int iter_max = 1;
    constexpr int skip_count = 0;

    std::srand(std::time(nullptr));

    auto input_file = g_config.get_input_path();

    struct stat sb;
    auto fd_temp = ::open(input_file.c_str(), O_RDONLY);
    fstat(fd_temp, &sb);
    uint64_t test_file_size = sb.st_size;
    ::close(fd_temp);

    auto tif = std::make_shared<cuslide::tiff::TIFF>(input_file,
                                                     O_RDONLY); // , cuslide::tiff::TIFF::kUseLibTiff
    tif->construct_ifds();
    tif->ifd(0)->write_offsets_(input_file.c_str());


    std::ifstream offsets(fmt::format("{}.offsets", input_file), std::ios::in | std::ios::binary);
    std::ifstream bytecounts(fmt::format("{}.bytecounts", input_file), std::ios::in | std::ios::binary);

    // Read image piece count
    uint32_t image_piece_count_ = 0;
    offsets.read(reinterpret_cast<char*>(&image_piece_count_), sizeof(image_piece_count_));
    bytecounts.read(reinterpret_cast<char*>(&image_piece_count_), sizeof(image_piece_count_));

    uint64_t image_piece_offsets_[image_piece_count_];
    uint64_t image_piece_bytecounts_[image_piece_count_];
    uint64_t min_bytecount = 9999999999;
    uint64_t max_bytecount = 0;
    uint64_t sum_bytecount = 0;

    uint64_t min_offset = 9999999999;
    uint64_t max_offset = 0;
    for (uint32_t i = 0; i < image_piece_count_; i++)
    {
        offsets.read((char*)&image_piece_offsets_[i], sizeof(image_piece_offsets_[i]));
        bytecounts.read((char*)&image_piece_bytecounts_[i], sizeof(image_piece_bytecounts_[i]));

        min_bytecount = std::min(min_bytecount, image_piece_bytecounts_[i]);
        max_bytecount = std::max(max_bytecount, image_piece_bytecounts_[i]);
        sum_bytecount += image_piece_bytecounts_[i];

        min_offset = std::min(min_offset, image_piece_offsets_[i]);
        max_offset = std::max(max_offset, image_piece_offsets_[i] + image_piece_bytecounts_[i]);
    }
    bytecounts.close();
    offsets.close();

    fmt::print("file_size    : {}\n", test_file_size);
    fmt::print("min_bytecount: {}\n", min_bytecount);
    fmt::print("max_bytecount: {}\n", max_bytecount);
    fmt::print("avg_bytecount: {}\n", static_cast<double>(sum_bytecount) / image_piece_count_);
    fmt::print("min_offset   : {}\n", min_offset);
    fmt::print("max_offset   : {}\n", max_offset);

    // Shuffle offsets
    if (SHUFFLE_LIST)
    {
        shuffle_offsets(image_piece_count_, image_piece_offsets_, image_piece_bytecounts_);
    }

    // Allocate memory
    uint8_t* unaligned_host = static_cast<uint8_t*>(malloc(test_file_size + BLOCK_SECTOR_SIZE * 2));
    uint8_t* buffer_host = static_cast<uint8_t*>(malloc(test_file_size + BLOCK_SECTOR_SIZE * 2));
    uint8_t* aligned_host = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_host, BLOCK_SECTOR_SIZE));

    //    uint8_t* unaligned_device;
    //    CUDA_ERROR(cudaMalloc(&unaligned_device, test_file_size + BLOCK_SECTOR_SIZE));
    //    uint8_t* aligned_device = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_device, BLOCK_SECTOR_SIZE));
    //
    //    uint8_t* unaligned_device_host;
    //    CUDA_ERROR(cudaMallocHost(&unaligned_device_host, test_file_size + BLOCK_SECTOR_SIZE));
    //    uint8_t* aligned_device_host = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_device_host, BLOCK_SECTOR_SIZE));
    //
    //    uint8_t* unaligned_device_managed;
    //    CUDA_ERROR(cudaMallocManaged(&unaligned_device_managed, test_file_size + BLOCK_SECTOR_SIZE));
    //    uint8_t* aligned_device_managed = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_device_managed,
    //    BLOCK_SECTOR_SIZE));

    cucim::filesystem::discard_page_cache(input_file.c_str());

    fmt::print("count:{} \n", image_piece_count_);

    SECTION("Regular POSIX")
    {
        fmt::print("Regular POSIX\n");

        double total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            auto fd = cucim::filesystem::open(input_file.c_str(), "rpn");
            {
                cucim::logger::Timer timer("- read whole : {:.7f}\n", true, false);

                fd->pread(aligned_host, test_file_size, 0);

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }
        }
        fmt::print("- Read whole average: {}\n", total_elapsed_time / (iter_max - skip_count));

        total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            auto fd = cucim::filesystem::open(input_file.c_str(), "rpn");
            {
                cucim::logger::Timer timer("- read tiles : {:.7f}\n", true, false);

                for (uint32_t i = 0; i < image_piece_count_; ++i)
                {
                    fd->pread(aligned_host, image_piece_bytecounts_[i], image_piece_offsets_[i]);
                }

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }
        }
        fmt::print("- Read tiles average: {}\n", total_elapsed_time / (iter_max - skip_count));
    }

    SECTION("O_DIRECT")
    {
        fmt::print("O_DIRECT\n");

        double total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            auto fd = cucim::filesystem::open(input_file.c_str(), "rp");
            {
                cucim::logger::Timer timer("- read whole : {:.7f}\n", true, false);

                fd->pread(aligned_host, test_file_size, 0);

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }
        }
        fmt::print("- Read whole average: {}\n", total_elapsed_time / (iter_max - skip_count));

        total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            auto fd = cucim::filesystem::open(input_file.c_str(), "rp");
            {
                cucim::logger::Timer timer("- read tiles : {:.7f}\n", true, false);

                for (uint32_t i = 0; i < image_piece_count_; ++i)
                {
                    fd->pread(buffer_host, image_piece_bytecounts_[i], image_piece_offsets_[i]);
                }

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }
        }
        fmt::print("- Read tiles average: {}\n", total_elapsed_time / (iter_max - skip_count));
    }

    SECTION("O_DIRECT pre-load")
    {
        fmt::print("O_DIRECT pre-load\n");

        size_t file_start_offset = ALIGN_DOWN(min_offset, BLOCK_SECTOR_SIZE);
        size_t end_boundary_offset = ALIGN_UP(max_offset + max_bytecount, BLOCK_SECTOR_SIZE);
        size_t large_block_size = end_boundary_offset - file_start_offset;

        fmt::print("- size:{}\n", end_boundary_offset - file_start_offset);

        double total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            auto fd = cucim::filesystem::open(input_file.c_str(), "rp");
            {
                cucim::logger::Timer timer("- preload : {:.7f}\n", true, false);

                fd->pread(aligned_host, large_block_size, file_start_offset);

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }
        }
        fmt::print("- Preload average: {}\n", total_elapsed_time / (iter_max - skip_count));

        total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            auto fd = cucim::filesystem::open(input_file.c_str(), "rp");
            {
                cucim::logger::Timer timer("- read tiles : {:.7f}\n", true, false);

                for (uint32_t i = 0; i < image_piece_count_; ++i)
                {
                    memcpy(buffer_host, aligned_host + image_piece_offsets_[i] - file_start_offset,
                           image_piece_bytecounts_[i]);
                }

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }
        }
        fmt::print("- Read tiles average: {}\n", total_elapsed_time / (iter_max - skip_count));
    }

    SECTION("mmap")
    {
        fmt::print("mmap\n");

        double total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            auto fd_mmap = open(input_file.c_str(), O_RDONLY);
            {
                cucim::logger::Timer timer("- open/close : {:.7f}\n", true, false);

                void* mmap_host = mmap((void*)0, test_file_size, PROT_READ, MAP_SHARED, fd_mmap, 0);

                REQUIRE(mmap_host != MAP_FAILED);

                if (mmap_host != MAP_FAILED)
                {
                    REQUIRE(munmap(mmap_host, test_file_size) != -1);
                    close(fd_mmap);
                }

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }
        }
        fmt::print("- mmap/munmap average: {}\n", total_elapsed_time / (iter_max - skip_count));


        total_elapsed_time = 0;
        for (int iter = 0; iter < iter_max; ++iter)
        {
            cucim::filesystem::discard_page_cache(input_file.c_str());
            //            auto fd_mmap = open(input_file, O_RDONLY);
            //            void* mmap_host = mmap((void*)0, test_file_size, PROT_READ, MAP_SHARED, fd_mmap, 0);
            //            REQUIRE(mmap_host != MAP_FAILED);
            auto fd = cucim::filesystem::open(input_file.c_str(), "rm");
            {
                cucim::logger::Timer timer("- read tiles : {:.7f}\n", true, false);

                for (uint32_t i = 0; i < image_piece_count_; ++i)
                {
                    // 3.441 => 3.489
                    fd->pread(buffer_host, image_piece_bytecounts_[i], image_piece_offsets_[i]);
                    //                                        memcpy(buffer_host, static_cast<char*>(mmap_host) +
                    //                                        image_piece_offsets_[i], image_piece_bytecounts_[i]);
                }

                double elapsed_time = timer.stop();
                if (iter >= skip_count)
                {
                    total_elapsed_time += elapsed_time;
                }
                timer.print();
            }

            //            if (mmap_host != MAP_FAILED)
            //            {
            //                REQUIRE(munmap(mmap_host, test_file_size) != -1);
            //            }
            //            close(fd_mmap);
        }
        fmt::print("- Read tiles average: {}\n", total_elapsed_time / (iter_max - skip_count));
    }

    free(unaligned_host);
    free(buffer_host);
}
