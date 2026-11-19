/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_PROFILER_CUDA_TEST_TRACE_H
#define CUCIM_PROFILER_CUDA_TEST_TRACE_H

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace cucim::profiler
{

inline void trace_cuda_test_phase(const char* component, const char* phase, uint64_t value = 0)
{
    const char* path = std::getenv("CUCIM_CUDA_TEST_TRACE_FILE");
    if (!path)
    {
        return;
    }
    if (auto* file = std::fopen(path, "a"))
    {
        const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
        const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::fprintf(file, "%lld %s %s %llu\n", static_cast<long long>(micros), component, phase,
                     static_cast<unsigned long long>(value));
        std::fclose(file);
    }
}

} // namespace cucim::profiler

#endif // CUCIM_PROFILER_CUDA_TEST_TRACE_H
