/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/logger/timer.h"

#include <fmt/format.h>

namespace cucim::logger
{

Timer::Timer(const char* message, bool auto_start, bool auto_output)
{
    message_ = message;
    is_auto_output_ = auto_output;
    if (auto_start)
    {
        elapsed_seconds_ = 0.0;
        start_ = std::chrono::high_resolution_clock::now();
    }
}

void Timer::start()
{
    elapsed_seconds_ = 0.0;
    start_ = std::chrono::high_resolution_clock::now();
}

double Timer::stop()
{
    end_ = std::chrono::high_resolution_clock::now();
    elapsed_seconds_ = std::chrono::duration_cast<std::chrono::duration<double>>(end_ - start_).count();
    return elapsed_seconds_;
}

double Timer::elapsed_time()
{
    return elapsed_seconds_;
}

void Timer::print(const char* message)
{
    if (message)
    {
        fmt::print(stderr, message, elapsed_seconds_);
    }
    else
    {
        fmt::print(stderr, message_, elapsed_seconds_);
    }
}

Timer::~Timer()
{
    if (elapsed_seconds_ <= 0.0)
    {
        end_ = std::chrono::high_resolution_clock::now();
        elapsed_seconds_ = std::chrono::duration_cast<std::chrono::duration<double>>(end_ - start_).count();
    }
    if (is_auto_output_)
    {
        print();
    }
}

} // namespace cucim::logger
