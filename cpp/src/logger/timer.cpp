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
