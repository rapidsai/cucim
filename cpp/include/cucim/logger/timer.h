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
#ifndef CUCIM_TIMER_H
#define CUCIM_TIMER_H

#include "cucim/macros/defines.h"

#include <chrono>

namespace cucim::logger
{
class EXPORT_VISIBLE Timer
{
public:
    Timer(const char* message, bool auto_start = true, bool auto_output = true);
    void start();
    double stop();
    double elapsed_time();
    void print(const char* message = nullptr);
    ~Timer();

private:
    const char* message_ = nullptr;
    bool is_auto_output_ = false;
    double elapsed_seconds_ = -1;
    std::chrono::time_point<std::chrono::system_clock> start_{};
    std::chrono::time_point<std::chrono::system_clock> end_{};
};

} // namespace cucim::logger


#endif // CUCIM_TIMER_H
