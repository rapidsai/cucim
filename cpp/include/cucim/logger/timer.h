/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
