/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/concurrent/threadpool.h"

#include <fmt/format.h>
#include <taskflow/taskflow.hpp>

#include "cucim/profiler/nvtx3.h"

namespace cucim::concurrent
{

struct ThreadPool::Executor : public tf::Executor
{
    // inherits  Constructor
    using tf::Executor::Executor;
};


ThreadPool::ThreadPool(int32_t num_workers)
{
    num_workers_ = num_workers;
    if (num_workers > 0)
    {
        executor_ = std::make_unique<Executor>(num_workers);
    }
}

ThreadPool::~ThreadPool()
{
    if (executor_)
    {
        executor_->wait_for_all();
    }
}

ThreadPool::operator bool() const
{
    return (num_workers_ > 0);
}

std::future<void> ThreadPool::enqueue(std::function<void()> task)
{
    return executor_->async([task]() { task(); });
}

void ThreadPool::wait()
{
    if (executor_)
    {
        executor_->wait_for_all();
    }
}

} // namespace cucim::concurrent
