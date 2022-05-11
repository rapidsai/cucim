/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
    auto future = executor_->async([task]() { task(); });
    return std::move(future);
}

void ThreadPool::wait()
{
    if (executor_)
    {
        executor_->wait_for_all();
    }
}

} // namespace cucim::concurrent
