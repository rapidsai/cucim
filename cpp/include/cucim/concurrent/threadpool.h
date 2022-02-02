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

#ifndef CUCIM_CONCURRENT_THREADPOOL_H
#define CUCIM_CONCURRENT_THREADPOOL_H

#include "cucim/macros/api_header.h"

#include <functional>
#include <future>
#include <memory>

namespace cucim::concurrent
{

class EXPORT_VISIBLE ThreadPool
{
public:
    explicit ThreadPool(int32_t num_workers);
    ThreadPool(const ThreadPool&) = delete;

    ThreadPool& operator=(const ThreadPool&) = delete;

    operator bool() const;

    ~ThreadPool();

    std::future<void> enqueue(std::function<void()> task);
    void wait();

private:
    struct Executor;
    std::unique_ptr<Executor> executor_;
    size_t num_workers_;
};

} // namespace cucim::concurrent

#endif // CUCIM_CONCURRENT_THREADPOOL_H
