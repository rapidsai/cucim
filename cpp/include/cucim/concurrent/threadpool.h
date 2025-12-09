/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
