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

#include "config.h"

#include <catch2/catch.hpp>
#include <cucim/concurrent/threadpool.h>
#include <fmt/format.h>
#include <taskflow/taskflow.hpp>

// #include <blockingconcurrentqueue.h>
// #include <future>

// class ThreadPool
// {
// public:
//     explicit ThreadPool(size_t numWorkers)
//     {
//         workers.reserve(numWorkers);
//         for (size_t i = 0; i != numWorkers; ++i)
//             workers.push_back(std::thread(&pump, this));
//     }

//     ~ThreadPool()
//     {
//         for (size_t i = 0; i != workers.size(); ++i)
//             tasks.enqueue(Task(true)); // stop task
//         for (auto& worker : workers)
//             worker.join();
//     }

//     template <typename F>
//     std::future<void> run(F const& f)
//     {
//         Task task;
//         task.function = [=] { f(); };
//         std::future<void> future = task.promise.get_future();
//         tasks.enqueue(std::move(task));
//         return future; // std::move(future);
//     }

//     ThreadPool(ThreadPool const&) = delete;
//     ThreadPool& operator=(ThreadPool const&) = delete;

// private:
//     static void pump(ThreadPool* pool)
//     {
//         Task task;
//         moodycamel::ConsumerToken tok(pool->tasks);
//         while (true)
//         {
//             pool->tasks.wait_dequeue(tok, task);
//             if (task.stop)
//                 break;
//             task.function();
//             task.promise.set_value();
//         }
//     }

// private:
//     struct Task
//     {
//         explicit Task(bool stop = false) : stop(stop)
//         {
//         }
//         Task(Task&& x) noexcept : function(std::move(x.function)), promise(std::move(x.promise)), stop(x.stop)
//         {
//         }
//         Task& operator=(Task&& x) noexcept
//         {
//             std::swap(stop, x.stop);
//             std::swap(function, x.function);
//             std::swap(promise, x.promise);
//             return *this;
//         }

//         std::function<void()> function;
//         std::promise<void> promise;
//         bool stop;
//     };

// private:
//     std::vector<std::thread> workers;
//     moodycamel::BlockingConcurrentQueue<Task> tasks;
// };


TEST_CASE("Test concurrentqueue", "[test_concurrentqueue.cpp]")
{
    // cucim::concurrent::ThreadPool pool(16);
    tf::Executor executor(16);
    tf::Taskflow tf;
    // ThreadPool pool(4);

    // 6.5 without std::packaged_task
    // 6.7 with std::packaged_task
    int count = 0;
    for (int i = 0; i < 100000; ++i)
    {

        // auto fu = pool.run(
        //     [=](int j) {
        //         std::this_thread::sleep_for(std::chrono::milliseconds(100));

        //         fmt::print("ID:{} ARG:{} {}\n", std::hash<std::thread::id>{}(std::this_thread::get_id()), i, j);
        //     },
        //     i);

        // auto fu = [i](int j) {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(100));

        //     fmt::print("ID:{} ARG:{} {}\n", std::hash<std::thread::id>{}(std::this_thread::get_id()), i, j);
        // };
        auto fu = [i, &count] {
            ++count;
            // std::this_thread::sleep_for(std::chrono::microseconds(10));
            // fmt::print("ID:{} ARG:{}\n", std::hash<std::thread::id>{}(std::this_thread::get_id()), i);
        };
        tf.emplace(std::move(fu));
        // executor.silent_async(std::move(fu));
        // pool.run(fu);
    }
    executor.run(tf).wait();
    // executor.wait_for_all();
    fmt::print("count:{}\n", count);

    REQUIRE(1 == 1);
}