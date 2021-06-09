/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "cucim/memory/memory_manager.h"

#include <vector>
#include <string>
#include <cstring>
#include <memory_resource>
#include <benchmark/benchmark.h>


static void vector_copy_push_back(benchmark::State& state)
{
    const int data_count = 50000;
    uint64_t data[data_count];

    // Code inside this loop is measured repeatedly
    for (auto _ : state)
    {
        std::vector<uint64_t> data_vec;
        for (int i = 0; i < data_count; ++i)
        {
            data_vec.push_back(data[i]);
        }
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(data_vec);
    }
}
// Register the function as a benchmark
BENCHMARK(vector_copy_push_back);

static void vector_copy_insert(benchmark::State& state)
{
    const int data_count = 50000;
    uint64_t data[data_count];

    // Code before the loop is not measured
    for (auto _ : state)
    {
        std::vector<uint64_t> data_vec;
        data_vec.insert(data_vec.end(), &data[0], &data[data_count]);
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(data_vec);
    }
}
BENCHMARK(vector_copy_insert);

static void vector_copy_vector_vector(benchmark::State& state)
{
    const int data_count = 50000;
    uint64_t data[data_count];

    // Code before the loop is not measured
    for (auto _ : state)
    {
        std::vector<uint64_t> data_vec;
        data_vec.insert(data_vec.end(), &data[0], &data[data_count]);
        std::vector<uint64_t> data_vec2(&data[0], &data[data_count]);

        data_vec.insert(data_vec.end(), data_vec2.begin(), data_vec2.end());
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(data_vec);
    }
}
BENCHMARK(vector_copy_vector_vector);

static void string_memcpy(benchmark::State& state)
{
    // Code before the loop is not measured
    for (auto _ : state)
    {
        std::string data = "#########################################################################################################################################################################################";
        const int size = data.size();

        char * c_str = (char*) malloc(size + 1);
        memcpy(c_str, data.data(), size);
        c_str[size] = '\0';
        free(c_str);
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(c_str);
        benchmark::DoNotOptimize(size);
    }
}
BENCHMARK(string_memcpy);

static void string_strcpy(benchmark::State& state)
{
    // Code before the loop is not measured
    for (auto _ : state)
    {
        std::string data = "#########################################################################################################################################################################################";
        char * c_str = (char*) malloc(data.size() + 1);
        strcpy(c_str, data.data());
        free(c_str);
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(c_str);
    }
}
BENCHMARK(string_strcpy);

static void string_strdup(benchmark::State& state)
{

    // Code before the loop is not measured
    for (auto _ : state)
    {
        std::string data = "#########################################################################################################################################################################################";
        char * c_str = strdup(data.data());
        free(c_str);
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(c_str);
    }
}
BENCHMARK(string_strdup);


static void alloc_malloc(benchmark::State& state)
{

    // Code before the loop is not measured
    for (auto _ : state)
    {
        char* arr[30000];
        for (int i = 0; i < 30000; i++)
        {
            arr[i] = (char*)malloc(10);
            arr[i][0] = i;
        }
        for (int i = 0; i < 30000; i++)
        {
            free(arr[i]);
        }
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(alloc_malloc);//->Iterations(100);


static void alloc_pmr(benchmark::State& state)
{

    // Code before the loop is not measured
    for (auto _ : state)
    {
        char* arr[30000];
        for (int i = 0; i < 30000; i++)
        {
            arr[i] = static_cast<char*>(cucim_malloc(10));
            arr[i][0] = i;
        }
        for (int i = 0; i < 30000; i++)
        {
            cucim_free(arr[i]);
        }
        // Make sure the variable is not optimized away by compiler
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(alloc_pmr);//->Iterations(100);

BENCHMARK_MAIN();

// Debug

// ```
// --------------------------------------------------------------------
// Benchmark                          Time             CPU   Iterations
// --------------------------------------------------------------------
// vector_copy_push_back         591517 ns       591510 ns         1267
// vector_copy_insert              8488 ns         8488 ns        85160
// vector_copy_vector_vector     225441 ns       225439 ns         3069
// string_memcpy                    169 ns          169 ns      3854598
// string_strcpy                    202 ns          202 ns      4114834
// string_strdup                    184 ns          184 ns      3666944
// ```

// Release

// ```
// --------------------------------------------------------------------
// Benchmark                          Time             CPU   Iterations
// --------------------------------------------------------------------
// vector_copy_push_back         118518 ns       118518 ns         5745
// vector_copy_insert              7779 ns         7779 ns        92190
// vector_copy_vector_vector     198800 ns       198793 ns         3347
// string_memcpy                   20.3 ns         20.3 ns     32102053
// string_strcpy                   24.8 ns         24.8 ns     27352024
// string_strdup                   32.4 ns         32.4 ns     21458177
// ```