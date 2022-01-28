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

#include "config.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>

#include <benchmark/benchmark.h>
#include <openslide/openslide.h>
#include <CLI/CLI.hpp>

#include "cucim/cuimage.h"

static AppConfig g_config;

static void test_cucim(benchmark::State& state)
{
    std::string input_path = g_config.get_input_path();

    int arg = -1;
    for (auto state_item : state)
    {
        state.PauseTiming();
        {
            // Use a different start random seed for the different argument
            if (arg != state.range())
            {
                arg = state.range();
                srand(g_config.random_seed + arg);
            }

            if (g_config.discard_cache)
            {
                int fd = open(input_path.c_str(), O_RDONLY);
                fdatasync(fd);
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                close(fd);
            }
        }
        state.ResumeTiming();

        int64_t request_location[2] = { 0, 0 };
        if (g_config.random_start_location)
        {
            request_location[0] = rand() % (g_config.image_width - state.range(0));
            request_location[1] = rand() % (g_config.image_height - state.range(0));
        }

        cucim::CuImage image = cucim::CuImage(input_path.c_str());
        cucim::CuImage region =
            image.read_region({ request_location[0], request_location[1] }, { state.range(0), state.range(0) }, 0);
    }
}

static void test_openslide(benchmark::State& state)
{
    std::string input_path = g_config.get_input_path();

    int arg = -1;
    for (auto _ : state)
    {
        state.PauseTiming();
        {
            // Use a different start random seed for the different argument
            if (arg != state.range())
            {
                arg = state.range();
                srand(g_config.random_seed + arg);
            }

            if (g_config.discard_cache)
            {
                int fd = open(input_path.c_str(), O_RDONLY);
                fdatasync(fd);
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                close(fd);
            }
        }
        state.ResumeTiming();

        openslide_t* slide = openslide_open(input_path.c_str());
        uint32_t* buf = static_cast<uint32_t*>(cucim_malloc(state.range(0) * state.range(0) * 4));
        int64_t request_location[2] = { 0, 0 };
        if (g_config.random_start_location)
        {
            request_location[0] = rand() % (g_config.image_width - state.range(0));
            request_location[1] = rand() % (g_config.image_height - state.range(0));
        }

        openslide_read_region(slide, buf, request_location[0], request_location[1], 0, state.range(0), state.range(0));
        cucim_free(buf);
        openslide_close(slide);
    }
}


BENCHMARK(test_cucim)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 4096);
BENCHMARK(test_openslide)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 4096);

static bool remove_help_option(int* argc, char** argv)
{
    for (int i = 1; argc && i < *argc; ++i)
    {
        if (strncmp(argv[i], "-h", 3) == 0 || strncmp(argv[i], "--help", 7) == 0)
        {
            for (int j = i + 1; argc && j < *argc; ++j)
            {
                argv[j - 1] = argv[j];
            }
            --(*argc);
            argv[*argc] = nullptr;
            return true;
        }
    }
    return false;
}

static bool setup_configuration()
{
    std::string input_path = g_config.get_input_path();

    openslide_t* slide = openslide_open(input_path.c_str());
    if (slide == nullptr)
    {
        fmt::print("[Error] Cannot load {}!\n", input_path);
        return false;
    }
    int64_t w, h;
    openslide_get_level0_dimensions(slide, &w, &h);

    g_config.image_width = w;
    g_config.image_height = h;

    openslide_close(slide);
    return true;
}

// BENCHMARK_MAIN();
int main(int argc, char** argv)
{

    // Skip processing help option
    bool has_help_option = remove_help_option(&argc, argv);

    ::benchmark::Initialize(&argc, argv);
    //    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    //        return 1;

    CLI::App app{ "cuCIM Benchmark" };
    app.add_option("--test_folder", g_config.test_folder, "An input test folder path");
    app.add_option("--test_file", g_config.test_file, "An input test image file path");
    app.add_option("--discard_cache", g_config.discard_cache, "Discard page cache for the input file for each iteration");
    app.add_option("--random_seed", g_config.random_seed, "A random seed number");
    app.add_option(
        "--random_start_location", g_config.random_start_location, "Randomize start location of read_region()");

    // Pseudo benchmark options
    app.add_option("--benchmark_list_tests", g_config.benchmark_list_tests, "{true|false}");
    app.add_option("--benchmark_filter", g_config.benchmark_filter, "<regex>");
    app.add_option("--benchmark_min_time", g_config.benchmark_min_time, "<min_time>");
    app.add_option("--benchmark_repetitions", g_config.benchmark_repetitions, "<num_repetitions>");
    app.add_option("--benchmark_report_aggregates_only", g_config.benchmark_report_aggregates_only, "{true|false}");
    app.add_option("--benchmark_display_aggregates_only", g_config.benchmark_display_aggregates_only, "{true|false}");
    app.add_option("--benchmark_format", g_config.benchmark_format, "<console|json|csv>");
    app.add_option("--benchmark_out", g_config.benchmark_out, "<filename>");
    app.add_option("--benchmark_out_format", g_config.benchmark_out_format, "<json|console|csv>");
    app.add_option("--benchmark_color", g_config.benchmark_color, "{auto|true|false}");
    app.add_option("--benchmark_counters_tabular", g_config.benchmark_counters_tabular, "{true|false}");
    app.add_option("--v", g_config.v, "<verbosity>");

    // Append help option if exists
    if (has_help_option)
    {
        argv[argc] = const_cast<char*>("--help");
        ++argc;
    }
    CLI11_PARSE(app, argc, argv);

    if (!setup_configuration())
    {
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();
}
