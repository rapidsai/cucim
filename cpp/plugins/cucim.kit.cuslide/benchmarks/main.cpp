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

#include <benchmark/benchmark.h>
#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <openslide/openslide.h>

#include "cucim/core/framework.h"
#include "cucim/io/format/image_format.h"
#include "cucim/memory/memory_manager.h"

#define XSTR(x) STR(x)
#define STR(x) #x

//#include <chrono>

CUCIM_FRAMEWORK_GLOBALS("cuslide.app")

static AppConfig g_config;


static void test_basic(benchmark::State& state)
{
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
                int fd = open(g_config.input_file.c_str(), O_RDONLY);
                fdatasync(fd);
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                close(fd);
            }
        }
        state.ResumeTiming();

        //        auto start = std::chrono::high_resolution_clock::now();
        cucim::Framework* framework = cucim::acquire_framework("cuslide.app");
        if (!framework)
        {
            fmt::print("framework is not available!\n");
            return;
        }

        cucim::io::format::IImageFormat* image_format =
            framework->acquire_interface_from_library<cucim::io::format::IImageFormat>(
                "cucim.kit.cuslide@" XSTR(CUSLIDE_VERSION) ".so");
        //        std::cout << image_format->formats[0].get_format_name() << std::endl;
        if (image_format == nullptr)
        {
            fmt::print("plugin library is not available!\n");
            return;
        }

        auto handle = image_format->formats[0].image_parser.open(g_config.input_file.c_str());

        cucim::io::format::ImageMetadata metadata{};
        image_format->formats[0].image_parser.parse(&handle, &metadata.desc());

        cucim::io::format::ImageReaderRegionRequestDesc request{};
        int64_t request_location[2] = { 0, 0 };
        if (g_config.random_start_location)
        {
            request_location[0] = rand() % (g_config.image_width - state.range(0));
            request_location[1] = rand() % (g_config.image_height - state.range(0));
        }

        request.location = request_location;
        request.level = 0;
        int64_t request_size[2] = { state.range(0), state.range(0) };
        request.size = request_size;
        request.device = const_cast<char*>("cpu");

        cucim::io::format::ImageDataDesc image_data;

        image_format->formats[0].image_reader.read(
            &handle, &metadata.desc(), &request, &image_data, nullptr /*out_metadata*/);
        cucim_free(image_data.container.data);

        image_format->formats[0].image_parser.close(&handle);

        //        auto end = std::chrono::high_resolution_clock::now();
        //        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        //        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void test_openslide(benchmark::State& state)
{
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
                int fd = open(g_config.input_file.c_str(), O_RDONLY);
                fdatasync(fd);
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                close(fd);
            }
        }
        state.ResumeTiming();

        openslide_t* slide = openslide_open(g_config.input_file.c_str());
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

BENCHMARK(test_basic)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 4096); //->UseManualTime();
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
    openslide_t* slide = openslide_open(g_config.input_file.c_str());
    if (slide == nullptr)
    {
        fmt::print("[Error] Cannot load {}!\n", g_config.input_file);
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
    CLI::App app{ "benchmark: cuSlide" };
    app.add_option("--test_file", g_config.input_file, "An input .tif/.svs file path");
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
        // https://github.com/matepek/vscode-catch2-test-adapter detects google benchmark binaries by the following
        // text:
        printf("benchmark [--benchmark_list_tests={true|false}]\n");
    }
    CLI11_PARSE(app, argc, argv);

    if (!setup_configuration())
    {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
}