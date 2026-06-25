/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

#include <benchmark/benchmark.h>
#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "cucim/core/framework.h"
#include "cucim/io/format/image_format.h"
#include "cucim/memory/memory_manager.h"

#define XSTR(x) STR(x)
#define STR(x) #x

//#include <chrono>

CUCIM_FRAMEWORK_GLOBALS("cumed.app")

static AppConfig g_config;


static void test_basic(benchmark::State& state)
{
    for (auto _ : state)
    {
        // Implement
        // auto item = state.range(0);
    }
}


BENCHMARK(test_basic)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 4096);

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

    return true;
}

// BENCHMARK_MAIN();
int main(int argc, char** argv)
{
    // Skip processing help option
    bool has_help_option = remove_help_option(&argc, argv);

    ::benchmark::Initialize(&argc, argv);
    CLI::App app{ "benchmark: cuMed" };
    app.add_option("--test_file", g_config.input_file, "An input file path");

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
