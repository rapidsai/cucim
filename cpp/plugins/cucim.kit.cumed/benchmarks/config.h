/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUMED_CONFIG_H
#define CUMED_CONFIG_H

#include <string>

struct AppConfig
{
    std::string input_file = "test_data/private/generic_tiff_000.tif";
    bool discard_cache = false;
    int random_seed = 0;
    bool random_start_location = false;

    int64_t image_width = 0;
    int64_t image_height = 0;

    // Pseudo configurations for google benchmark
    bool benchmark_list_tests = false;
    std::string benchmark_filter; // <regex>
    int benchmark_min_time = 0; // <min_time>
    int benchmark_repetitions = 0; // <num_repetitions>
    bool benchmark_report_aggregates_only = false;
    bool benchmark_display_aggregates_only = false;
    std::string benchmark_format; // <console|json|csv>
    std::string benchmark_out; // <filename>
    std::string benchmark_out_format; // <json|console|csv>
    std::string benchmark_color; //  {auto|true|false}
    std::string benchmark_counters_tabular;
    std::string v; // <verbosity>
};

#endif // CUMED_CONFIG_H
