/*
 * Apache License, Version 2.0
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_CONFIG_H
#define CUSLIDE_CONFIG_H

#include <string>

struct AppConfig
{
    std::string test_folder;
    std::string test_file;
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

    std::string get_input_path(const std::string default_value = "generated/tiff_stripe_4096x4096_256.tif") const
    {
        // If `test_file` is absolute path
        if (!test_folder.empty() && test_file.substr(0, 1) == "/")
        {
            return test_file;
        }
        else
        {
            std::string test_data_folder = test_folder;
            if (test_data_folder.empty())
            {
                if (const char* env_p = std::getenv("CUCIM_TESTDATA_FOLDER"))
                {
                    test_data_folder = env_p;
                }
                else
                {
                    test_data_folder = "test_data";
                }
            }
            if (test_file.empty())
            {
                return test_data_folder + "/" + default_value;
            }
            else
            {
                return test_data_folder + "/" + test_file;
            }
        }
    }
};

#endif // CUSLIDE_CONFIG_H
