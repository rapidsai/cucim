/*
 * Apache License, Version 2.0
 * Copyright 2020 NVIDIA Corporation
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
#ifndef CUCIM_CONFIG_H
#define CUCIM_CONFIG_H

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

#endif // CUCIM_CONFIG_H
