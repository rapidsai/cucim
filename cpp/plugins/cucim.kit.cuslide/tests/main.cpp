/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

//#define CATCH_CONFIG_MAIN
//#include <catch2/catch.hpp>

// Implement main explicitly to handle additional parameters.
#define CATCH_CONFIG_RUNNER
#include "config.h"
#include "cucim/core/framework.h"

#include <catch2/catch.hpp>
#include <string>
#include <fmt/format.h>

CUCIM_FRAMEWORK_GLOBALS("sample.app")

// Global config object
AppConfig g_config;

/**
 * Extract `--[option]` or `--[option]=` string from command and set the value to g_config object.
 *
 * @param argc number of arguments used for command
 * @param argv arguments for command
 * @param obj object reference to modify
 * @param argument name of argument(option)
 * @return true if it extracted the value for the option
 */
static bool extract_test_file_option(int* argc, char** argv, std::string& obj, const char* argument)
{
    std::string arg_str = fmt::format("--{}=", argument); // test_file => --test_file=
    std::string arg_str2 = fmt::format("--{}", argument); // test_file => --test_file

    char* value_ptr = nullptr;
    for (int i = 1; argc && i < *argc; ++i)
    {
        if (strncmp(argv[i], arg_str.c_str(), arg_str.size()) == 0)
        {
            value_ptr = &argv[i][arg_str.size()];
            for (int j = i + 1; argc && j < *argc; ++j)
            {
                argv[j - 1] = argv[j];
            }
            --(*argc);
            argv[*argc] = nullptr;
            break;
        }
        if (strncmp(argv[i], arg_str2.c_str(), arg_str2.size()) == 0 && i + 1 < *argc)
        {
            value_ptr = argv[i + 1];
            for (int j = i + 2; argc && j < *argc; ++j)
            {
                argv[j - 2] = argv[j];
            }
            *argc -= 2;
            argv[*argc] = nullptr;
            argv[*argc + 1] = nullptr;
            break;
        }
    }

    if (value_ptr) {
        obj = value_ptr;
        return true;
    }
    else {
        return false;
    }
}

int main (int argc, char** argv) {
    extract_test_file_option(&argc, argv, g_config.test_folder, "test_folder");
    extract_test_file_option(&argc, argv, g_config.test_file, "test_file");
    extract_test_file_option(&argc, argv, g_config.temp_folder, "temp_folder");
    printf("Target test folder: %s (use --test_folder option to change this)\n", g_config.test_folder.c_str());
    printf("Target test file  : %s (use --test_file option to change this)\n", g_config.test_file.c_str());
    printf("Temp folder       : %s (use --temp_folder option to change this)\n", g_config.temp_folder.c_str());
    int result = Catch::Session().run(argc, argv);
    return result;
}
