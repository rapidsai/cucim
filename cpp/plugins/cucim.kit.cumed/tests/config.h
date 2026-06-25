/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUMED_TESTS_CONFIG_H
#define CUMED_TESTS_CONFIG_H

#include <string>
#include <cstdlib>

struct AppConfig
{
    std::string test_folder;
    std::string test_file;
    std::string temp_folder = "/tmp";
    std::string get_input_path(const char* default_value = "private/generic_tiff_000.tif") const
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

extern AppConfig g_config;

#endif // CUMED_TESTS_CONFIG_H
