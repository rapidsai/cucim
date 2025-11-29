/*
 * Apache License, Version 2.0
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_TESTS_CONFIG_H
#define CUSLIDE_TESTS_CONFIG_H

#include <string>
#include <cstdlib>

struct AppConfig
{
    std::string test_folder;
    std::string test_file;
    std::string temp_folder = "/tmp";
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

extern AppConfig g_config;

#endif // CUSLIDE_TESTS_CONFIG_H
