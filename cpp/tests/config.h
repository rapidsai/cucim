/*
 * Apache License, Version 2.0
 * Copyright 2021 NVIDIA Corporation
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
#ifndef CUCIM_TESTS_CONFIG_H
#define CUCIM_TESTS_CONFIG_H

#include <string>
#include <cstdlib>

#define XSTR(x) STR(x)
#define STR(x) #x

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
    std::string get_plugin_path(const char* default_value = "cucim.kit.cuslide@" XSTR(CUCIM_VERSION) ".so")
    {
        std::string plugin_path = default_value;
        if (const char* env_p = std::getenv("CUCIM_TEST_PLUGIN_PATH"))
        {
            plugin_path = env_p;
        }
        return plugin_path;
    }
};

extern AppConfig g_config;

#endif // CUCIM_TESTS_CONFIG_H
