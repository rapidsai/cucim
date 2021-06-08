/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "cucim/config/config.h"

#include "cucim/cache/cache_type.h"
#include "cucim/util/file.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>

using json = nlohmann::json;

namespace cucim::config
{

Config::Config()
{
    std::string config_path = get_config_path();

    bool is_configured_from_file = false;
    if (!config_path.empty())
    {
        is_configured_from_file = parse_config(config_path);
    }
    if (is_configured_from_file)
    {
        source_path_ = config_path;
    }
    else
    {
        set_default_configuration();
    }
}

cucim::cache::ImageCacheConfig& Config::cache()
{
    return cache_;
}

std::string Config::shm_name() const
{
    return fmt::format("cucim-shm.{}", pgid());
}


pid_t Config::pid() const
{
    return getpid();
}
pid_t Config::ppid() const
{
    return getppid();
}
pid_t Config::pgid() const
{
    return getpgid(getpid());
}


std::string Config::get_config_path() const
{
    // Read config file from:
    //   1. A path specified by 'CUCIM_CONFIG_PATH'
    //   2. (current folder)/.cucim.json
    //   3. $HOME/.cucim.json
    std::string config_path;

    if (const char* env_p = std::getenv("CUCIM_CONFIG_PATH"))
    {
        if (cucim::util::file_exists(env_p))
        {
            config_path = env_p;
        }
    }
    if (config_path.empty() && cucim::util::file_exists(kDefaultConfigFileName))
    {
        config_path = kDefaultConfigFileName;
    }
    if (config_path.empty())
    {
        if (const char* env_p = std::getenv("HOME"))
        {
            auto home_path = (std::filesystem::path(env_p) / kDefaultConfigFileName).string();
            if (cucim::util::file_exists(home_path.c_str()))
            {
                config_path = home_path;
            }
        }
    }
    return config_path;
}
bool Config::parse_config(std::string& path)
{
    try
    {
        std::ifstream ifs(path);
        json obj = json::parse(ifs, nullptr /*cb*/, true /*allow_exceptions*/, true /*ignore_comments*/);
        json cache = obj["cache"];
        if (cache.is_object())
        {
            cache_.load_config(&cache);
        }
    }
    catch (const json::parse_error& e)
    {
        fmt::print(stderr,
                   "Failed to load configuration file: {}\n"
                   "  message: {}\n"
                   "  exception id: {}\n"
                   "  byte position of error: {}\n",
                   path, e.what(), e.id, e.byte);
        return false;
    }
    return true;
}
void Config::set_default_configuration()
{
    // Override if the initializer of Config class is not enough.
}

} // namespace cucim::config