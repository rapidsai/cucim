/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

    // Override config with environment variables
    override_from_envs();
    init_configs();
}

cucim::cache::ImageCacheConfig& Config::cache()
{
    return cache_;
}

cucim::plugin::PluginConfig& Config::plugin()
{
    return plugin_;
}

cucim::profiler::ProfilerConfig& Config::profiler()
{
    return profiler_;
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

        json plugin = obj["plugin"];
        if (plugin.is_object())
        {
            plugin_.load_config(&plugin);
        }

        json profiler = obj["profiler"];
        if (profiler.is_object())
        {
            profiler_.load_config(&profiler);
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
void Config::override_from_envs()
{
    if (const char* env_p = std::getenv("CUCIM_TRACE"))
    {
        if (env_p)
        {
            if (env_p[0] == '1')
            {
                profiler_.trace = true;
            }
            else
            {
                profiler_.trace = false;
            }
        }
    }
}
void Config::init_configs()
{
    // Initialization if needed.
}

} // namespace cucim::config
