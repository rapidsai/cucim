/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_CONFIG_CONFIG_H
#define CUCIM_CONFIG_CONFIG_H

#include "cucim/macros/api_header.h"
#include "cucim/cache/cache_type.h"
#include "cucim/cache/image_cache_config.h"
#include "cucim/plugin/plugin_config.h"
#include "cucim/profiler/profiler_config.h"

#include <string>
#include <string_view>

#include <sys/types.h>
#include <unistd.h>

namespace cucim::config
{

constexpr const char* kDefaultConfigFileName = ".cucim.json";

class EXPORT_VISIBLE Config
{
public:
    Config();

    cucim::cache::ImageCacheConfig& cache();
    cucim::plugin::PluginConfig& plugin();
    cucim::profiler::ProfilerConfig& profiler();

    std::string shm_name() const;
    pid_t pid() const;
    pid_t ppid() const;
    pid_t pgid() const;

private:
    std::string get_config_path() const;
    bool parse_config(std::string& path);
    void set_default_configuration();
    void override_from_envs();
    void init_configs();

    std::string source_path_;

    cucim::cache::ImageCacheConfig cache_;
    cucim::plugin::PluginConfig plugin_;
    cucim::profiler::ProfilerConfig profiler_;
};

} // namespace cucim::config

#endif // CUCIM_CONFIG_CONFIG_H
