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

#ifndef CUCIM_CONFIG_CONFIG_H
#define CUCIM_CONFIG_CONFIG_H

#include "cucim/macros/api_header.h"
#include "cucim/cache/cache_type.h"
#include "cucim/cache/image_cache_config.h"

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

    std::string shm_name() const;
    pid_t pid() const;
    pid_t ppid() const;
    pid_t pgid() const;

private:
    std::string get_config_path() const;
    bool parse_config(std::string& path);
    void set_default_configuration();

    std::string source_path_;

    cucim::cache::ImageCacheConfig cache_;
};

} // namespace cucim::config

#endif // CUCIM_CONFIG_CONFIG_H
