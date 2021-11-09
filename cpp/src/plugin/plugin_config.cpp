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

#include "cucim/plugin/plugin_config.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace cucim::plugin
{

void PluginConfig::load_config(const void* json_obj)
{
    const json& plugin_config = *(static_cast<const json*>(json_obj));

    if (plugin_config.contains("names") && plugin_config["names"].is_array())
    {
        std::vector<std::string> names;
        names.reserve(16);
        for (const auto& name : plugin_config["names"])
        {
            names.push_back(name);
        }
        plugin_names = std::move(names);
    }
}

} // namespace cucim::plugin
