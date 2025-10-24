/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
