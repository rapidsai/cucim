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

#ifndef CUCIM_PLUGIN_PLUGIN_CONFIG_H
#define CUCIM_PLUGIN_PLUGIN_CONFIG_H

#include "cucim/core/framework.h"

#include <string>
#include <vector>

namespace cucim::plugin
{

#define XSTR(x) STR(x)
#define STR(x) #x

struct EXPORT_VISIBLE PluginConfig
{
    void load_config(const void* json_obj);

    std::vector<std::string> plugin_names{ std::string("cucim.kit.cuslide@" XSTR(CUCIM_VERSION) ".so"),
                                           std::string("cucim.kit.cumed@" XSTR(CUCIM_VERSION) ".so") };
};

#undef STR
#undef XSTR

} // namespace cucim::plugin

#endif // CUCIM_PLUGIN_PLUGIN_CONFIG_H
