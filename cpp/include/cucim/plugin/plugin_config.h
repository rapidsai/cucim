/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
