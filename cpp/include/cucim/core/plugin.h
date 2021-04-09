/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#ifndef CUCIM_PLUGIN_H
#define CUCIM_PLUGIN_H


#include "cucim/macros/api_header.h"
#include "cucim/core/interface.h"

#include <cstddef>
#include <cstdint>

namespace cucim
{

enum class PluginHotReload : std::uint8_t
{
    kDisabled,
    kEnabled,
};

struct PluginImplDesc
{
    const char* name;
    Version version;
    const char* build;
    const char* author;
    const char* description;
    const char* long_description;
    const char* license;
    const char* url;
    const char* platforms;
    PluginHotReload hot_reload;
};

struct PluginEntry
{
    PluginImplDesc desc;

    struct Interface
    {
        InterfaceDesc desc;
        const void* ptr;
        size_t size;
    };

    Interface* interfaces;
    size_t interface_count;
};

struct PluginDesc
{
    PluginImplDesc desc;
    const char* lib_path;
    const InterfaceDesc* interfaces;
    size_t interface_count;
    const InterfaceDesc* dependencies;
    size_t dependency_count;
};

typedef Version(CUCIM_ABI* OnGetFrameworkVersionFn)();
typedef void(CUCIM_ABI* OnPluginRegisterFn)(struct Framework* framework, PluginEntry* out_entry);
typedef void(CUCIM_ABI* OnGetPluginDepsFn)(InterfaceDesc** interface_desc, size_t* count);


} // namespace cucim
#endif // CUCIM_PLUGIN_H
