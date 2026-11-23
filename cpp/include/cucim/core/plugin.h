/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


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
