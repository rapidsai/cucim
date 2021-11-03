/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#ifndef CUCIM_FRAMEWORK_H
#define CUCIM_FRAMEWORK_H

#include "cucim/core/plugin.h"

#include <cstdlib>

#include "cucim/memory/memory_manager.h"

#define CUCIM_FRAMEWORK_GLOBALS(client_name)                                                                           \
    CUCIM_NO_EXPORT const char* g_cucim_client_name = client_name;                                                     \
    CUCIM_NO_EXPORT cucim::Framework* g_cucim_framework = nullptr;


namespace cucim
{
#define TEMP_STR(x) #x
#define TEMP_XSTR(x) TEMP_STR(x)
const struct Version kFrameworkVersion = { static_cast<uint32_t>(std::atoi(TEMP_XSTR(CUCIM_VERSION_MAJOR))),
                                           static_cast<uint32_t>(std::atoi(TEMP_XSTR(CUCIM_VERSION_MINOR))),
                                           static_cast<uint32_t>(std::atoi(TEMP_XSTR(CUCIM_VERSION_PATCH))) };
#undef TEMP_STR
#undef TEMP_XSTR

struct PluginRegistrationDesc
{
    OnPluginRegisterFn on_register; ///< Required
    //    OnPluginStartupFn on_startup_fn; ///! Can be nullptr
    //    OnPluginShutdownFn on_shutdown_fn; ///! Can be nullptr
    OnGetPluginDepsFn on_get_deps; ///! Can be nullptr
    //    OnReloadDependencyFn on_reload_dependency_fn; ///! Can be nullptr
    //    OnPluginPreStartupFn on_pre_startup_fn; ///! Can be nullptr
    //    OnPluginPostShutdownFn on_post_shutdown_fn;  ///! Can be nullptr
};

struct PluginLoadingDesc
{
    const char* plugin_path;

    static PluginLoadingDesc get_default()
    {
        static constexpr const char* default_plugin_path = "cucim@0.0.1.so";
        return { default_plugin_path };
    }
};


struct Framework
{
    // TODO: need to update for better plugin support - https://github.com/rapidsai/cucim/issues/134
    // void load_plugins(const PluginLoadingDesc& desc = PluginLoadingDesc::get_default());
    bool(CUCIM_ABI* register_plugin)(const char* client_name, const PluginRegistrationDesc& desc);
    void*(CUCIM_ABI* acquire_interface_from_library_with_client)(const char* client_name,
                                                                 InterfaceDesc desc,
                                                                 const char* library_path);
    void(CUCIM_ABI* unload_all_plugins)();

    template <typename T>
    T* acquire_interface_from_library(const char* library_path);

    // cuCIM-specific methods
    void(CUCIM_ABI* load_plugin)(const char* library_path);
    const char*(CUCIM_ABI* get_plugin_root)();
    void(CUCIM_ABI* set_plugin_root)(const char* path);
};

CUCIM_API cucim::Framework* acquire_framework(const char* app_name, Version framework_version = kFrameworkVersion);

CUCIM_API void release_framework();

} // namespace cucim

extern const char* g_cucim_client_name;
extern cucim::Framework* g_cucim_framework;

namespace cucim
{

inline Framework* get_framework()
{
    return g_cucim_framework;
}


template <typename T>
T* Framework::acquire_interface_from_library(const char* library_path)
{
    const auto desc = T::get_interface_desc();
    return static_cast<T*>(this->acquire_interface_from_library_with_client(g_cucim_client_name, desc, library_path));
}
} // namespace cucim


#endif // CUCIM_FRAMEWORK_H
