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

#ifndef CUCIM_CUCIM_FRAMEWORK_H
#define CUCIM_CUCIM_FRAMEWORK_H

#include "cucim/core/framework.h"
#include "plugin_manager.h"
#include "cucim_plugin.h"

#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>

namespace cucim
{

class CuCIMFramework
{
public:
    CuCIMFramework();
    ~CuCIMFramework();

    bool register_plugin(const char* client_name, const PluginRegistrationDesc& desc);
    bool register_plugin(const std::string& file_path, bool reloadable = false, bool unload = false);
    bool register_plugin(const std::shared_ptr<Plugin>& plugin);
    bool unregister_plugin(const char* name);
    void unregister_plugin(Plugin* plugin);
    bool try_terminate_plugin(Plugin* plugin, std::vector<Plugin*>* plugins_to_load);
    void load_plugins(const PluginLoadingDesc& desc);
    void unload_all_plugins();


    bool resolve_plugin_dependencies(Plugin* plugin);
    bool resolve_interface_dependency(const Plugin::InterfaceData& info, bool log_errors);
    bool resolve_interface_dependency_with_logging(const Plugin::InterfaceData& desc);
    bool resolve_interface_dependency_no_logging(const Plugin::InterfaceData& desc);
    Plugin::Interface get_default_plugin(const InterfaceDesc& desc, bool optional);
    Plugin::Interface get_specific_plugin(const InterfaceDesc& desc, const char* plugin_name, bool optional);


    void* acquire_interface(const char* client, const InterfaceDesc& desc, const char* plugin_name, bool optional = false);
    void* acquire_interface_from_library(const char* client,
                                         const InterfaceDesc& desc,
                                         const char* library_path,
                                         bool optional = false);
    size_t get_plugin_count() const;
    void get_plugins(PluginDesc* out_plugins) const;
    size_t get_plugin_index(const char* name) const;
    Plugin* get_plugin(size_t index) const;
    Plugin* get_plugin(const char* name) const;
    Plugin* get_plugin_by_library_path(const std::string& library_path);

    // cuCIM-specific methods;
    void load_plugin(const char* library_path);
    std::string& get_plugin_root();
    void set_plugin_root(const char* path);

private:
    struct CandidatesEntry
    {
        std::vector<Plugin::Interface> candidates;
        Plugin::Interface selected = {};
        std::string specifiedDefaultPlugin;
    };

    using Mutex = std::recursive_mutex;
    using ScopedLock = std::unique_lock<Mutex>;
    mutable Mutex mutex_;

    std::vector<size_t> plugin_load_order_;
    PluginManager plugin_manager_;
    std::unordered_map<std::string, size_t> library_path_to_plugin_index_;
    std::unordered_map<std::string, size_t> name_to_plugin_index_;
    std::unordered_map<std::string, CandidatesEntry> interface_candidates_;
    std::unordered_map<const void*, Plugin::Interface> ptr_to_interface_;

    // cuCIM-specific fields;
    std::string plugin_root_path_;
};
} // namespace cucim

#endif // CUCIM_CUCIM_FRAMEWORK_H
