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

#ifndef CUCIM_CUCIM_PLUGIN_H
#define CUCIM_CUCIM_PLUGIN_H

#include <string>
#include <vector>
#include <mutex>
#include <sstream>
#include "cucim/core/plugin.h"
#include "cucim/core/interface.h"
#include "cucim/dynlib/helper.h"
#include "plugin_manager.h"
#include "version.inl"

namespace cucim
{

class Plugin
{
public:
    enum class ResolveState
    {
        kUnused,
        kInprocess,
        kResolved,
        kFailed
    };

    // Returns whether the initialization has happened. didNow is set true only if the initialize happened during this
    // call This ensures atomicity between isInitialized/initialize()
    enum class InitResult
    {
        kFailedInitialize = 0,
        kAlreadyInitialized,
        kDidInitialize
    };

    struct InterfaceData
    {
        std::string name;
        InterfaceVersion version = { 0, 0 };
        void* ptr = nullptr;
        uint64_t size = 0;

        InterfaceDesc to_interface_desc() const
        {
            return InterfaceDesc{ name.c_str(), { version.major, version.minor } };
        }

        void store(const InterfaceDesc& desc);
    };

    struct ImplementationDesc
    {
        std::string name;
        Version version;
        std::string build;
        std::string author;
        std::string description;
        std::string long_description;
        std::string license;
        std::string url;
        std::string platforms;
        PluginHotReload hot_reload = PluginHotReload::kDisabled;

        PluginImplDesc to_plugin_impl() const
        {
            return PluginImplDesc{ name.c_str(),        version,
                                   build.c_str(),       author.c_str(),
                                   description.c_str(), long_description.c_str(),
                                   license.c_str(),     url.c_str(),
                                   platforms.c_str(), PluginHotReload::kDisabled};
        }

        void store(const PluginImplDesc& desc);
    };

    struct Interface
    {
        Interface() : plugin_index(kInvalidPluginIndex), interface_index(0)
        {
        }

        Interface(size_t plugin_idx, size_t interface_idx) : plugin_index(plugin_idx), interface_index(interface_idx)
        {
        }

        size_t plugin_index;
        size_t interface_index;

        Plugin* get_plugin(const PluginManager& registry) const
        {
            return plugin_index != kInvalidPluginIndex ? registry.get_plugin(plugin_index) : nullptr;
        }

        const Plugin::InterfaceData& get_interface_desc(const PluginManager& registry) const
        {
            return registry.get_plugin(plugin_index)->get_interfaces()[interface_index];
        }

        bool operator==(const Interface& other) const
        {
            return ((plugin_index == other.plugin_index) && (interface_index == other.interface_index));
        }
    };

    Plugin();
    explicit Plugin(const std::string& file_path);
    ~Plugin();

    const char* name_cstr() const
    {
        return name_.c_str();
    }

    std::string name_str() const
    {
        return name_;
    }
    const char* library_path() const
    {
        return library_path_.c_str();
    }

    bool is_initialized() const
    {
        return is_initialized_;
    }
    bool is_in_initialization() const
    {
        return is_in_initialization_;
    }


    bool preload(bool reloadable, bool unload);
    InitResult ensure_initialized();
    bool initialize();
    void terminate();
    void unload();


    size_t index_;
    ResolveState resolve_state_;


    const std::vector<Plugin::InterfaceData>& get_interfaces() const
    {
        return data_[kVersionStateCurrent].interfaces;
    }
    const ImplementationDesc& get_impl_desc() const
    {
        return data_[kVersionStateCurrent].desc;
    }
    const PluginDesc& get_plugin_desc() const
    {
        return plugin_desc_;
    }

private:
    static constexpr uint32_t kVersionStateBackup = 0;
    static constexpr uint32_t kVersionStateCurrent = 1;
    static constexpr uint32_t kVersionStateCount = 2;

    struct VersionedData
    {
        VersionedData() = default;
        int version = 0;
        ImplementationDesc desc;
        uint64_t interface_size = 0;
        std::vector<InterfaceData> interfaces;
        std::vector<InterfaceDesc> plugin_interfaces;
        std::vector<InterfaceData> dependencies;
        std::vector<InterfaceDesc> plugin_dependencies;
    };

    template <typename T>
    bool init_plugin_fn(T& handle, const char* name, bool optional = false) const;
    bool prepare_file_to_load(std::string& out_lib_file_path, int version);
    bool fill_registration_data(int version, bool full, const std::string& lib_file);
    bool check_framework_version();

    bool try_load(int version, bool full);
    bool load(int version = 0, bool full = true);


    VersionedData data_[kVersionStateCount];

    std::string library_path_;
    std::string name_;
    PluginDesc plugin_desc_;
    dynlib::LibraryHandle library_handle_;


    OnGetFrameworkVersionFn on_get_framework_version_;
    OnPluginRegisterFn on_register_;
    OnGetPluginDepsFn on_get_deps_;

    bool is_loaded_;
    bool is_initialized_;
    bool is_in_initialization_;
    bool is_reloadable_;
    int next_version_;

    std::recursive_mutex init_lock_;
};


inline bool operator==(const Plugin::InterfaceData& lhs, const Plugin::InterfaceData& rhs)
{
    return lhs.name == rhs.name && lhs.version == rhs.version;
}

inline std::ostream& operator<<(std::ostream& o, const Plugin::InterfaceData& info)
{
    o << "[" << info.name << " v" << info.version.major << "." << info.version.minor << "]";
    return o;
}

inline std::ostream& operator<<(std::ostream& o, const std::vector<Plugin::InterfaceData>& interfaces)
{
    for (size_t i = 0; i < interfaces.size(); i++)
    {
        o << (i > 0 ? "," : "") << interfaces[i];
    }
    return o;
}

inline std::ostream& operator<<(std::ostream& o, const Plugin::ImplementationDesc& info)
{
    o << info.name;
    return o;
}

inline std::ostream& operator<<(std::ostream& o, const InterfaceDesc& info)
{
    o << Plugin::InterfaceData{ info.name, info.version };
    return o;
}


template <class T>
std::string toString(const T& x)
{
    std::ostringstream ss;
    ss << x;
    return ss.str();
}

#define CSTR(x) toString(x).c_str()


} // namespace cucim

#endif // CUCIM_CUCIM_PLUGIN_H
