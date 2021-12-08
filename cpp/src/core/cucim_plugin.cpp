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


#include "cucim_plugin.h"
#include "cucim/core/framework.h"
#include "cucim/core/plugin_util.h"

#include <algorithm>
#include <cstring>
#include <filesystem>

namespace cucim
{

Plugin::Plugin()
    : index_(0),
      resolve_state_(ResolveState::kUnused),
      plugin_desc_(),
      library_handle_(nullptr),
      on_get_framework_version_(nullptr),
      on_register_(nullptr),
      on_get_deps_(nullptr),
      //      m_carbOnPluginPreStartupFn(nullptr),
      //      m_carbOnPluginStartupFn(nullptr),
      //      m_carbOnPluginShutdownFn(nullptr),
      //      m_carbOnPluginPostShutdownFn(nullptr),
      //      m_carbOnReloadDependencyFn(nullptr),
      is_loaded_(false),
      is_initialized_(false),
      is_in_initialization_(false),
      is_reloadable_(false),
      next_version_(0)
//      m_fileSystem(fs)
{
}

Plugin::Plugin(const std::string& file_path) : Plugin()
{
    auto file = std::filesystem::path(file_path);
    auto filename = file.filename().string();
    std::size_t pivot = filename.find("@");
    if (pivot != std::string::npos)
    {
        std::string plugin_name = filename.substr(0, pivot);
        name_ = std::move(plugin_name);
    }
    else
    {
        name_ = "cucim.unknown";
    }
    library_path_ = file_path;
}

Plugin::~Plugin()
{
    unload();
}

template <typename T>
bool Plugin::init_plugin_fn(T& handle, const char* name, bool optional) const
{
    handle = dynlib::get_library_symbol<T>(library_handle_, name);
    if (!handle && !optional)
    {
        CUCIM_LOG_WARN("[Plugin: %s] Could not locate the function: %s", name_cstr(), name);
        return false;
    }
    return true;
}


bool Plugin::prepare_file_to_load(std::string& out_lib_file_path, int version)
{
    (void) version;

    //    if (!is_reloadable_)
    //    {
    out_lib_file_path = library_path_;
    return true;
    //    }

    //    if (m_tempFolder.empty())
    //    {
    //        CUCIM_LOG_ERROR("Can't load plugin %s as reloadable, temp folder doesn't exist.", getName());
    //        m_reloadable = false;
    //        return true;
    //    }
    //
    //    m_lastWriteTime = m_fileSystem->getModTime(m_libFilePath.c_str());
    //
    //    extras::Path path(m_libFilePath.c_str());
    //    std::string newLibFilename = path.getStem() + std::to_string(version) + path.getExtension();
    //    auto newLibPath = m_tempFolder + "/" + newLibFilename;
    //
    //    if (!m_fileSystem->exists(newLibPath.c_str()))
    //    {
    //        if (!m_fileSystem->copy(m_libFilePath.c_str(), newLibPath.c_str()))
    //        {
    //            return false;
    //        }
    //
    //#if CARB_COMPILER_MSC
    //        extras::Path newPdbFile(newLibPath.c_str());
    //        newPdbFile.replaceExtension(".pdb");
    //
    //        if (!relinkAndCopyPdbFile(newLibPath.c_str(), newPdbFile))
    //        {
    //            CUCIM_LOG_WARN(
    //                "[Plugin: %s] Couldn't process PDB, debugging may be affected and/or reload may fail.",
    //                getName());
    //        }
    //#endif
    //    }
    //    out_lib_file_path = newLibPath;
    //    return true;
}

bool Plugin::fill_registration_data(int version, bool full, const std::string& lib_file)
{
    (void)lib_file;

    // Retrieve registration information
    PluginEntry entry;
    on_register_(get_framework(), &entry);

    // Versioned data to fill:
    VersionedData& d = data_[kVersionStateCurrent];

    // Sort interfaces by name to keep order always the same
    std::sort(entry.interfaces, entry.interfaces + entry.interface_count,
              [](const PluginEntry::Interface& a, const PluginEntry::Interface& b) -> bool {
                  return std::strcmp(a.desc.name, b.desc.name) < 0;
              });

    d.plugin_interfaces.resize(entry.interface_count);
    d.interfaces.resize(entry.interface_count);
    for (size_t i = 0; i < entry.interface_count; i++)
    {
        d.interfaces[i].store(entry.interfaces[i].desc);
        d.plugin_interfaces[i] = d.interfaces[i].to_interface_desc();
    }
    d.desc.store(entry.desc);
    name_ = d.desc.name;

    if (full)
    {
        //        // Load the plugin interfaces
        //        {
        //            // Prepare interface buffers count
        //            if (version == 0)
        //            {
        //                m_interfaceBufs.resize(entry.interfaceCount);
        //                m_interfaceParents.resize(entry.interfaceCount);
        //            }
        //            else
        //            {
        //                if (m_interfaceBufs.size() != entry.interfaceCount)
        //                {
        //                    CUCIM_LOG_ERROR(
        //                        "[Plugin: %s] New version is incompatible for reload: interfaces count changed.",
        //                        getName());
        //                    return false;
        //                }
        //            }
        //
        for (size_t i = 0; i < entry.interface_count; i++)
        {
            const void* iface_ptr = entry.interfaces[i].ptr;
            uint64_t iface_size = entry.interfaces[i].size;
            //                if (ifaceSize == 0 || ifacePtr == nullptr)
            //                {
            //                    CUCIM_LOG_ERROR("[Plugin: %s] Interface is empty.", name_cstr());
            //                    return false;
            //                }
            //                if (version == 0)
            //                {
            //                    // First time allocating place for an interface buffer of a particular interface
            //                    // let's for now reserve twice as much space in case the plugin will be reloaded (or
            //                    implementation
            //                    // changes to other version) in runtime. That would allow it to grow.
            //                    m_interfaceBufs[i].resize(ifaceSize * 2);
            //                }
            //                if (m_interfaceBufs[i].size() < ifaceSize)
            //                {
            //                    CUCIM_LOG_ERROR("[Plugin: %s] New version is incompatible for reload: interface size
            //                    grown too much.",
            //                                   getName());
            //                    return false;
            //                }
            //                    // Copy an interface in a buffer, that allows us to reuse the same pointer if a plugin
            //                    is reloaded. std::memcpy(m_interfaceBufs[i].data(), ifacePtr, ifaceSize);
            d.interfaces[i].ptr = const_cast<void*>(iface_ptr); // m_interfaceBufs[i].data();
            d.interfaces[i].size = iface_size;
        }
        //        }
    }
    //
    //    // Data sections:
    //    if (m_reloadable && full && !lib_file.empty())
    //    {
    //        // Failed to load sections
    //        if (!loadSections(m_fileSystem, m_libraryHandle, lib_file, d.bssSection, d.stateSection))
    //            m_reloadable = false;
    //    }
    //
    //    // Get dependencies
    //    d.dependencies.clear();
    //    d.pluginDependencies.clear();
    //    if (m_carbGetPluginDepsFn)
    //    {
    //        InterfaceDesc* depDescs;
    //        size_t depDescCount;
    //        m_carbGetPluginDepsFn(&depDescs, &depDescCount);
    //        d.dependencies.reserve(depDescCount);
    //        d.pluginDependencies.resize(depDescCount);
    //        for (size_t i = 0; i < depDescCount; i++)
    //        {
    //            d.dependencies.push_back({ depDescs[i].name, depDescs[i].version });
    //            d.pluginDependencies[i] = d.dependencies[i].to_interface_desc();
    //        }
    //    }

    // Fill PluginDesc
    plugin_desc_ = { get_impl_desc().to_plugin_impl(), library_path_.c_str(),        d.plugin_interfaces.data(),
                     d.plugin_interfaces.size(),       d.plugin_dependencies.data(), d.plugin_dependencies.size() };

    // Save version
    d.version = version;

    return true;
}

bool Plugin::check_framework_version()
{
    const Version version = on_get_framework_version_();
    if (kFrameworkVersion.major != version.major)
    {
        CUCIM_LOG_ERROR(
            "[Plugin: %s] Incompatible Framework API major version: %" PRIu32 "", name_cstr(), kFrameworkVersion.major);
        return false;
    }
    if (kFrameworkVersion.minor < version.minor)
    {
        CUCIM_LOG_ERROR(
            "[Plugin: %s] Incompatible Framework API minor version: %" PRIu32 "", name_cstr(), kFrameworkVersion.major);
        return false;
    }
    return true;
}


bool Plugin::try_load(int version, bool full)
{
    if (is_loaded_)
    {
        return is_loaded_;
    }
    // CUCIM_LOG_VERBOSE("[Plugin: %s] %s", name_cstr(), full ? "Loading..." : "Preloading...");

    std::string lib_file;
    if (!prepare_file_to_load(lib_file, version))
    {
        return false;
    }
    // Load library
    CUCIM_LOG_VERBOSE("[Plugin: %s] Loading the dynamic library from: %s", name_cstr(), lib_file.c_str());
    library_handle_ = dynlib::load_library(lib_file.c_str());

    if (!library_handle_)
    {
        CUCIM_LOG_ERROR("[Plugin: %s] Could not load the dynamic library from %s. Error: %s", name_cstr(),
                        lib_file.c_str(), dynlib::get_last_load_library_error().c_str());
        return false;
    }

    // Load all the plugin function handles
    if (!init_plugin_fn(on_get_framework_version_, kCuCIMOnGetFrameworkVersionFnName))
        return false;
    if (!check_framework_version())
        return false;
    if (!init_plugin_fn(on_register_, kCuCIMOnPluginRegisterFnName))
        return false;
    if (!init_plugin_fn(on_get_deps_, kCuCIMOnGetPluginDepsFnName, true))
        return false;

    //    if (full)
    //    {
    //        init_plugin_fn(m_carbOnPluginPreStartupFn, kCarbOnPluginPreStartupFnName, true);
    //        init_plugin_fn(m_carbOnPluginStartupFn, kCarbOnPluginStartupFnName, true);
    //        init_plugin_fn(m_carbOnPluginShutdownFn, kCarbOnPluginShutdownFnName, true);
    //        init_plugin_fn(m_carbOnPluginPostShutdownFn, kCarbOnPluginPostShutdownFnName, true);
    //        init_plugin_fn(m_carbOnReloadDependencyFn, kCarbOnReloadDependencyFnName, true);
    //    }

    // Register
    if (!fill_registration_data(version, full, lib_file))
    {
        CUCIM_LOG_ERROR("[Plugin: %s] Could not load the dynamic library from %s. Error: fill_registration_data() failed!",
                        name_cstr(), lib_file.c_str());
        return false;
    }

    // Load was successful
    // CUCIM_LOG_VERBOSE("[Plugin: %s] %s successfully. Version: %d", name_cstr(), full ? "loaded" : "preloaded",
    // version);
    is_loaded_ = true;
    return is_loaded_;
}


bool Plugin::load(int version, bool full)
{
    if (!try_load(version, full))
    {
        unload();
        return false;
    }
    return true;
}

void Plugin::unload()
{
    if (library_handle_)
    {
        dynlib::unload_library(library_handle_);
        library_handle_ = nullptr;
        is_loaded_ = false;
        CUCIM_LOG_VERBOSE("[Plugin: %s] Unloaded.", name_cstr());
    }
}


bool Plugin::preload(bool reloadable, bool unload)
{
    is_reloadable_ = reloadable;

    bool full_load = !unload;
    if (load(0, full_load))
    {
        if (unload)
            this->unload();
        return true;
    }
    return false;
}

Plugin::InitResult Plugin::ensure_initialized()
{
    // Fast path: already initialized
    if (is_initialized_)
    {
        return InitResult::kAlreadyInitialized;
    }

    // Check again after locking mutex
    std::lock_guard<std::recursive_mutex> lock(init_lock_);
    if (is_initialized_)
    {
        return InitResult::kAlreadyInitialized;
    }

    return initialize() ? InitResult::kDidInitialize : InitResult::kFailedInitialize;
}

bool Plugin::initialize()
{
    std::lock_guard<std::recursive_mutex> lock(init_lock_);

    // another thread could have beaten us into the locked region between when the 'initialized'
    // flag was originally checked (before this call) and when the lock was actually acquired.
    // If this flag is set, that means the other thread won and the plugin has already been
    // fully initialized.  In this case there is nothing left for us to do here but succeed.
    if (is_initialized_)
    {
        return true;
    }

    if (is_in_initialization_)
    {
        // Don't recursively initialize
        return false;
    }

    CUCIM_LOG_INFO("Initializing plugin: %s (interfaces: %s) (impl: %s)", name_cstr(), CSTR(get_interfaces()),
                   CSTR(get_impl_desc()));

    is_in_initialization_ = true;

    // failed to load the plugin library iself => fail and allow the caller to try again later.
    if (load(next_version_++))
    {
        //        // run the pre-startup function for the plugin.
        //        if (m_carbOnPluginPreStartupFn)
        //        {
        //            m_carbOnPluginPreStartupFn();
        //        }
        //
        //        // run the startup function for the plugin.
        //        if (m_carbOnPluginStartupFn)
        //        {
        //            m_carbOnPluginStartupFn();
        //        }

        is_initialized_ = true;
    }

    is_in_initialization_ = false;

    return is_initialized_;
}

void Plugin::terminate()
{
    std::lock_guard<std::recursive_mutex> lock(init_lock_);

    if (!is_initialized_ || !is_loaded_)
        return;

    //    if (m_carbOnPluginShutdownFn)
    //    {
    //        m_carbOnPluginShutdownFn();
    //    }
    //
    //    if (m_carbOnPluginPostShutdownFn)
    //    {
    //        m_carbOnPluginPostShutdownFn();
    //    }

    is_initialized_ = false;
}


static void update_if_changed(std::string& str, const char* value)
{
    if (str != value)
        str = value;
}

void Plugin::InterfaceData::store(const InterfaceDesc& desc)
{
    update_if_changed(name, desc.name);
    version = desc.version;
}

void Plugin::ImplementationDesc::store(const PluginImplDesc& desc)
{
    update_if_changed(name, desc.name);
    version = desc.version;
    update_if_changed(build, desc.build);
    update_if_changed(author, desc.author);
    update_if_changed(description, desc.description);
    update_if_changed(long_description, desc.long_description);
    update_if_changed(license, desc.license);
    update_if_changed(url, desc.url);
    update_if_changed(platforms, desc.platforms);
    hot_reload = desc.hot_reload;
}

} // namespace cucim