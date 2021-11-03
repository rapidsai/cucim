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

#include "cucim_framework.h"
#include "cucim_plugin.h"

#include <memory>
#include <algorithm>

namespace cucim
{

CuCIMFramework::CuCIMFramework()
{
}

CuCIMFramework::~CuCIMFramework()
{
    g_cucim_framework = nullptr;
    //    assert::deregisterAssertForClient();
    //    logging::deregisterLoggingForClient();
}


bool CuCIMFramework::register_plugin(const char* client_name, const PluginRegistrationDesc& desc)
{
    (void)client_name;
    (void)desc;

    return false;
}

bool CuCIMFramework::unregister_plugin(const char* plugin_name)
{
    ScopedLock g(mutex_);

    Plugin* plugin = get_plugin(plugin_name);
    if (!plugin)
    {
        CUCIM_LOG_WARN("unregisterPlugin: Failed to find a plugin with a name: %s.", plugin_name ? plugin_name : "");
        return false;
    }

    std::vector<Plugin*> plugins_to_unload;
    if (try_terminate_plugin(plugin, &plugins_to_unload))
    {
        for (size_t idx = 0, plugin_count = plugins_to_unload.size(); idx < plugin_count; ++idx)
        {
            plugins_to_unload[idx]->unload();
        }
        unregister_plugin(plugin);
        return true;
    }

    return false;
}

void CuCIMFramework::unregister_plugin(Plugin* plugin)
{
    // Remove plugin from all storages

    name_to_plugin_index_.erase(plugin->name_cstr());

    const std::string& file_path = plugin->library_path();
    if (!file_path.empty())
        library_path_to_plugin_index_.erase(file_path);

    // Remove all its interfaces from candidates and reset selected if it not valid anymore
    const auto& interfaces = plugin->get_interfaces();
    for (size_t i = 0; i < interfaces.size(); i++)
    {
        CandidatesEntry& entry = interface_candidates_[interfaces[i].name];
        for (size_t k = 0; k < entry.candidates.size(); k++)
        {
            if (entry.candidates[k].plugin_index == plugin->index_)
            {
                // Replace with last element (unordered fast remove)
                if (entry.candidates.size() > 1)
                {
                    entry.candidates[k] = entry.candidates.back();
                }
                entry.candidates.resize(entry.candidates.size() - 1);
            }
        }

        if (!entry.selected.get_plugin(plugin_manager_))
            entry.selected = {};
    }

    plugin_manager_.remove_plugin(plugin->index_);
    delete plugin;
}

void CuCIMFramework::load_plugins(const PluginLoadingDesc& desc)
{
    (void)desc;
}


bool CuCIMFramework::register_plugin(const std::shared_ptr<Plugin>& plugin)
{
    ScopedLock g(mutex_);

    // TODO: duplicate check

    // Success storing plugin in all registries
    size_t plugin_index = plugin_manager_.add_plugin(plugin);
    plugin->index_ = plugin_index;

    const auto& interfaces = plugin->get_interfaces();
    for (size_t i = 0; i < interfaces.size(); i++)
    {
        interface_candidates_[interfaces[i].name].candidates.push_back({ plugin_index, i });
    }

    // TODO: reloadable check
    name_to_plugin_index_[plugin->name_cstr()] = plugin_index;
    return true;
}

size_t CuCIMFramework::get_plugin_count() const
{
    ScopedLock g(mutex_);
    return plugin_manager_.get_plugin_indices().size();
}

void CuCIMFramework::get_plugins(PluginDesc* out_plugins) const
{
    ScopedLock g(mutex_);
    const std::unordered_set<size_t>& plugins = plugin_manager_.get_plugin_indices();
    size_t i = 0;
    for (const auto& plugin_index : plugins)
    {
        if (out_plugins)
        {
            out_plugins[i++] = plugin_manager_.get_plugin(plugin_index)->get_plugin_desc();
        }
    }
}

size_t CuCIMFramework::get_plugin_index(const char* name) const
{
    auto it = name_to_plugin_index_.find(name);
    if (it != name_to_plugin_index_.end())
    {
        return it->second;
    }
    return kInvalidPluginIndex;
}
Plugin* CuCIMFramework::get_plugin(size_t index) const
{
    return index != kInvalidPluginIndex ? plugin_manager_.get_plugin(index) : nullptr;
}

Plugin* CuCIMFramework::get_plugin(const char* name) const
{
    return get_plugin(get_plugin_index(name));
}


Plugin* CuCIMFramework::get_plugin_by_library_path(const std::string& library_path)
{

    auto it = library_path_to_plugin_index_.find(library_path);
    if (it != library_path_to_plugin_index_.end())
    {
        return get_plugin(it->second);
    }
    return nullptr;
}

bool CuCIMFramework::resolve_plugin_dependencies(Plugin* plugin)
{
    if (plugin->resolve_state_ == Plugin::ResolveState::kResolved)
        return true;

    //    const bool failed_before = (plugin->resolve_state_ == Plugin::ResolveState::kFailed);

    plugin->resolve_state_ = Plugin::ResolveState::kInprocess;
    //    bool resolveFailed = false;
    //    for (auto& dep : plugin->getDeps())
    //    {
    //        if (!resolveInterfaceDependencyWithLogging(dep))
    //        {
    //            CUCIM_LOG_ERROR("[Plugin: %s] Dependency: %s failed to be resolved.", plugin->getName(), CSTR(dep));
    //            resolveFailed = true;
    //        }
    //    }
    //
    //    if (resolveFailed)
    //    {
    //        plugin->resolveState = Plugin::ResolveState::eFailed;
    //        return false;
    //    }
    //
    //    if (failed_before)
    //    {
    //        CUCIM_LOG_INFO("[Plugin: %s] Dependencies were resolved now (failed before).", plugin->getName());
    //    }

    plugin->resolve_state_ = Plugin::ResolveState::kResolved;
    return true;
}

bool CuCIMFramework::resolve_interface_dependency(const Plugin::InterfaceData& desc, bool log_errors)
{
    (void)log_errors;

    const auto it = interface_candidates_.find(desc.name);
    if (it != interface_candidates_.cend())
    {
        // Check for selected (default) plugins first
        CandidatesEntry& entry = (*it).second;
        Plugin* plugin = entry.selected.get_plugin(plugin_manager_);
        if (plugin)
        {
            if (plugin->resolve_state_ == Plugin::ResolveState::kInprocess)
            {
                //                // todo: Give more insight on how it happened
                //                if (log_errors)
                //                {
                //                    CUCIM_LOG_ERROR(
                //                        "Circular dependency detected! Interface: %s requested. But plugin with an
                //                        interface: %s is already in resolving state.", CSTR(desc),
                //                        CSTR(entry.selected.get_interface_desc(m_registry)));
                //                }
                return false;
            }
            if (!is_version_semantically_compatible(
                    desc.version, entry.selected.get_interface_desc(plugin_manager_).version))
            {
                //                if (log_errors)
                //                {
                //                    CUCIM_LOG_ERROR(
                //                        "Interface: %s requested. But there is already a plugin with an interface: %s
                //                        loaded. Versions are incompatible. Only one version of the same
                //                        interface/plugin can exist at a time.", CSTR(desc),
                //                        CSTR(entry.selected.get_interface_desc(m_registry)));
                //                }
                return false;
            }
            return true;
        }

        // Search for all plugins with that interface for matching version. If any of them marked as default - pick it
        // and early out. If there is no defaults - the first one to match is selected, which should the highest
        // compatible version.
        Plugin::Interface candidate = {};
        for (Plugin::Interface& c : entry.candidates)
        {
            // Check that candidate is still valid (could have been unregistered)
            Plugin* candidatePlugin = c.get_plugin(plugin_manager_);
            if (candidatePlugin)
            {
                if (candidate.plugin_index == kInvalidPluginIndex)
                    candidate = c;
                if (c.get_plugin(plugin_manager_)->name_str() == entry.specifiedDefaultPlugin)
                {
                    candidate = c;
                    break;
                }
            }
        }

        // Resolve all dependencies recursively for the candidate if it has changed
        Plugin* candidate_plugin = candidate.get_plugin(plugin_manager_);
        if (candidate_plugin && entry.selected.plugin_index != candidate_plugin->index_)
        {
            // set candidate as selected to catch circular dependencies
            entry.selected = candidate;

            if (resolve_plugin_dependencies(candidate_plugin))
            {
                //                // the default plugin was just set for this interface: notify subscribers
                //                CUCIM_LOG_INFO(
                //                    "FrameworkImpl::resolveInterfaceDependency(): default plugin: %s was set for an
                //                    interface: %s", candidate_plugin->getName(), CSTR(desc));
                //                checkIfBasicPluginsAcquired(entry);
                return is_version_semantically_compatible(
                    desc.version, candidate.get_interface_desc(plugin_manager_).version);
            }
            else
            {
                entry.selected = {};
                return false;
            }
        }
    }
    return false;
}

bool CuCIMFramework::resolve_interface_dependency_with_logging(const Plugin::InterfaceData& desc)
{
    return resolve_interface_dependency(desc, true);
}

bool CuCIMFramework::resolve_interface_dependency_no_logging(const Plugin::InterfaceData& desc)
{
    return resolve_interface_dependency(desc, false);
}


bool CuCIMFramework::try_terminate_plugin(Plugin* plugin, std::vector<Plugin*>* plugins_to_unload)
{
    //    // Terminate plugin if all clients released it
    //    if (!plugin->hasAnyParents())
    {
        // Shut down the plugin first
        plugin->terminate();

        //        // Release parent <-> child dependency recursively
        //        const Plugin::InterfaceSet& children = plugin->getChildren();
        //        for (const Plugin::Interface& child : children)
        //        {
        //            releasePluginDependency(plugin->getName(), child, pluginsToUnload);
        //        }
        //        plugin->clearChildren();

        if (plugins_to_unload)
        {
            plugins_to_unload->push_back(plugin);
        }
        else
        {
            CUCIM_LOG_WARN("%s: out-of-order unloading plugin %s", __func__, plugin->name_cstr());
            plugin->unload();
        }

        return true;
    }

    return false;
}


Plugin::Interface CuCIMFramework::get_default_plugin(const InterfaceDesc& desc, bool optional)
{
    const auto it = interface_candidates_.find(desc.name);
    if (it != interface_candidates_.cend())
    {
        CandidatesEntry& entry = (*it).second;

        // If there is already selected plugin for this interface name, take it. Otherwise run
        // resolve process with will select plugins for all dependent interfaces recursively
        if (!entry.selected.get_plugin(plugin_manager_))
        {
            resolve_interface_dependency_no_logging(Plugin::InterfaceData{ desc.name, desc.version });
        }

        // In case of successful resolve there should be a valid candidate in this registry entry
        const Plugin::Interface& candidate = entry.selected;
        if (candidate.get_plugin(plugin_manager_))
        {
            // The version still could mismatch in case the candidate is the result of previous getInterface
            // calls
            if (!is_version_semantically_compatible(desc.version, candidate.get_interface_desc(plugin_manager_).version))
            {
                if (!optional)
                {
                    //                    CUCIM_LOG_ERROR(
                    //                        "Interface: %s requested. But there is already a plugin with an interface:
                    //                        %s loaded. Versions are incompatible. Only one version of the same
                    //                        interface/plugin can exist at a time.", CSTR(desc),
                    //                        CSTR(candidate.get_interface_desc(plugin_manager_)));
                }
                return {};
            }
            return candidate;
        }
    }
    return {};
}


Plugin::Interface CuCIMFramework::get_specific_plugin(const InterfaceDesc& desc, const char* plugin_name, bool optional)
{
    // Search plugin by name
    Plugin* plugin = get_plugin(plugin_name);
    if (!plugin)
    {
        if (!optional)
        {
            //            CUCIM_LOG_ERROR("Failed to find a plugin with a name: %s", plugin_name);
        }
        return {};
    }

    // The interface version or name could mismatch, need to check
    const auto& interfaces = plugin->get_interfaces();
    Plugin::Interface candidate = {};
    for (size_t i = 0; i < interfaces.size(); i++)
    {
        if (interfaces[i].name == desc.name && is_version_semantically_compatible(desc.version, interfaces[i].version))
        {
            candidate = { plugin->index_, i };
            break;
        }
    }

    Plugin* candidatePlugin = candidate.get_plugin(plugin_manager_);
    if (!candidatePlugin)
    {
        if (!optional)
        {
            //            CUCIM_LOG_ERROR("Interface: %s with a plugin name: %s requested. Interface mismatched, it
            //            has interfaces: %s",
            //                           CSTR(desc), plugin->name_cstr(), CSTR(plugin->get_interfaces()));
        }
        return {};
    }

    // Check deps resolve, the actual resolve process could be triggered here if that's the first time plugin is
    // requested
    if (!resolve_plugin_dependencies(candidatePlugin))
    {
        if (!optional)
        {
            //            CUCIM_LOG_ERROR(
            //                "Interface: %s with a plugin name: %s requested. One of the plugin's dependencies failed
            //                to resolve.", CSTR(desc), candidatePlugin->name_cstr());
        }
        return {};
    }

    return candidate;
}

void CuCIMFramework::unload_all_plugins()
{
    ScopedLock g(mutex_);

    CUCIM_LOG_VERBOSE("Unload all plugins.");

    // Get all plugins from the registry and copy the set (because we are updating registry it inside of loops below)
    std::unordered_set<size_t> plugins = plugin_manager_.get_plugin_indices();

    // Unregister all plugins which aren't initialized (not used atm).
    for (size_t plugin_index : plugins)
    {
        Plugin* plugin = plugin_manager_.get_plugin(plugin_index);
        if (plugin && !plugin->is_initialized())
            unregister_plugin(plugin);
    }

    // Terminate and unload all plugins in reverse order compared to initialization
    for (auto it = plugin_load_order_.rbegin(); it != plugin_load_order_.rend(); ++it)
    {
        Plugin* plugin = get_plugin(*it);
        if (plugin)
            plugin->terminate();
    }
    for (auto it = plugin_load_order_.rbegin(); it != plugin_load_order_.rend(); ++it)
    {
        Plugin* plugin = get_plugin(*it);
        if (plugin)
            plugin->unload();
    }
    plugin_load_order_.clear();

    // Destroy all plugins in registry
    for (size_t plugin_index : plugins)
    {
        Plugin* plugin = plugin_manager_.get_plugin(plugin_index);
        if (plugin)
            unregister_plugin(plugin);
    }

    //    m_reloadablePlugins.clear();
    interface_candidates_.clear();

    // Verify that now everything is back to initial state
    CUCIM_ASSERT(plugin_manager_.get_plugin_indices().empty() == true);
    CUCIM_ASSERT(name_to_plugin_index_.empty() == true);
    CUCIM_ASSERT(library_path_to_plugin_index_.empty() == true);
}

void* CuCIMFramework::acquire_interface(const char* client, const InterfaceDesc& desc, const char* plugin_name, bool optional)
{
    if (!client)
        return nullptr;

    ScopedLock g(mutex_);

    const bool acquire_as_default = plugin_name ? false : true;
    Plugin::Interface candidate =
        acquire_as_default ? get_default_plugin(desc, optional) : get_specific_plugin(desc, plugin_name, optional);
    Plugin* plugin = get_plugin(candidate.plugin_index);
    if (!plugin)
    {
        if (!optional)
        {
            //            CUCIM_LOG_ERROR(
            //                "Failed to acquire interface: %s, by client: %s (plugin name: %s)", CSTR(desc), client,
            //                pluginName);
        }
        return nullptr;
    }

    if (!plugin->is_initialized())
    {
        // Don't hold the mutex during initialization
        g.unlock();

        // Lazily initialize plugins only when requested (on demand)
        Plugin::InitResult result = plugin->ensure_initialized();

        g.lock();

        if (result != Plugin::InitResult::kAlreadyInitialized)
        {
            if (result == Plugin::InitResult::kFailedInitialize)
            {
                if (!optional)
                {
                    if (plugin->is_in_initialization())
                    {
                        //                        CUCIM_LOG_ERROR(
                        //                            "Trying to acquire plugin during it's initialization: %s
                        //                            (interfaces: %s) (impl: %s). Circular acquire calls.",
                        //                            plugin->name_cstr(), CSTR(plugin->get_interfaces()),
                        //                            CSTR(plugin->get_impl_desc()));
                    }
                    else
                    {
                        //                        CUCIM_LOG_ERROR("Plugin load failed: %s (interfaces: %s) (impl:
                        //                        %s).", plugin->name_cstr(),
                        //                                       CSTR(plugin->get_interfaces()),
                        //                                       CSTR(plugin->get_impl_desc()));
                    }
                }
                return nullptr;
            }

            // Add to the load order since loading was successful
            // TODO: Replace load order with dependency graph
            if (std::find(plugin_load_order_.begin(), plugin_load_order_.end(), plugin->index_) ==
                plugin_load_order_.end())
            {
                plugin_load_order_.push_back(plugin->index_);
            }
        }
    }

    // Finish up now that the plugin is initialized
    CUCIM_ASSERT(g.owns_lock());

    void* iface = candidate.get_interface_desc(plugin_manager_).ptr;
    CUCIM_ASSERT(iface);

    // Store plugin in the interface->plugin map
    ptr_to_interface_[iface] = candidate;

    //    // Saving callers/clients of a plugin.
    //    plugin->addParent(candidate.interfaceIndex, client, acquireAsDefault);
    //
    //    // Saving child for a parent
    //    if (parent)
    //        parent->addChild(candidate);

    return iface;
}


void* CuCIMFramework::acquire_interface_from_library(const char* client,
                                                     const InterfaceDesc& desc,
                                                     const char* library_path,
                                                     bool optional)
{
    ScopedLock g(mutex_);
    // Check if plugin with this library path was already loaded

    const std::string canonical_library_path(library_path);

    Plugin* plugin = get_plugin_by_library_path(canonical_library_path);
    if (!plugin)
    {
        // It was not loaded, try to register such plugin and get it again
        if (register_plugin(canonical_library_path))
        {
            plugin = get_plugin_by_library_path(canonical_library_path);
        }
    }

    if (plugin)
    {
        // Library path leads to valid plugin which now was loaded, try acquire requested interface on it:
        return acquire_interface(client, desc, plugin->name_cstr(), optional);
    }

    return nullptr;
}
bool CuCIMFramework::register_plugin(const std::string& file_path, bool reloadable, bool unload)
{
    std::shared_ptr<Plugin> plugin = std::make_shared<Plugin>(file_path);

    // Try preload
    if (!plugin->preload(reloadable, unload))
    {
        //        CUCIM_LOG_WARN("Potential plugin preload failed: %s", plugin->library_path());
        return false;
    }

    if (register_plugin(plugin))
    {
        library_path_to_plugin_index_[file_path] = plugin->index_;
        return true;
    }
    return false;
}

// cuCIM-specific methods

void CuCIMFramework::load_plugin(const char* library_path)
{

    ScopedLock g(mutex_);

    const std::string canonical_library_path(library_path);

    Plugin* plugin = get_plugin_by_library_path(canonical_library_path);
    // Check if plugin with this library path was already loaded
    if (!plugin)
    {
        // It was not loaded, try to register such plugin and get it again
        register_plugin(canonical_library_path);
    }
}

std::string& CuCIMFramework::get_plugin_root()
{
    return plugin_root_path_;
}

void CuCIMFramework::set_plugin_root(const char* path)
{
    plugin_root_path_ = std::string(path);
}

} // namespace cucim