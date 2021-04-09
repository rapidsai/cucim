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
#define CUCIM_EXPORTS

#include "cucim/core/framework.h"
#include "cucim_framework.h"
#include "cucim/macros/defines.h"
#include <memory>
#include <mutex>


CUCIM_FRAMEWORK_GLOBALS("cucim")


namespace cucim
{

static std::unique_ptr<CuCIMFramework> g_framework;


static bool register_plugin(const char* client_name, const PluginRegistrationDesc& desc)
{
    CUCIM_ASSERT(g_framework);
    return g_framework->register_plugin(client_name, desc);
}

// static void load_plugins(const PluginLoadingDesc& desc)
//{
//    CUCIM_ASSERT(g_framework);
//    return g_framework->load_plugins(desc);
//}

static void* acquire_interface_from_library_with_client(const char* client_name,
                                                        InterfaceDesc desc,
                                                        const char* library_path)
{
    CUCIM_ASSERT(g_framework);
    return g_framework->acquire_interface_from_library(client_name, desc, library_path, false);
}

static void unload_all_plugins()
{
    CUCIM_ASSERT(g_framework);
    g_framework->unload_all_plugins();
}

static const char* get_plugin_root()
{
    CUCIM_ASSERT(g_framework);
    return g_framework->get_plugin_root().c_str();
}

static void set_plugin_root(const char* path)
{
    CUCIM_ASSERT(g_framework);
    g_framework->set_plugin_root(path);
}

static Framework get_framework_impl()
{
    // clang-format off
    return
    {
        register_plugin,
        acquire_interface_from_library_with_client,
        unload_all_plugins,
        get_plugin_root,
        set_plugin_root,
    };
    // clang-format on
}


namespace
{
std::mutex& acquire_framework_mutex()
{
    static std::mutex mutex;
    return mutex;
}
} // namespace


CUCIM_API Framework* acquire_framework(const char* app_name, Version framework_version)
{
    (void) app_name;
    (void) framework_version;
    //    if (!is_version_semantically_compatible(kFrameworkVersion, frameworkVersion))
    //    {
    //        // Using CARB_LOG here is pointless because logging hasn't been set up yet.
    //        fprintf(stderr,
    //                "[App: %s] Incompatible Framework API version. Framework version: %" PRIu32 ".%" PRIu32
    //                ". Application requested version: %" PRIu32 ".%" PRIu32 ".\n",
    //                appName, kFrameworkVersion.major, kFrameworkVersion.minor, frameworkVersion.major,
    //                frameworkVersion.minor);
    //        return nullptr;
    //    }

    static Framework framework = get_framework_impl();
    if (!g_framework)
    {
        std::lock_guard<std::mutex> g(acquire_framework_mutex());
        if (!g_framework) // Try again after locking mutex
        {
            g_framework = std::make_unique<CuCIMFramework>();
            g_cucim_framework = &framework;
            g_cucim_client_name = "cucim";
        }
    }
    return &framework;
}

CUCIM_API void release_framework()
{
    std::lock_guard<std::mutex> g(acquire_framework_mutex());
    if (g_framework)
    {
        g_framework->unload_all_plugins();
        g_cucim_framework = nullptr;
        g_framework.reset(nullptr);
    }
}


} // namespace cucim
