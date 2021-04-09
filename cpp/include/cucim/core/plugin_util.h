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

#ifndef CUCIM_PLUGIN_UTIL_H
#define CUCIM_PLUGIN_UTIL_H

#include "plugin.h"

constexpr const char* const kCuCIMOnGetFrameworkVersionFnName =
    "cucim_on_get_framework_version"; // type: OnGetFrameworkVersionFn

constexpr const char* const kCuCIMOnPluginRegisterFnName = "cucim_on_plugin_register"; // type: OnPluginRegisterFn

/*
 * Optional functions:
 */
constexpr const char* const kCuCIMOnGetPluginDepsFnName = "cucim_on_get_plugin_deps"; // type: OnGetPluginDepsFn


// const char* const kCarbOnPluginPreStartupFnName = "carbOnPluginPreStartup"; // type: OnPluginPreStartupFn
// const char* const kCarbOnPluginStartupFnName = "carbOnPluginStartup"; // type: OnPluginStartupFn
//
// const char* const kCarbOnPluginShutdownFnName = "carbOnPluginShutdown"; // type: OnPluginShutdownFn
// const char* const kCarbOnPluginPostShutdownFnName = "carbOnPluginPostShutdown"; // type: OnPluginPostShutdownFn

//
// const char* const kCarbOnReloadDependencyFnName = "carbOnReloadDependency"; // type: OnReloadDependencyFn

/**
 * FOR_EACH macro implementation, use as FOR_EACH(OTHER_MACRO, p0, p1, p2,)
 */
#define EXPAND(x) x
#define FE_1(WHAT, X) EXPAND(WHAT(X))
#define FE_2(WHAT, X, ...) EXPAND(WHAT(X) FE_1(WHAT, __VA_ARGS__))
#define FE_3(WHAT, X, ...) EXPAND(WHAT(X) FE_2(WHAT, __VA_ARGS__))
#define FE_4(WHAT, X, ...) EXPAND(WHAT(X) FE_3(WHAT, __VA_ARGS__))
#define FE_5(WHAT, X, ...) EXPAND(WHAT(X) FE_4(WHAT, __VA_ARGS__))
#define FE_6(WHAT, X, ...) EXPAND(WHAT(X) FE_5(WHAT, __VA_ARGS__))
#define FE_7(WHAT, X, ...) EXPAND(WHAT(X) FE_6(WHAT, __VA_ARGS__))
#define FE_8(WHAT, X, ...) EXPAND(WHAT(X) FE_7(WHAT, __VA_ARGS__))
#define FE_9(WHAT, X, ...) EXPAND(WHAT(X) FE_8(WHAT, __VA_ARGS__))
#define FE_10(WHAT, X, ...) EXPAND(WHAT(X) FE_9(WHAT, __VA_ARGS__))
#define FE_11(WHAT, X, ...) EXPAND(WHAT(X) FE_10(WHAT, __VA_ARGS__))
#define FE_12(WHAT, X, ...) EXPAND(WHAT(X) FE_11(WHAT, __VA_ARGS__))
#define FE_13(WHAT, X, ...) EXPAND(WHAT(X) FE_12(WHAT, __VA_ARGS__))
#define FE_14(WHAT, X, ...) EXPAND(WHAT(X) FE_13(WHAT, __VA_ARGS__))
#define FE_15(WHAT, X, ...) EXPAND(WHAT(X) FE_14(WHAT, __VA_ARGS__))
#define FE_16(WHAT, X, ...) EXPAND(WHAT(X) FE_15(WHAT, __VA_ARGS__))
#define FE_17(WHAT, X, ...) EXPAND(WHAT(X) FE_16(WHAT, __VA_ARGS__))
#define FE_18(WHAT, X, ...) EXPAND(WHAT(X) FE_17(WHAT, __VA_ARGS__))
#define FE_19(WHAT, X, ...) EXPAND(WHAT(X) FE_18(WHAT, __VA_ARGS__))
#define FE_20(WHAT, X, ...) EXPAND(WHAT(X) FE_19(WHAT, __VA_ARGS__))
#define FE_21(WHAT, X, ...) EXPAND(WHAT(X) FE_20(WHAT, __VA_ARGS__))
#define FE_22(WHAT, X, ...) EXPAND(WHAT(X) FE_21(WHAT, __VA_ARGS__))
#define FE_23(WHAT, X, ...) EXPAND(WHAT(X) FE_22(WHAT, __VA_ARGS__))
#define FE_24(WHAT, X, ...) EXPAND(WHAT(X) FE_23(WHAT, __VA_ARGS__))
#define FE_25(WHAT, X, ...) EXPAND(WHAT(X) FE_24(WHAT, __VA_ARGS__))
#define FE_26(WHAT, X, ...) EXPAND(WHAT(X) FE_25(WHAT, __VA_ARGS__))
#define FE_27(WHAT, X, ...) EXPAND(WHAT(X) FE_26(WHAT, __VA_ARGS__))
#define FE_28(WHAT, X, ...) EXPAND(WHAT(X) FE_27(WHAT, __VA_ARGS__))
#define FE_29(WHAT, X, ...) EXPAND(WHAT(X) FE_28(WHAT, __VA_ARGS__))
#define FE_30(WHAT, X, ...) EXPAND(WHAT(X) FE_29(WHAT, __VA_ARGS__))
#define FE_31(WHAT, X, ...) EXPAND(WHAT(X) FE_30(WHAT, __VA_ARGS__))
#define FE_32(WHAT, X, ...) EXPAND(WHAT(X) FE_31(WHAT, __VA_ARGS__))


//... repeat as needed
#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, \
                  _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, NAME, ...)                                         \
    NAME
#define FOR_EACH(action, ...)                                                                                          \
    EXPAND(GET_MACRO(__VA_ARGS__, FE_32, FE_31, FE_30, FE_29, FE_28, FE_27, FE_26, FE_25, FE_24, FE_23, FE_22, FE_21,  \
                     FE_20, FE_19, FE_18, FE_17, FE_16, FE_15, FE_14, FE_13, FE_12, FE_11, FE_10, FE_9, FE_8, FE_7,    \
                     FE_6, FE_5, FE_4, FE_3, FE_2, FE_1)(action, __VA_ARGS__))


#define DECLARE_FILL_FUNCTION(X) void fill_interface(X& iface);

/**
 * Macros to declare a plugin implementation with custom static initializer.
 *
 * It does the following:
 *
 *     1. Defines cucim_on_get_framework_version and cucim_on_plugin_register functions.
 *     2. Defines global framework variable for cucim::getFramework() to work.
 *     3. Defines global client variable (which is set to a plugin name). It is used for acquiring interfaces,
 * such that framework knows who calls it.
 *     4. Forward declares void fill_interface(InterfaceType& iface) functions for every interface to be
 * used to provide interfaces to the framework.
 *
 * This macro must be defined in a global namespace.
 *
 * @param impl The PluginImplDesc constant to be used as plugin description.
 * @param ... One or more interface types to be implemented by the plugin. Interface is a struct with
 * CUCIM_PLUGIN_INTERFACE() macro inside.
 */
#define CUCIM_PLUGIN_IMPL_WITH_INIT(impl, ...)                                                                       \
                                                                                                                       \
    /* Forward declare fill functions for every interface */                                                           \
    FOR_EACH(DECLARE_FILL_FUNCTION, __VA_ARGS__)                                                                       \
                                                                                                                       \
    template <typename T1>                                                                                             \
    void fill_interface(cucim::PluginEntry::Interface* interfaces)                                                                   \
    {                                                                                                                  \
        interfaces[0].desc = T1::get_interface_desc();                                                                 \
        static T1 s_plugin_interface;                                                                                  \
        fill_interface(s_plugin_interface);                                                                            \
        interfaces[0].ptr = &s_plugin_interface;                                                                       \
        interfaces[0].size = sizeof(T1);                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    template <typename T1, typename T2, typename... Types>                                                             \
    void fill_interface(cucim::PluginEntry::Interface* interfaces)                                                                   \
    {                                                                                                                  \
        fill_interface<T1>(interfaces);                                                                                \
        fill_interface<T2, Types...>(interfaces + 1);                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    template <typename... Types>                                                                                       \
    static void on_plugin_register(cucim::Framework* framework, cucim::PluginEntry* out_entry)                                                     \
    {                                                                                                                  \
        static cucim::PluginEntry::Interface s_interfaces[sizeof...(Types)];                                                         \
        fill_interface<Types...>(s_interfaces);                                                                        \
        out_entry->interfaces = s_interfaces;                                                                          \
        out_entry->interface_count = sizeof(s_interfaces) / sizeof(s_interfaces[0]);                                   \
        out_entry->desc = impl;                                                                                        \
                                                                                                                       \
        g_cucim_framework = framework;                                                                                               \
        g_cucim_client_name = impl.name;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    CUCIM_API void cucim_on_plugin_register(cucim::Framework* framework, cucim::PluginEntry* out_entry)                                                        \
    {                                                                                                                  \
        on_plugin_register<__VA_ARGS__>(framework, out_entry);                                                         \
    }                                                                                                                  \
                                                                                                                       \
    CUCIM_API cucim::Version cucim_on_get_framework_version()                                                                                    \
    {                                                                                                                  \
        return cucim::kFrameworkVersion;                                                                                             \
    }


/**
 * Macros to declare a plugin implementation dependencies.
 *
 * If a plugin lists an interface "A" as dependency it is guaranteed that Framework::acquireInterface<A>() call
 * will return it, otherwise it can return nullptr. Framework checks and resolves all dependencies before loading the
 * plugin.
 *
 * @param ... One or more interface types to list as dependencies for this plugin.
 */
#define CUCIM_PLUGIN_IMPL_DEPS(...)                                                                                  \
    template <typename... Types>                                                                                       \
    static void get_plugin_deps_typed(struct cucim::InterfaceDesc** deps, size_t* count)                                   \
    {                                                                                                                  \
        static cucim::InterfaceDesc depends[] = { Types::get_interface_desc()... };                                        \
        *deps = depends;                                                                                               \
        *count = sizeof(depends) / sizeof(depends[0]);                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    CUCIM_API void cucim_on_get_plugin_deps(struct cucim::InterfaceDesc** deps, size_t* count)                               \
    {                                                                                                                  \
        get_plugin_deps_typed<__VA_ARGS__>(deps, count);                                                               \
    }

/**
 * Macro to declare no plugin implementation dependencies.
 */
#define CUCIM_PLUGIN_IMPL_NO_DEPS()                                                                                  \
    CUCIM_API void cucim_on_get_plugin_deps(struct cucim::InterfaceDesc** deps, size_t* count)                       \
    {                                                                                                                  \
        *deps = nullptr;                                                                                               \
        *count = 0;                                                                                                    \
    }

/**
 * Macro to declare a plugin implementation with an empty scoped initializer.
 * Useful for those who wants to use bare Carbonite Framework without the pre-registered plugins,
 * contrary to what CUCIM_PLUGIN_IMPL suggests.
 */
#define CUCIM_PLUGIN_IMPL_MINIMAL(impl, ...)                                                                         \
    CUCIM_FRAMEWORK_GLOBALS(kPluginImpl.name)                                                                        \
    CUCIM_PLUGIN_IMPL_WITH_INIT(impl, __VA_ARGS__)


#endif // CUCIM_PLUGIN_UTIL_H
