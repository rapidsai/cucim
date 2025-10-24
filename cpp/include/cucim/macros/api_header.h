/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_API_H
#define CUCIM_API_H

#if defined(__linux__)
#    define CUCIM_PLATFORM_LINUX 1
#    define CUCIM_PLATFORM_WINDOWS 0
#elif _WIN32
#    define CUCIM_PLATFORM_LINUX 0
#    define CUCIM_PLATFORM_WINDOWS 1
#else
#    error "This platform is not supported!"
#endif

#if CUCIM_PLATFORM_WINDOWS
#    define CUCIM_ABI __cdecl
#else
#    define CUCIM_ABI
#endif

//#ifdef CARB_EXPORTS
//#    ifdef __cplusplus
//#        define CARB_EXPORT_C extern "C"
//#    else
//#        define CARB_EXPORT_C
//#    endif
//
//#    undef CARB_EXPORT
//#    define CARB_EXPORT CARB_EXPORT_C CARB_DECLSPEC(dllexport) CARB_ATTRIBUTE(visibility("default"))
//#else
//#    undef CARB_EXPORT
//#    define CARB_EXPORT extern "C"
//#endif

#ifndef EXPORT_VISIBLE
#   define EXPORT_VISIBLE __attribute__((visibility("default")))
#endif
#ifndef EXPORT_HIDDEN
#   define EXPORT_HIDDEN __attribute__((visibility("hidden")))
#endif

#ifdef CUCIM_STATIC_DEFINE
#    define CUCIM_API
#    define CUCIM_NO_EXPORT
#else
#    ifdef __cplusplus
#        define CUCIM_EXPORT_C extern "C"
#    else
#        define CUCIM_EXPORT_C
#    endif
#    ifdef CUCIM_EXPORTS
#        undef CUCIM_API
#        define CUCIM_API CUCIM_EXPORT_C EXPORT_VISIBLE
#    else
#        undef CUCIM_API
#        define CUCIM_API CUCIM_EXPORT_C
#    endif
#    ifndef CUCIM_NO_EXPORT
#        define CUCIM_NO_EXPORT EXPORT_HIDDEN
#    endif
#endif

#ifndef CUCIM_DEPRECATED
#    define CUCIM_DEPRECATED __attribute__((__deprecated__))
#endif

#ifndef CUCIM_DEPRECATED_EXPORT
#    define CUCIM_DEPRECATED_EXPORT CUCIM_API CUCIM_DEPRECATED
#endif

#ifndef CUCIM_DEPRECATED_NO_EXPORT
#    define CUCIM_DEPRECATED_NO_EXPORT CUCIM_NO_EXPORT CUCIM_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#    ifndef CUCIM_NO_DEPRECATED
#        define CUCIM_NO_DEPRECATED
#    endif
#endif

#endif /* CUCIM_API_H */
