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

#ifndef CUCIM_DEFINES_H
#define CUCIM_DEFINES_H

#include "cucim/macros/api_header.h"


/*******************************************************************************
 Platform-related definitions
*******************************************************************************/


/*******************************************************************************
 Memory-related definitions
*******************************************************************************/

#define CUCIM_ALIGN_AS(T) alignas(T)


/*******************************************************************************
 Debug-related definitions
*******************************************************************************/

#if CUCIM_PLATFORM_LINUX
#    include <signal.h>
#    define CUCIM_BREAK_POINT() ::raise(SIGTRAP)
#elif CUCIM_PLATFORM_WINDOWS
#    define CUCIM_BREAK_POINT() ::__debugbreak()
#else
#    error "This platform is not supported!"
#endif

#define CUCIM_CHECK_ENABLED 1
#define CUCIM_CHECK(cond, ...) ((void)0)


#if CUCIM_DEBUG
#    define CUCIM_ASSERT_ENABLED 1
#    define CUCIM_ASSERT(cond, ...) CUCIM_CHECK(cond, ##__VA_ARGS__)
#else
#    define CUCIM_ASSERT_ENABLED 0
#    define CUCIM_ASSERT(cond, ...) (void)0;
#endif


#include <cstdio>
#include <cinttypes>
#define CUCIM_LOG_VERBOSE(fmt, ...) ::fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define CUCIM_LOG_INFO(fmt, ...) ::fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define CUCIM_LOG_WARN(fmt, ...) ::fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define CUCIM_LOG_ERROR(fmt, ...) ::fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define CUCIM_LOG_FATAL(fmt, ...) ::fprintf(stderr, fmt "\n", ##__VA_ARGS__)

#include <exception>
#define CUCIM_ERROR(fmt, ...)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        ::fprintf(stderr, fmt "\n", ##__VA_ARGS__);                                                                    \
        throw std::runtime_error("Error!");                                                                            \
    } while (0)

// Check float type size
#include <climits>
static_assert(sizeof(float) * CHAR_BIT == 32, "float data type is not 32 bits!");

#endif // CUCIM_DEFINES_H
