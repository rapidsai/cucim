#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::googlebenchmark)
    FetchContent_Declare(
            deps-googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.5.1
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching googlebenchmark sources")

    # Create static library
    cucim_set_build_shared_libs(OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    FetchContent_MakeAvailable(deps-googlebenchmark)
    message(STATUS "Fetching googlebenchmark sources - done")

    cucim_restore_build_shared_libs()

    add_library(deps::googlebenchmark INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::googlebenchmark INTERFACE benchmark::benchmark)
    set(deps-googlebenchmark_SOURCE_DIR ${deps-googlebenchmark_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-googlebenchmark_SOURCE_DIR)
endif ()
