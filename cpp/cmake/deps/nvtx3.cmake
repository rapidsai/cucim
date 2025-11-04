#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::nvtx3)
    FetchContent_Declare(
            deps-nvtx3
            GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
            GIT_TAG v3.1.1
            GIT_SHALLOW TRUE
            SOURCE_SUBDIR c
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching nvtx3 sources")

    # Create shared library
    cucim_set_build_shared_libs(ON) # since nvtx3 is header-only library, this may not needed.

    set(BUILD_TESTS OFF)
    set(BUILD_BENCHMARKS OFF)

    FetchContent_MakeAvailable(deps-nvtx3)
    message(STATUS "Fetching nvtx3 sources - done")

    cucim_restore_build_shared_libs()

    add_library(deps::nvtx3 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::nvtx3 INTERFACE nvtx3-cpp)
    set(deps-nvtx3_SOURCE_DIR ${deps-nvtx3_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-nvtx3_SOURCE_DIR)
endif ()
