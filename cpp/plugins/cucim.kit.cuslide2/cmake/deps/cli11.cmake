#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::cli11)
    FetchContent_Declare(
            deps-cli11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG v2.5.0
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching cli11 sources")
    set(CLI11_BUILD_DOCS OFF)
    set(CLI11_BUILD_EXAMPLES OFF)
    set(CLI11_BUILD_TESTS OFF)
    FetchContent_MakeAvailable(deps-cli11)
    message(STATUS "Fetching cli11 sources - done")

    add_library(deps::cli11 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::cli11 INTERFACE CLI11::CLI11)
    set(deps-cli11_SOURCE_DIR ${deps-cli11_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-cli11_SOURCE_DIR)
endif ()

# Note that library had a failure with nvcc compiler and gcc 9.x headers
#  ...c++/9/tuple(553): error: pack "_UElements" does not have the same number of elements as "_Elements"
#            __and_<is_nothrow_assignable<_Elements&, _UElements>...>::value;
# Not using nvcc for main code that uses cli11 solved the issue.
