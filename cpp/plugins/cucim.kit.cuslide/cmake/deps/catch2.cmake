#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::catch2)
    FetchContent_Declare(
            deps-catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.4.0
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching catch2 sources")
    FetchContent_MakeAvailable(deps-catch2)
    message(STATUS "Fetching catch2 sources - done")

    # Include Append catch2's cmake module path so that we can use `include(Catch)`.
    # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#catchcmake-and-catchaddtestscmake
    list(APPEND CMAKE_MODULE_PATH "${deps-catch2_SOURCE_DIR}/extras")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)

    add_library(deps::catch2 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::catch2 INTERFACE Catch2::Catch2)
    set(deps-catch2_SOURCE_DIR ${deps-catch2_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-catch2_SOURCE_DIR)
endif ()
