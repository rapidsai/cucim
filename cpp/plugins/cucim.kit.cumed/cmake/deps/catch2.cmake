#
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if (NOT TARGET deps::catch2)
    # Using Catch2 from Conda package has some issues with the CMake integration.
    # - [Add C++17 support that "just works" via package managers · Issue #2462 · catchorg/Catch2](https://github.com/catchorg/Catch2/issues/2462)
    # - https://stackoverflow.com/a/70320798/16361228

    # if (DEFINED ENV{CONDA_PREFIX})
    #     find_package(Catch2 3 REQUIRED)
    #     if (NOT Catch2_FOUND)
    #         message(FATAL_ERROR "Catch2 package not found in conda environment")
    #     endif ()

    #     get_target_property(catch2_INCLUDE_DIR Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
    #     add_library(deps::catch2 ALIAS Catch2::Catch2)

    #     # Include Append catch2's cmake module path so that we can use `include(Catch)`.
    #     # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#catchcmake-and-catchaddtestscmake
    #     list(APPEND CMAKE_MODULE_PATH "${catch2_INCLUDE_DIR}/../lib/cmake/Catch2")
    #     set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)

    #     set(catch2_INCLUDE_DIR ${catch2_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
    #     mark_as_advanced(catch2_INCLUDE_DIR)
    # else ()
        # Fallback to fetching fmt sources
        FetchContent_Declare(
                deps-catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG v3.6.0
                GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-catch2)
        if (NOT deps-catch2_POPULATED)
            message(STATUS "Fetching catch2 sources")
            FetchContent_Populate(deps-catch2)
            message(STATUS "Fetching catch2 sources - done")
        endif ()

        add_subdirectory(${deps-catch2_SOURCE_DIR} ${deps-catch2_BINARY_DIR} EXCLUDE_FROM_ALL)

        # Include Append catch2's cmake module path so that we can use `include(Catch)`.
        # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#catchcmake-and-catchaddtestscmake
        list(APPEND CMAKE_MODULE_PATH "${deps-catch2_SOURCE_DIR}/extras")
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)

        add_library(deps::catch2 INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::catch2 INTERFACE Catch2::Catch2)
        set(catch2_INCLUDE_DIR ${deps-catch2_SOURCE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(catch2_INCLUDE_DIR)
    # endif ()
endif ()
