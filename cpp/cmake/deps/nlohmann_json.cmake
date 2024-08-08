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

if (NOT TARGET deps::nlohmann_json)
    if (DEFINED ENV{CONDA_PREFIX})
        find_package(nlohmann_json REQUIRED)
        if (NOT nlohmann_json_FOUND)
            message(FATAL_ERROR "nlohmann_json package not found in conda environment")
        endif()

        add_library(deps::nlohmann_json ALIAS nlohmann_json::nlohmann_json)

        get_target_property(nlohmann_json_INCLUDE_DIR nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)

        set(nlohmann_json_INCLUDE_DIR ${nlohmann_json_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(nlohmann_json_INCLUDE_DIR)
    else()
        # Fallback to fetching nlohmann_json sources
        FetchContent_Declare(
            deps-nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.11.3
            GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-nlohmann_json)
        if (NOT deps-nlohmann_json_POPULATED)
            message(STATUS "Fetching nlohmann_json sources")
            FetchContent_Populate(deps-nlohmann_json)
            message(STATUS "Fetching json sources - done")
        endif ()

        # Typically you don't care so much for a third party library's tests to be
        # run from your own project's code.
        set(nlohmann_json_BuildTests OFF CACHE INTERNAL "")

        add_subdirectory(${deps-nlohmann_json_SOURCE_DIR} ${deps-nlohmann_json_BINARY_DIR} EXCLUDE_FROM_ALL)

        add_library(deps::nlohmann_json INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::nlohmann_json INTERFACE nlohmann_json::nlohmann_json)
        set(nlohmann_json_INCLUDE_DIR ${deps-nlohmann_json_SOURCE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(nlohmann_json_INCLUDE_DIR)
    endif()
endif ()
