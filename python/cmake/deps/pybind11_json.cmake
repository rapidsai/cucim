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

if (NOT TARGET deps::pybind11_json)
    if (DEFINED ENV{CONDA_PREFIX})
        find_package(pybind11_json REQUIRED)
        if (NOT pybind11_json_FOUND)
            message(FATAL_ERROR "pybind11_json package not found in conda environment")
        endif ()

        add_library(deps::pybind11_json INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::pybind11_json INTERFACE pybind11_json)
        get_target_property(pybind11_json_INCLUDE_DIR pybind11_json INTERFACE_INCLUDE_DIRECTORIES)

        set(pybind11_json_INCLUDE_DIR ${pybind11_json_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(pybind11_json_INCLUDE_DIR)
    else ()
        # Fallback to fetching pybind11_json sources
        FetchContent_Declare(
            deps-pybind11_json
            GIT_REPOSITORY https://github.com/pybind/pybind11_json.git
            GIT_TAG 0.2.14
            GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-pybind11_json)
        if (NOT deps-pybind11_json_POPULATED)
            message(STATUS "Fetching pybind11_json sources")
            FetchContent_Populate(deps-pybind11_json)
            message(STATUS "Fetching pybind11_json sources - done")
        endif ()

        add_subdirectory(${deps-pybind11_json_SOURCE_DIR} ${deps-pybind11_json_BINARY_DIR} EXCLUDE_FROM_ALL)

        add_library(deps::pybind11_json INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::pybind11_json INTERFACE pybind11_json)
        set(pybind11_json_INCLUDE_DIR ${deps-pybind11_json_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)
        mark_as_advanced(pybind11_json_INCLUDE_DIR)
    endif ()
endif ()
