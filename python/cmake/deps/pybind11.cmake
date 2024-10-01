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

if (NOT TARGET deps::pybind11)
    if (DEFINED ENV{CONDA_PREFIX})
        find_package(pybind11 REQUIRED)
        if (NOT pybind11_FOUND)
            message(FATAL_ERROR "pybind11 package not found in conda environment")
        endif ()

        add_library(deps::pybind11 INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::pybind11 INTERFACE pybind11::module)
        get_target_property(pybind11_INCLUDE_DIR pybind11::pybind11_headers INTERFACE_INCLUDE_DIRECTORIES)

        set(pybind11_INCLUDE_DIR ${pybind11_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(pybind11_INCLUDE_DIR)
    else ()
        # Fallback to fetching fmt sources
        FetchContent_Declare(
                deps-pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG v2.13.1
                GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-pybind11)
        if (NOT deps-pybind11_POPULATED)
            message(STATUS "Fetching pybind11 sources")
            FetchContent_Populate(deps-pybind11)
            message(STATUS "Fetching pybind11 sources - done")
        endif ()

        add_subdirectory(${deps-pybind11_SOURCE_DIR} ${deps-pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)

        add_library(deps::pybind11 INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::pybind11 INTERFACE pybind11::module)
        set(deps-pybind11_SOURCE_DIR ${deps-pybind11_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)
        mark_as_advanced(deps-pybind11_SOURCE_DIR)
    endif ()
endif ()
