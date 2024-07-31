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

if (NOT TARGET deps::cli11)
    if (DEFINED ENV{CONDA_PREFIX})
        # Use cli11 headers from conda environment
        find_path(cli11_INCLUDE_DIR
            NAMES CLI/CLI.hpp
            HINTS $ENV{CONDA_PREFIX}/include
        )

        if (NOT cli11_INCLUDE_DIR)
            message(FATAL_ERROR "cli11 headers not found in conda environment")
        endif ()

        add_library(deps::cli11 INTERFACE IMPORTED GLOBAL)
        target_include_directories(deps::cli11 INTERFACE ${cli11_INCLUDE_DIR})

        set(cli11_INCLUDE_DIR ${cli11_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(cli11_INCLUDE_DIR)
    else ()
        # Fallback to fetching fmt sources
        FetchContent_Declare(
                deps-cli11
                GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
                GIT_TAG v2.4.1
                GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-cli11)
        if (NOT deps-cli11_POPULATED)
            message(STATUS "Fetching cli11 sources")
            FetchContent_Populate(deps-cli11)
            message(STATUS "Fetching cli11 sources - done")
        endif ()

        add_subdirectory(${deps-cli11_SOURCE_DIR} ${deps-cli11_BINARY_DIR} EXCLUDE_FROM_ALL)

        add_library(deps::cli11 INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::cli11 INTERFACE CLI11::CLI11)
        set(cli11_INCLUDE_DIR ${deps-cli11_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)
        mark_as_advanced(cli11_INCLUDE_DIR)
    endif ()
endif ()
