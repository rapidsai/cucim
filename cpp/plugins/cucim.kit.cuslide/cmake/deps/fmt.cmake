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

if (NOT TARGET deps::fmt)
    if (DEFINED ENV{CONDA_PREFIX})
        find_package(fmt REQUIRED)
        if (NOT fmt_FOUND)
            message(FATAL_ERROR "fmt package not found in conda environment")
        endif ()

        add_library(deps::fmt INTERFACE IMPORTED GLOBAL)
        get_target_property(fmt_INCLUDE_DIR fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(deps::fmt INTERFACE
            "$<BUILD_INTERFACE:${fmt_INCLUDE_DIR}>"
            "$<INSTALL_INTERFACE:include/cucim/3rdparty>"
        )
        target_compile_definitions(deps::fmt INTERFACE
            FMT_HEADER_ONLY
        )

        set(fmt_INCLUDE_DIR ${fmt_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(fmt_INCLUDE_DIR)
    else ()
        # Fallback to fetching fmt sources
        FetchContent_Declare(
                deps-fmt
                GIT_REPOSITORY https://github.com/fmtlib/fmt.git
                GIT_TAG 11.0.1
                GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-fmt)
        if (NOT deps-fmt_POPULATED)
            message(STATUS "Fetching fmt sources")
            # TODO: use FetchContent_MakeAvailable (with EXCLUDE_FROM_ALL option in FetchContent_Declare) when CMake 3.30 is minimum required
            #       (https://cmake.org/cmake/help/latest/policy/CMP0169.html#policy:CMP0169)
            FetchContent_Populate(deps-fmt)
            message(STATUS "Fetching fmt sources - done")
        endif ()

        # Create static library
        cucim_set_build_shared_libs(OFF)
        add_subdirectory(${deps-fmt_SOURCE_DIR} ${deps-fmt_BINARY_DIR} EXCLUDE_FROM_ALL)
        cucim_restore_build_shared_libs()

        add_library(deps::fmt INTERFACE IMPORTED GLOBAL)
        get_target_property(fmt_INCLUDE_DIR fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(deps::fmt INTERFACE
            "$<BUILD_INTERFACE:${fmt_INCLUDE_DIR}>"
            "$<INSTALL_INTERFACE:include/cucim/3rdparty>"
        )
        target_compile_definitions(deps::fmt INTERFACE
            FMT_HEADER_ONLY
        )

        set(fmt_INCLUDE_DIR ${deps-fmt_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)
        mark_as_advanced(fmt_INCLUDE_DIR)
    endif ()
endif ()
