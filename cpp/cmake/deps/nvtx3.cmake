#
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

if (NOT TARGET deps::nvtx3)
    if (DEFINED ENV{CONDA_PREFIX})
        # Use nvtx3 headers from conda environment
        find_path(nvtx3_INCLUDE_DIR
            NAMES nvtx3/nvtx3.hpp
            HINTS $ENV{CONDA_PREFIX}/include
        )

        if (NOT nvtx3_INCLUDE_DIR)
            message(FATAL_ERROR "nvtx3 headers not found in conda environment")
        endif ()

        add_library(deps::nvtx3 INTERFACE IMPORTED GLOBAL)

        set_target_properties(deps::nvtx3 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${nvtx3_INCLUDE_DIR}"
        )
        target_link_libraries(deps::nvtx3 INTERFACE ${CMAKE_DL_LIBS})

        set(nvtx3_INCLUDE_DIR ${nvtx3_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(nvtx3_INCLUDE_DIR)
    else ()
        # Fallback to fetching nvtx3 sources
        FetchContent_Declare(
                deps-nvtx3
                GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
                GIT_TAG v3.1.0
                GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-nvtx3)
        if (NOT deps-nvtx3_POPULATED)
            message(STATUS "Fetching nvtx3 sources")
            FetchContent_Populate(deps-nvtx3)
            message(STATUS "Fetching nvtx3 sources - done")
        endif ()

        # Create shared library
        cucim_set_build_shared_libs(ON) # since nvtx3 is header-only library, this may not needed.

        set(BUILD_TESTS OFF)
        set(BUILD_BENCHMARKS OFF)

        add_subdirectory(${deps-nvtx3_SOURCE_DIR}/c ${deps-nvtx3_BINARY_DIR} EXCLUDE_FROM_ALL)
        cucim_restore_build_shared_libs()

        add_library(deps::nvtx3 INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::nvtx3 INTERFACE nvtx3-cpp)
        set(nvtx3_INCLUDE_DIR ${deps-nvtx3_SOURCE_DIR}/c/include CACHE INTERNAL "" FORCE)
        mark_as_advanced(nvtx3_INCLUDE_DIR)
    endif ()
endif ()
