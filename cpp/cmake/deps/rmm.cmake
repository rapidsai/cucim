#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# Note: importing rmm is tricky as it depends on googletest/thrust/spdlog and can conflicts with target names in
#       the original project.
#       There is a suggestion in CMake but it seems that it takes time to resolve the issue.
#       - Namespace support for target names in nested projects: https://gitlab.kitware.com/cmake/cmake/-/issues/16414

if (NOT TARGET deps::rmm)
    FetchContent_Declare(
            deps-rmm
            GIT_REPOSITORY https://github.com/rapidsai/rmm.git
            GIT_TAG branch-0.17
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-rmm)
    if (NOT deps-rmm_POPULATED)
        message(STATUS "Fetching rmm sources")
        FetchContent_Populate(deps-rmm)
        message(STATUS "Fetching rmm sources - done")
    endif ()

    # Create shared library
    cucim_set_build_shared_libs(ON) # Since rmm doesn't use BUILD_SHARED_LIBS, it always build shared library

    add_subdirectory(${deps-rmm_SOURCE_DIR} ${deps-rmm_BINARY_DIR} EXCLUDE_FROM_ALL)
    cucim_restore_build_shared_libs()

    add_library(deps::rmm INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::rmm INTERFACE rmm)
    set(deps-rmm_SOURCE_DIR ${deps-rmm_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-rmm_SOURCE_DIR)
endif ()
