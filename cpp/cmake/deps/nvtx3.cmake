#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
    FetchContent_Declare(
            deps-nvtx3
            GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
            GIT_TAG v3.1.1
            GIT_SHALLOW TRUE
            SOURCE_SUBDIR c
    )
    message(STATUS "Fetching nvtx3 sources")

    # Create shared library
    cucim_set_build_shared_libs(ON) # since nvtx3 is header-only library, this may not needed.

    set(BUILD_TESTS OFF)
    set(BUILD_BENCHMARKS OFF)

    FetchContent_MakeAvailable(deps-nvtx3)
    message(STATUS "Fetching nvtx3 sources - done")

    cucim_restore_build_shared_libs()

    add_library(deps::nvtx3 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::nvtx3 INTERFACE nvtx3-cpp)
    set(deps-nvtx3_SOURCE_DIR ${deps-nvtx3_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-nvtx3_SOURCE_DIR)
endif ()
