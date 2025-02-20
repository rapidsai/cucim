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

if (NOT TARGET deps::googlebenchmark)
    FetchContent_Declare(
            deps-googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.5.1
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching googlebenchmark sources")
    cucim_set_build_shared_libs(OFF)
    # Create static library
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    FetchContent_MakeAvailable(deps-googlebenchmark)
    message(STATUS "Fetching googlebenchmark sources - done")

    cucim_restore_build_shared_libs()

    add_library(deps::googlebenchmark INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::googlebenchmark INTERFACE benchmark::benchmark)
    set(deps-googlebenchmark_SOURCE_DIR ${deps-googlebenchmark_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-googlebenchmark_SOURCE_DIR)
endif ()
