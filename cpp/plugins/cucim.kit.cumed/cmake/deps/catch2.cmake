#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
    FetchContent_Declare(
            deps-catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.4.0
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching catch2 sources")
    FetchContent_MakeAvailable(deps-catch2)
    message(STATUS "Fetching catch2 sources - done")

    # Include Append catch2's cmake module path so that we can use `include(Catch)`.
    # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#catchcmake-and-catchaddtestscmake
    list(APPEND CMAKE_MODULE_PATH "${deps-catch2_SOURCE_DIR}/extras")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)

    add_library(deps::catch2 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::catch2 INTERFACE Catch2::Catch2)
    set(deps-catch2_SOURCE_DIR ${deps-catch2_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-catch2_SOURCE_DIR)
endif ()
