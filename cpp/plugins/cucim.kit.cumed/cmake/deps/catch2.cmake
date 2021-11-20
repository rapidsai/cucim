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

if (NOT TARGET deps::catch2)
    FetchContent_Declare(
            deps-catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v2.13.1
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-catch2)
    if (NOT deps-catch2_POPULATED)
        message(STATUS "Fetching catch2 sources")
        FetchContent_Populate(deps-catch2)
        message(STATUS "Fetching catch2 sources - done")
    endif ()

    add_subdirectory(${deps-catch2_SOURCE_DIR} ${deps-catch2_BINARY_DIR} EXCLUDE_FROM_ALL)

    # Include Append catch2's cmake module path so that we can use `include(ParseAndAddCatchTests)`.
    list(APPEND CMAKE_MODULE_PATH "${deps-catch2_SOURCE_DIR}/contrib")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)

    add_library(deps::catch2 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::catch2 INTERFACE Catch2::Catch2)
    set(deps-catch2_SOURCE_DIR ${deps-catch2_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-catch2_SOURCE_DIR)
endif ()
