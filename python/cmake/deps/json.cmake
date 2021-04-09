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

if (NOT TARGET deps::json)
    FetchContent_Declare(
            deps-json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.9.1
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-json)
    if (NOT deps-json_POPULATED)
        message(STATUS "Fetching json sources")
        FetchContent_Populate(deps-json)
        message(STATUS "Fetching json sources - done")
    endif ()

    # Typically you don't care so much for a third party library's tests to be
    # run from your own project's code.
    set(JSON_BuildTests OFF CACHE INTERNAL "")

    add_subdirectory(${deps-json_SOURCE_DIR} ${deps-json_BINARY_DIR} EXCLUDE_FROM_ALL)

    add_library(deps::json INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::json INTERFACE nlohmann_json::nlohmann_json)
    set(deps-json_SOURCE_DIR ${deps-json_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-json_SOURCE_DIR)
endif ()