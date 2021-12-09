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

if (NOT TARGET deps::concurrentqueue)
    FetchContent_Declare(
            deps-concurrentqueue
            GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
            GIT_TAG v1.0.3
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-concurrentqueue)
    if (NOT deps-concurrentqueue_POPULATED)
        message(STATUS "Fetching concurrentqueue sources")
        FetchContent_Populate(deps-concurrentqueue)
        message(STATUS "Fetching concurrentqueue sources - done")
    endif ()

    add_subdirectory(${deps-concurrentqueue_SOURCE_DIR} ${deps-concurrentqueue_BINARY_DIR} EXCLUDE_FROM_ALL)

    add_library(deps::concurrentqueue INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::concurrentqueue INTERFACE concurrentqueue)
    set(deps-concurrentqueue_SOURCE_DIR ${deps-concurrentqueue_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-concurrentqueue_SOURCE_DIR)
endif ()