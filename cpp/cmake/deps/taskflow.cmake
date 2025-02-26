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

if (NOT TARGET deps::taskflow)
    FetchContent_Declare(
            deps-taskflow
            GIT_REPOSITORY https://github.com/taskflow/taskflow.git
            GIT_TAG v3.2.0
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching taskflow sources")
    set(TF_BUILD_TESTS OFF)
    set(TF_BUILD_EXAMPLES OFF)

    FetchContent_MakeAvailable(deps-taskflow)
    message(STATUS "Fetching taskflow sources - done")

    add_library(deps::taskflow INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::taskflow INTERFACE Taskflow)
    set(deps-taskflow_SOURCE_DIR ${deps-taskflow_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-taskflow_SOURCE_DIR)
endif ()
