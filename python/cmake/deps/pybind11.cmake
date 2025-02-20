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

if (NOT TARGET deps::pybind11)
    find_package(Git REQUIRED)

    FetchContent_Declare(
            deps-pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
            GIT_SHALLOW TRUE
            PATCH_COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_LIST_DIR}/pybind11_pr4857_4877.patch"
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching pybind11 sources")
    FetchContent_MakeAvailable(deps-pybind11)
    message(STATUS "Fetching pybind11 sources - done")

    # https://pybind11.readthedocs.io/en/stable/compiling.html#configuration-variables
    add_library(deps::pybind11 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::pybind11 INTERFACE pybind11::module)
    set(deps-pybind11_SOURCE_DIR ${deps-pybind11_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-pybind11_SOURCE_DIR)
endif ()
