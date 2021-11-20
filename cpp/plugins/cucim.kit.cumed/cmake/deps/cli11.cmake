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

if (NOT TARGET deps::cli11)
    FetchContent_Declare(
            deps-cli11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG v1.9.1
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-cli11)
    if (NOT deps-cli11_POPULATED)
        message(STATUS "Fetching cli11 sources")
        FetchContent_Populate(deps-cli11)
        message(STATUS "Fetching cli11 sources - done")
    endif ()

    add_subdirectory(${deps-cli11_SOURCE_DIR} ${deps-cli11_BINARY_DIR} EXCLUDE_FROM_ALL)

    add_library(deps::cli11 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::cli11 INTERFACE CLI11::CLI11)
    set(deps-cli11_SOURCE_DIR ${deps-cli11_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-cli11_SOURCE_DIR)
endif ()

# Note that library had a failure with nvcc compiler and gcc 9.x headers
#  ...c++/9/tuple(553): error: pack "_UElements" does not have the same number of elements as "_Elements"
#            __and_<is_nothrow_assignable<_Elements&, _UElements>...>::value;
# Not using nvcc for main code that uses cli11 solved the issue.
