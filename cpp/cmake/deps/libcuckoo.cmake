# Apache License, Version 2.0
# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (NOT TARGET deps::libcuckoo)
    find_package(Git REQUIRED)

    FetchContent_Declare(
            deps-libcuckoo
            GIT_REPOSITORY https://github.com/efficient/libcuckoo
            GIT_TAG v0.3
            GIT_SHALLOW TRUE
            PATCH_COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_LIST_DIR}/libcuckoo.patch" || true
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching libcuckoo sources")
    # Create static library
    cucim_set_build_shared_libs(OFF)
    FetchContent_MakeAvailable(deps-libcuckoo)
    message(STATUS "Fetching libcuckoo sources - done")

    # libcuckoo's CMakeLists.txt is not compatible with `add_subdirectory` method (it uses ${CMAKE_SOURCE_DIR} instead of ${CMAKE_CURRENT_SOURCE_DIR})
    # so add include directories explicitly.
    target_include_directories(libcuckoo INTERFACE
        $<BUILD_INTERFACE:${deps-libcuckoo_SOURCE_DIR}>
    )

    cucim_restore_build_shared_libs()

    add_library(deps::libcuckoo INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::libcuckoo INTERFACE libcuckoo)
    set(deps-libcuckoo_SOURCE_DIR ${deps-libcuckoo_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libcuckoo_SOURCE_DIR)
endif ()
