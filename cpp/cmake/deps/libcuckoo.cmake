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
    FetchContent_Declare(
            deps-libcuckoo
            GIT_REPOSITORY https://github.com/efficient/libcuckoo
            GIT_TAG v0.3
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-libcuckoo)
    if (NOT deps-libcuckoo_POPULATED)
        message(STATUS "Fetching libcuckoo sources")
        FetchContent_Populate(deps-libcuckoo)
        message(STATUS "Fetching libcuckoo sources - done")

        message(STATUS "Applying patch for libcuckoo")
        find_package(Git)
        if(Git_FOUND OR GIT_FOUND)
            execute_process(
                COMMAND bash -c "${GIT_EXECUTABLE} reset HEAD --hard && ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_LIST_DIR}/libcuckoo.patch"
                WORKING_DIRECTORY "${deps-libcuckoo_SOURCE_DIR}"
                RESULT_VARIABLE exec_result
                ERROR_VARIABLE exec_error
                ERROR_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE exec_output
                OUTPUT_STRIP_TRAILING_WHITESPACE
                )
            if(exec_result EQUAL 0)
                message(STATUS "Applying patch for libcuckoo - done")
            else()
                message(STATUS "Applying patch for libcuckoo - failed")
                message(FATAL_ERROR "${exec_output}\n${exec_error}")
            endif()
        endif ()
    endif ()

    # Create static library
    cucim_set_build_shared_libs(OFF)
    add_subdirectory(${deps-libcuckoo_SOURCE_DIR} ${deps-libcuckoo_BINARY_DIR} EXCLUDE_FROM_ALL)
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