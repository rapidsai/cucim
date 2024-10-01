#
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

if (NOT TARGET deps::boost-header-only)
    set(Boost_VERSION 1.85.0)
    set(boost_component_list "interprocess" "config" "intrusive" "move" "assert" "static_assert" "container" "core" "date_time" "smart_ptr" "throw_exception" "utility" "type_traits" "numeric/conversion" "mpl" "preprocessor" "container_hash" "integer" "detail" "describe" "mp11")

    if (DEFINED ENV{CONDA_PREFIX})
        # Use boost headers from conda environment
        find_path(boost_INCLUDE_DIR
            NAMES boost/version.hpp
            HINTS $ENV{CONDA_PREFIX}/include
        )

        if (NOT boost_INCLUDE_DIR)
            message(FATAL_ERROR "boost headers not found in conda environment")
        endif()

        # Apply patch for boost-header-only
        message(STATUS "Applying patch for boost-header-only")
        find_package(Git)

        if(Git_FOUND OR GIT_FOUND)
            # Check if the patch is already applied
            file(READ "${boost_INCLUDE_DIR}/boost/interprocess/mem_algo/rbtree_best_fit.hpp" file_content)
            if("${file_content}" MATCHES "cuCIM patch")
                message(STATUS "Patch for boost-header-only is already applied")
            else()
                execute_process(
                    COMMAND bash -c "${GIT_EXECUTABLE} apply --whitespace=nowarn ${CMAKE_CURRENT_LIST_DIR}/boost-header-only.patch"
                    WORKING_DIRECTORY "${boost_INCLUDE_DIR}/.."  # Assuming the patch is relative to the include directory
                    RESULT_VARIABLE exec_result
                    ERROR_VARIABLE exec_error
                    ERROR_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE exec_output
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                if(exec_result EQUAL 0)
                    message(STATUS "Applying patch for boost-header-only - done")
                else()
                    message(STATUS "Applying patch for boost-header-only - failed")
                    message(STATUS "${exec_output}\n${exec_error}")
                endif()
            endif()
        endif()

        add_library(deps::boost-header-only INTERFACE IMPORTED GLOBAL)

        unset(boost_include_string)
        # Create a list of components
        foreach(item IN LISTS boost_component_list)
            set(boost_include_string "${boost_include_string}" "${boost_INCLUDE_DIR}")
        endforeach(item)

        set_target_properties(deps::boost-header-only PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${boost_include_string}"
            INTERFACE_COMPILE_DEFINITIONS
                BOOST_DATE_TIME_NO_LIB=1
        )

        set(boost_INCLUDE_DIR ${boost_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(boost_INCLUDE_DIR)
    else()
        # Fallback to fetching Boost sources
        FetchContent_Declare(
                deps-boost-header-only
                GIT_REPOSITORY https://github.com/boostorg/boost.git
                GIT_TAG boost-${Boost_VERSION}
                GIT_SHALLOW TRUE
        )

        FetchContent_GetProperties(deps-boost-header-only)
        if (NOT deps-boost-header-only_POPULATED)
            message(STATUS "Fetching boost-header-only sources")
            FetchContent_MakeAvailable(deps-boost-header-only)
            message(STATUS "Fetching boost-header-only sources - done")

            message(STATUS "Applying patch for boost-header-only")
            find_package(Git)
            if(Git_FOUND OR GIT_FOUND)
                execute_process(
                    COMMAND bash -c "${GIT_EXECUTABLE} reset HEAD --hard && ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_LIST_DIR}/boost-header-only.patch"
                    WORKING_DIRECTORY "${deps-boost-header-only_SOURCE_DIR}/libs/interprocess"
                    RESULT_VARIABLE exec_result
                    ERROR_VARIABLE exec_error
                    ERROR_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE exec_output
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
                if(exec_result EQUAL 0)
                    message(STATUS "Applying patch for boost-header-only - done")
                else()
                    message(STATUS "Applying patch for boost-header-only - failed")
                    message(FATAL_ERROR "${exec_output}\n${exec_error}")
                endif()
            endif ()
        endif ()

        add_library(deps::boost-header-only INTERFACE IMPORTED GLOBAL)

        unset(boost_include_string)
        # Create a list of components
        foreach(item IN LISTS boost_component_list)
            set(boost_include_string "${boost_include_string}" "${deps-boost-header-only_SOURCE_DIR}/libs/${item}/include")
        endforeach(item)
        # https://www.boost.org/doc/libs/1_75_0/doc/html/interprocess.html#interprocess.intro.introduction_building_interprocess
        set_target_properties(deps::boost-header-only PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${boost_include_string}"
            INTERFACE_COMPILE_DEFINITIONS
                BOOST_DATE_TIME_NO_LIB=1
        )

        set(boost_INCLUDE_DIR ${deps-boost-header-only_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)
        mark_as_advanced(boost_INCLUDE_DIR)
    endif()
endif ()
