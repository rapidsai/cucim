#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

if (NOT TARGET deps::boost)
    set(Boost_VERSION 1.75.0)
    set(Boost_BUILD_COMPONENTS container)
    set(Boost_BUILD_OPTIONS "threading=multi cxxflags=-fPIC runtime-link=static variant=release link=static address-model=64 --layout=system")
    set(Boost_COMPILE_DEFINITIONS
        BOOST_COROUTINES_NO_DEPRECATION_WARNING=1
        BOOST_ALL_NO_LIB=1
        BOOST_UUID_RANDOM_PROVIDER_FORCE_WINCRYPT=1
        CACHE INTERNAL "Boost compile definitions")

    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME ON)

    foreach(component_name ${Boost_BUILD_COMPONENTS})
	    list(APPEND Boost_BUILD_VARIANTS --with-${component_name})
    endforeach()

    FetchContent_Declare(
            deps-boost
            GIT_REPOSITORY https://github.com/boostorg/boost.git
            GIT_TAG boost-${Boost_VERSION}
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-boost)
    if (NOT deps-boost_POPULATED)
        message(STATUS "Fetching boost sources")
        FetchContent_Populate(deps-boost)
        message(STATUS "Fetching boost sources - done")
    endif ()

    if (deps-boost_POPULATED AND NOT EXISTS "${deps-boost_BINARY_DIR}/install")
        include(ProcessorCount)
        ProcessorCount(PROCESSOR_COUNT)

        execute_process(COMMAND /bin/bash -c "./bootstrap.sh --prefix=${deps-boost_BINARY_DIR}/install && ./b2 install --build-dir=${deps-boost_BINARY_DIR}/build --stagedir=${deps-boost_BINARY_DIR}/stage -j${PROCESSOR_COUNT} ${Boost_BUILD_VARIANTS} ${Boost_BUILD_OPTIONS}"
                        WORKING_DIRECTORY ${deps-boost_SOURCE_DIR}
                        COMMAND_ECHO STDOUT
                        RESULT_VARIABLE Boost_BUILD_RESULT)
        if(NOT Boost_BUILD_RESULT EQUAL "0")
            message(FATAL_ERROR "boost library build failed with ${Boost_BUILD_RESULT}, please checkout the boost module configurations")
        endif()
    endif()

    find_package(Boost 1.75 CONFIG REQUIRED COMPONENTS ${Boost_BUILD_COMPONENTS}
        HINTS ${deps-boost_BINARY_DIR}/install) # /lib/cmake/Boost-${Boost_VERSION}

    message(STATUS "Boost version: ${Boost_VERSION}")

    add_library(deps::boost INTERFACE IMPORTED GLOBAL)

    set_target_properties(deps::boost PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}"
        INTERFACE_COMPILE_DEFINITIONS "${Boost_COMPILE_DEFINITIONS}"
        INTERFACE_LINK_LIBRARIES "${Boost_LIBRARIES}"
    )

    set(deps-boost_SOURCE_DIR ${deps-boost_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-boost_SOURCE_DIR)
endif ()