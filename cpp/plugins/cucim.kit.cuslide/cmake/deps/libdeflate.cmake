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

if (NOT TARGET deps::libdeflate)
    FetchContent_Declare(
            deps-libdeflate
            GIT_REPOSITORY https://github.com/ebiggers/libdeflate.git
            GIT_TAG v1.7
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-libdeflate)
    if (NOT deps-libdeflate_POPULATED)
        message(STATUS "Fetching libdeflate sources")
        FetchContent_Populate(deps-libdeflate)
        message(STATUS "Fetching libdeflate sources - done")
    endif ()

    if (deps-libdeflate_POPULATED AND NOT EXISTS "${deps-libdeflate_BINARY_DIR}/install")
        include(ProcessorCount)
        ProcessorCount(PROCESSOR_COUNT)

        # /opt/rh/devtoolset-9/root/usr/libexec/gcc/x86_64-redhat-linux/9/ld: _deps/deps-libdeflate-build/install/lib/libdeflate.a(deflate_decompress.o): relocation R_X86_64_32 against `.rodata' can not be used when making a shared object; recompile with -fPIC
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(LIBDEFLATE_CMAKE_ARGS "-e CFLAGS='-O0 -g3 -fPIC'")
        else()
            set(LIBDEFLATE_CMAKE_ARGS "-e CFLAGS='-fPIC'")
        endif()

        execute_process(COMMAND /bin/bash -c "make -e PREFIX=${deps-libdeflate_BINARY_DIR}/install ${LIBDEFLATE_CMAKE_ARGS} install -j${PROCESSOR_COUNT}"
                        WORKING_DIRECTORY ${deps-libdeflate_SOURCE_DIR}
                        COMMAND_ECHO STDOUT
                        RESULT_VARIABLE libdeflate_BUILD_RESULT)
        if(NOT libdeflate_BUILD_RESULT EQUAL "0")
            message(FATAL_ERROR "libdeflate library build failed with ${libdeflate_BUILD_RESULT}, please checkout the configurations")
        endif()
    endif()

    add_library(deps::libdeflate INTERFACE IMPORTED GLOBAL)

    set_target_properties(deps::libdeflate PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${deps-libdeflate_BINARY_DIR}/install/include"
        INTERFACE_LINK_LIBRARIES "${deps-libdeflate_BINARY_DIR}/install/lib/libdeflate.a"
    )

    set(deps-libdeflate_SOURCE_DIR ${deps-libdeflate_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libdeflate_SOURCE_DIR)
endif ()
