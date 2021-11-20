# Apache License, Version 2.0
# Copyright 2020-2021 NVIDIA Corporation
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

if (NOT TARGET deps::libjpeg-turbo)
#    add_library(deps::libjpeg-turbo SHARED IMPORTED GLOBAL)
#
#    set_target_properties(deps::libjpeg-turbo PROPERTIES
#        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libjpeg-turbo.so"
#        INTERFACE_INCLUDE_DIRECTORIES "/usr/include/x86_64-linux-gnu"
#    )

    FetchContent_Declare(
            deps-libjpeg-turbo
            GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
            GIT_TAG 2.0.6
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-libjpeg-turbo)
    if (NOT deps-libjpeg-turbo_POPULATED)
        message(STATUS "Fetching libjpeg-turbo sources")
        FetchContent_Populate(deps-libjpeg-turbo)
        message(STATUS "Fetching libjpeg-turbo sources - done")
    endif ()

    # Set policies for libjpeg-turbo
    set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_LIST_DIR}/libjpeg-turbo-policies-fix.cmake")

    # Create static library
    cucim_set_build_shared_libs(OFF)

    # Tell CMake where to find the compiler by setting either the environment
    # variable "ASM_NASM" or the CMake cache entry CMAKE_ASM_NASM_COMPILER to the
    # full path to the compiler, or to the compiler name if it is in the PATH.
    # yasm is available through `sudo apt-get install yasm` on Debian Linux.
    # See _deps/deps-libjpeg-turbo-src/simd/CMakeLists.txt:25.
    set(CMAKE_ASM_NASM_COMPILER yasm)
    set(REQUIRE_SIMD 1) # CMP0077

    add_subdirectory(${deps-libjpeg-turbo_SOURCE_DIR} ${deps-libjpeg-turbo_BINARY_DIR} EXCLUDE_FROM_ALL)

    # Disable visibility to not expose unnecessary symbols
    set_target_properties(turbojpeg-static
    PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)

    # Set PIC to prevent the following error message
    # : /usr/bin/ld: lib/libturbojpeg.a(turbojpeg.c.o): relocation R_X86_64_TPOFF32 against `errStr' can not be used when making a shared object; recompile with -fPIC
    #   /usr/bin/ld: final link failed: Nonrepresentable section on output
    set_target_properties(turbojpeg-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cucim_restore_build_shared_libs()

    add_library(deps::libjpeg-turbo INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::libjpeg-turbo INTERFACE turbojpeg-static)
    target_include_directories(deps::libjpeg-turbo
        INTERFACE
            # turbojpeg.h is not included in 'turbojpeg-static' so manually include
            ${deps-libjpeg-turbo_SOURCE_DIR}
    )

    set(deps-libjpeg-turbo_SOURCE_DIR ${deps-libjpeg-turbo_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libjpeg-turbo_SOURCE_DIR)
    set(deps-libjpeg-turbo_BINARY_DIR ${deps-libjpeg-turbo_BINARY_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libjpeg-turbo_BINARY_DIR)
endif ()