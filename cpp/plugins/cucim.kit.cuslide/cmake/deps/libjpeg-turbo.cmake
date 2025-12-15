# cmake-format: off
# SPDX-FileCopyrightText: Copyright 2020-2025 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

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
            PATCH_COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_LIST_DIR}/libjpeg-turbo.patch"
            EXCLUDE_FROM_ALL
    )

    # Set policies for libjpeg-turbo
    set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_LIST_DIR}/libjpeg-turbo-policies-fix.cmake")

    # Create static library
    cucim_set_build_shared_libs(OFF)

    # Tell CMake where to find the compiler by setting either the environment
    # variable "ASM_NASM" or the CMake cache entry CMAKE_ASM_NASM_COMPILER to the
    # full path to the compiler, or to the compiler name if it is in the PATH.
    # nasm is available through `sudo apt-get install nasm` on Debian Linux.
    # See _deps/deps-libjpeg-turbo-src/simd/CMakeLists.txt:25.

    # Try to find yasm in conda environment first, then system paths
    if(DEFINED ENV{CONDA_PREFIX})
        find_program(YASM_EXECUTABLE NAMES yasm PATHS $ENV{CONDA_PREFIX}/bin NO_DEFAULT_PATH)
    endif()
    if(NOT YASM_EXECUTABLE)
        find_program(YASM_EXECUTABLE NAMES yasm)
    endif()

    if(YASM_EXECUTABLE)
        set(CMAKE_ASM_NASM_COMPILER ${YASM_EXECUTABLE})
        message(STATUS "Found yasm: ${YASM_EXECUTABLE}")
    else()
        set(CMAKE_ASM_NASM_COMPILER yasm)
        message(WARNING "yasm not found, using 'yasm' and hoping it's in PATH")
    endif()
    set(REQUIRE_SIMD 1) # CMP0077

    message(STATUS "Fetching libjpeg-turbo sources")
    FetchContent_MakeAvailable(deps-libjpeg-turbo)
    message(STATUS "Fetching libjpeg-turbo sources - done")

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
