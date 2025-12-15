# Apache License, Version 2.0
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

################################################################################
# Prerequisite statements
################################################################################

# Set VERSION
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/../../../VERSION VERSION)
# strip alpha version info
string(REGEX REPLACE "a.*$" "" VERSION ${VERSION})

# Reuse cmake modules from cuslide plugin (for SuperBuildUtils.cmake and shared deps)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cucim.kit.cuslide/cmake/modules")
# Include central CuCIM cmake modules (for CuCIMUtils.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cpp/cmake/modules")

project(cuslide2 VERSION ${VERSION} DESCRIPTION "cuslide2" LANGUAGES C CXX)
set(CUCIM_PLUGIN_NAME "cucim.kit.cuslide2")

################################################################################
# Include utilities
################################################################################
# Set additional deps directory for cuslide2-specific dependencies (e.g., nvimgcodec)
set(SUPERBUILD_ADDITIONAL_DEPS_DIRS "${CMAKE_CURRENT_LIST_DIR}/cmake/deps")
include(SuperBuildUtils)
include(CuCIMUtils)

################################################################################
# Basic setup
################################################################################

# Set default build type
set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Set default output directories
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
endif()
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
endif()
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endif()

# Find CUDAToolkit as rmm depends on it
find_package(CUDAToolkit REQUIRED)
# For Threads::Threads
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Include CUDA headers explicitly for VSCode intelli-sense
include_directories(AFTER SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Disable visibility to not expose unnecessary symbols
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Set RPATH
if (NOT APPLE)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

# Set Installation setup
if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/install) # CACHE PATH "install here" FORCE)
endif ()

include(GNUInstallDirs)
# Force to set CMAKE_INSTALL_LIBDIR to lib as the library can be built with Cent OS ('lib64' is set) and
# /usr/local/lib64 or /usr/local/lib is not part of ld.so.conf* (`cat /etc/ld.so.conf.d/* | grep lib64`)
# https://gitlab.kitware.com/cmake/cmake/-/issues/20565
set(CMAKE_INSTALL_LIBDIR lib)

include(ExternalProject)

################################################################################
# Options
################################################################################

# Setup CXX11 ABI
# : Adds CXX11 ABI definition to the compiler command line for targets in the current directory,
#   whether added before or after this command is invoked, and for the ones in sub-directories added after.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0) # TODO: create two library, one with CXX11 ABI and one without it.

################################################################################
# Define dependencies
################################################################################
superbuild_depend(fmt)
superbuild_depend(catch2)
superbuild_depend(openslide)
superbuild_depend(googletest)
superbuild_depend(googlebenchmark)
superbuild_depend(cli11)
superbuild_depend(pugixml)
superbuild_depend(json)
superbuild_depend(nvimgcodec)

################################################################################
# Find cucim package
################################################################################
if (NOT CUCIM_SDK_PATH)
    get_filename_component(CUCIM_SDK_PATH "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
    message("CUCIM_SDK_PATH is not set. Using '${CUCIM_SDK_PATH}'")
else()
    message("CUCIM_SDK_PATH is set to ${CUCIM_SDK_PATH}")
endif()

find_package(cucim CONFIG REQUIRED
    HINTS ${CUCIM_SDK_PATH}/install/${CMAKE_INSTALL_LIBDIR}/cmake/cucim
          $ENV{PREFIX}/include/cmake/cucim # In case conda build is used
    )


################################################################################
# Define compile options
################################################################################

################################################################################
# Add library: cucim
################################################################################

option(CUSLIDE2_INFRA_ONLY "Infrastructure-only mode: validate dependencies/configure only; do not build cuslide2 sources" ON)

if(CUSLIDE2_INFRA_ONLY)
    message(STATUS "=============================================================")
    message(STATUS "cuslide2 infrastructure-only configuration")
    message(STATUS "")
    message(STATUS "This mode is intended for infra-only PRs (deps/conda/CMake).")
    message(STATUS "No cuslide2 plugin sources will be compiled in this configuration.")
    message(STATUS "")
    message(STATUS "Dependencies configured:")
    message(STATUS "  ✓ fmt, json, pugixml, openslide, test frameworks")
    if(TARGET deps::nvimgcodec)
        message(STATUS "  ✓ nvImageCodec (deps::nvimgcodec)")
    else()
        message(STATUS "  ⚠ nvImageCodec not found (deps::nvimgcodec missing) - OK for infra-only mode")
    endif()
    message(STATUS "=============================================================")

    # Provide a stub target so that `cmake --build ... --target cucim.kit.cuslide2`
    # doesn't fail with "No rule to make target" in infra-only configurations.
    add_custom_target(${CUCIM_PLUGIN_NAME}
        COMMAND ${CMAKE_COMMAND} -E echo
            "cuslide2 infra-only: no plugin is built in this configuration. Reconfigure with -DCUSLIDE2_INFRA_ONLY=OFF to build the real plugin target."
    )
else()
    message(FATAL_ERROR "CUSLIDE2_INFRA_ONLY=OFF requested, but this PR intentionally removes cuslide2 implementation sources. Use an implementation PR/branch to build the real plugin.")

endif() # CUSLIDE2_INFRA_ONLY

unset(BUILD_SHARED_LIBS CACHE)
