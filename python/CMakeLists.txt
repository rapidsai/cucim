#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

################################################################################
# Prerequisite statements
################################################################################

# Set VERSION
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/../VERSION VERSION)
# strip alpha version info
string(REGEX REPLACE "a.*$" "" VERSION ${VERSION})

# Append local cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
project(cucim VERSION ${VERSION} DESCRIPTION "cucim" LANGUAGES CXX)

################################################################################
# Include utilities
################################################################################
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
endif ()
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
endif ()
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endif ()

find_package(CUDAToolkit) # cucim library depends on CUDA::cudart_static
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

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
# Define dependencies
################################################################################
superbuild_depend(fmt)
superbuild_depend(json)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    COMMAND_ERROR_IS_FATAL ANY
    )
find_package(nanobind CONFIG REQUIRED)

################################################################################
# Find cucim package
################################################################################
if (NOT CUCIM_SDK_PATH)
    get_filename_component(CUCIM_SDK_PATH "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
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

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

# Note: On CentOS, if we do not set MAKE_BUILD_RPATH to $ORIGIN, installed library also do not have $ORIGIN in RPATH.
#       The following is same with `set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)`.
set(CMAKE_BUILD_RPATH $ORIGIN)


################################################################################
# Add library: cucim
################################################################################

nanobind_add_module(cucim
    STABLE_ABI
    NB_STATIC
    PROTECT_STACK
    NOMINSIZE
        nanobind/macros.h
        nanobind/compat.h
        nanobind/cucim_py.h
        nanobind/cucim_pydoc.h
        nanobind/cucim_py.cpp
        nanobind/cache/cache_py.h
        nanobind/cache/cache_pydoc.h
        nanobind/cache/cache_py.cpp
        nanobind/cache/image_cache_pydoc.h
        nanobind/io/io_py.h
        nanobind/io/io_py.cpp
        nanobind/io/device_pydoc.h
        nanobind/filesystem/filesystem_py.h
        nanobind/filesystem/filesystem_pydoc.h
        nanobind/filesystem/filesystem_py.cpp
        nanobind/filesystem/cufile_py.h
        nanobind/filesystem/cufile_pydoc.h
        nanobind/filesystem/cufile_py.cpp
        nanobind/memory/memory_py.h
        nanobind/memory/memory_py.cpp
        nanobind/profiler/profiler_py.h
        nanobind/profiler/profiler_pydoc.h
        nanobind/profiler/profiler_py.cpp
        )
target_link_libraries(cucim
    PRIVATE
        cucim::cucim
        deps::fmt
        deps::json
        )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cucim)
set_target_properties(cucim PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
        OUTPUT_NAME cucim/_cucim
        )


################################################################################
# Install
################################################################################
set(INSTALL_TARGETS
        cucim
        )

install(TARGETS cucim
        EXPORT cucim-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT cucim_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT cucim_Runtime
                NAMELINK_COMPONENT cucim_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT cucim_Development
        )

export(PACKAGE cucim)

unset(BUILD_SHARED_LIBS CACHE)
