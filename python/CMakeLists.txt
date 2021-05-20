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

# CUDA_STANDARD 17 is supported from CMAKE 3.18
# : https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_STANDARD.html
cmake_minimum_required(VERSION 3.18)

################################################################################
# Prerequisite statements
################################################################################

# Set VERSION
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/../VERSION VERSION)

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

find_package(CUDAToolkit) # cucim library depends on CUDA::cudart
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
# Options
################################################################################

# Setup CXX11 ABI
# : Adds CXX11 ABI definition to the compiler command line for targets in the current directory,
#   whether added before or after this command is invoked, and for the ones in sub-directories added after.
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) # TODO: create two library, one with CXX11 ABI and one without it.

################################################################################
# Define dependencies
################################################################################
superbuild_depend(pybind11)
superbuild_depend(fmt)
superbuild_depend(json)
superbuild_depend(pybind11_json)

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

#get_target_property(TT cucim::cucim INTERFACE_INCLUDE_DIRECTORIES)
#set(PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDE_DIR} /ssd/repo/cucim/install/include)

pybind11_add_module(cucim
    MODULE
        pybind11/macros.h
        pybind11/cucim_py.h
        pybind11/cucim_pydoc.h
        pybind11/cucim_py.cpp
        pybind11/cache/cache_py.h
        pybind11/cache/cache_pydoc.h
        pybind11/cache/cache_py.cpp
        pybind11/cache/image_cache_pydoc.h
        pybind11/cache/image_cache_py.h
        pybind11/cache/image_cache_py.cpp
        pybind11/io/io_py.h
        pybind11/io/io_pydoc.h
        pybind11/io/io_py.cpp
        pybind11/io/device_pydoc.h
        pybind11/io/device_py.cpp
        pybind11/filesystem/filesystem_py.h
        pybind11/filesystem/filesystem_pydoc.h
        pybind11/filesystem/filesystem_py.cpp
        pybind11/filesystem/cufile_py.h
        pybind11/filesystem/cufile_pydoc.h
        pybind11/filesystem/cufile_py.cpp
        pybind11/memory/memory_py.h
        pybind11/memory/memory_pydoc.h
        pybind11/memory/memory_py.cpp
        )
target_link_libraries(cucim
    PRIVATE
        cucim::cucim
        deps::fmt
        deps::json
        deps::pybind11_json
        )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cucim)
set_target_properties(cucim PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
        OUTPUT_NAME cucim/_cucim
        )

#target_compile_options(cucim PRIVATE -g -O0)

#pybind11_add_module(cucim_io
#    MODULE
#        pybind11/io/device.cpp
#        )
#target_link_libraries(cucim_io
#    PRIVATE
#        cucim::cucim)
#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/cucim/io)
#set_target_properties(cucim_io PROPERTIES OUTPUT_NAME cucim/io/_io)


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
