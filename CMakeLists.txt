#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

################################################################################
# Prerequisite statements
################################################################################

# Set VERSION and BUILD
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/VERSION VERSION)
# strip alpha version info
string(REGEX REPLACE "a.*$" "" VERSION ${VERSION})
set(PROJECT_VERSION_BUILD dev)

# Append local cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cpp/cmake/modules")
project(libcucim VERSION ${VERSION} DESCRIPTION "libcucim" LANGUAGES CXX)

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

find_package(CUDAToolkit REQUIRED)
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
option(CUCIM_SUPPORT_GDS "Support cufile library" ON)
option(CUCIM_STATIC_GDS "Use static cufile library" OFF)
option(CUCIM_SUPPORT_CUDA "Support CUDA" ON)
option(CUCIM_SUPPORT_NVTX "Support NVTX" ON)

# Setup CXX11 ABI
# : Adds CXX11 ABI definition to the compiler command line for targets in the current directory,
#   whether added before or after this command is invoked, and for the ones in sub-directories added after.
#
# Why 0 (this is the canonical explanation; the other CMakeLists.txt files that
# set the same definition point here):
#
#   This dates to building manylinux2014 wheels. That platform tag requires a
#   gcc4-compatible libstdc++ ABI (gcc configured with
#   '--with-default-libstdcxx-abi=gcc4-compatible', see
#   https://gcc.gnu.org/onlinedocs/libstdc++/manual/configure.html), which forces
#   _GLIBCXX_USE_CXX11_ABI=0 on CentOS 7.
#
# What it costs: parts of the standard library that are only defined for the new
# ABI are unavailable, most visibly std::pmr::string. ImageMetadata therefore
# keeps two sets of member declarations and stores strings as std::string_view
# into its own memory resource; see cpp/include/cucim/io/format/image_format.h
# and ImageMetadata::store_string().
#
# Changing it is not a local edit: cpp/CMakeLists.txt exports this definition as
# an INTERFACE compile definition, so it is part of the installed package's
# contract and flipping it breaks ABI for anything already linked against
# libcucim. The manylinux2014 constraint itself no longer appears to apply (the
# repository no longer references manylinux anywhere and requires-python is
# >=3.11), so this is worth revisiting deliberately as its own change rather
# than in passing.
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

################################################################################
# Define basic dependencies
################################################################################
# Include pthread
# (https://cmake.org/cmake/help/v3.18/module/FindThreads.html)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

################################################################################
# Define dependencies
################################################################################
superbuild_depend(fmt)
#superbuild_depend(boost)
superbuild_depend(abseil)
superbuild_depend(googletest)
superbuild_depend(googlebenchmark)
superbuild_depend(openslide)
superbuild_depend(catch2)
superbuild_depend(cli11)
superbuild_depend(json)
superbuild_depend(libcuckoo)
superbuild_depend(boost-header-only)
superbuild_depend(nvtx3)
superbuild_depend(taskflow)

################################################################################
# Define some names
################################################################################
set(CUCIM_PACKAGE_NAME cucim) # cucim

################################################################################
# Add subdirectories
################################################################################
add_subdirectory(cpp)
add_subdirectory(gds)
add_subdirectory(benchmarks)
add_subdirectory(examples/cpp)

################################################################################
# Write CMakeLists.txt for C++ examples
################################################################################

configure_file(${CMAKE_CURRENT_LIST_DIR}/examples/cpp/CMakeLists.txt.examples.release.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt.examples.release
    @ONLY)

################################################################################
# Install
################################################################################
set(INSTALL_TARGETS
        ${CUCIM_PACKAGE_NAME}
        fmt-header-only
        cucim_benchmarks
        )

install(TARGETS ${INSTALL_TARGETS}
        EXPORT ${CUCIM_PACKAGE_NAME}-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT ${CUCIM_PACKAGE_NAME}_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT ${CUCIM_PACKAGE_NAME}_Runtime
                NAMELINK_COMPONENT ${CUCIM_PACKAGE_NAME}_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT ${CUCIM_PACKAGE_NAME}_Development
        )
install(EXPORT ${CUCIM_PACKAGE_NAME}-targets
        FILE
            ${CUCIM_PACKAGE_NAME}-targets.cmake
        NAMESPACE
            ${CUCIM_PACKAGE_NAME}::
        DESTINATION
            ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PACKAGE_NAME})

# Write package configs
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cmake/${CUCIM_PACKAGE_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cpp/cmake/${CUCIM_PACKAGE_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PACKAGE_NAME}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cpp/cmake/${CUCIM_PACKAGE_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cpp/cmake/${CUCIM_PACKAGE_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cpp/cmake/${CUCIM_PACKAGE_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PACKAGE_NAME}
)

install(DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/ # Suffix '/' is necessary to not copy to install/include/include folder.
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR})

# Copy 3rdparty headers
install(DIRECTORY
            ${deps-fmt_SOURCE_DIR}/include/
            ${deps-nvtx3_SOURCE_DIR}/c/include/
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR}/${CUCIM_PACKAGE_NAME}/3rdparty)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE ${CUCIM_PACKAGE_NAME})

#set(CPACK_PACKAGE_NAME "${CUCIM_PACKAGE_NAME}")
#set(CPACK_PACKAGE_VENDOR "nvidia.com")
#set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "cuCIM - GPU-accelerated image processing toolkit")
#set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
#set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
#set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
#set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
#set(CPACK_PACKAGE_INSTALL_DIRECTORY "cucim_cpack")  # TODO: update this
#include(CPack)

# Unset cached options needed
unset(CUCIM_STATIC_GDS CACHE)
