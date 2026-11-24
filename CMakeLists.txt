#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

cmake_minimum_required(VERSION 4.3 FATAL_ERROR)

################################################################################
# Prerequisite statements
################################################################################

# Set VERSION and BUILD
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/VERSION VERSION)
# strip alpha version info
string(REGEX REPLACE "a.*$" "" VERSION ${VERSION})
set(PROJECT_VERSION_BUILD dev)

# Fetch rapids-cmake (used to obtain the RAPIDS-pinned CCCL version)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/rapids_config.cmake)

# Build for all RAPIDS-supported GPU architectures unless CMAKE_CUDA_ARCHITECTURES
# or the CUDAARCHS environment variable is set (e.g. CUDAARCHS=NATIVE for the
# GPU(s) in the build machine only).
include(rapids-cuda)
rapids_cuda_init_architectures(libcucim)

# Append local cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cpp/cmake/modules")
project(libcucim VERSION ${VERSION} DESCRIPTION "libcucim" LANGUAGES CXX CUDA)

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
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

# Include CUDA headers explicitly for VSCode intelli-sense
include_directories(AFTER SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Disable visibility to not expose unnecessary symbols. For CUDA sources this
# also keeps the CUB/libcu++ symbols they instantiate from being exported and
# interposing with other CUB users loaded in the same process (e.g. CuPy).
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
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

# Use the CCCL version pinned by rapids-cmake (as other RAPIDS projects do)
# rather than the one bundled with the CUDA Toolkit, which for every supported
# toolkit predates the cuda::mr types in <cucim/memory/device_resource.h>.
#
# The install export set is what makes rapids-cmake install CCCL, under
# include/rapids and lib/rapids, so that installed consumers of that public
# header find it next to cuCIM; cuCIM does not otherwise use rapids-cmake
# export sets.
include(rapids-cmake)
include(rapids-cpm)
rapids_cpm_init()
include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl(INSTALL_EXPORT_SET cucim-exports)
set(CUCIM_CCCL_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/rapids")
get_target_property(CUCIM_CCCL_INCLUDE_DIRS CCCL::libcudacxx INTERFACE_INCLUDE_DIRECTORIES)

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
option(CUCIM_INSTALL_BENCHMARKS "Install the C++ benchmark executables" ON)

set(INSTALL_TARGETS
        ${CUCIM_PACKAGE_NAME}
        )

if(CUCIM_INSTALL_BENCHMARKS)
    list(APPEND INSTALL_TARGETS cucim_benchmarks)
endif()

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
