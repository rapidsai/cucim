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

# Set VERSION and BUILD
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/VERSION VERSION)
set(PROJECT_VERSION_BUILD dev)

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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17) # Clion issue: https://youtrack.jetbrains.com/issue/CPP-19165 (fixed)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
cucim_define_cuda_architectures(60;70;75;80;86)
# https://github.com/Kitware/CMake/blob/master/Modules/Compiler/NVIDIA-CUDA.cmake#L11
# https://gitlab.kitware.com/cmake/cmake/-/issues/19017
# For CUDA >= 10.2, we cannot use --compiler-options as '-forward-unknown-to-host-compiler' wouldbe added by default to nvcc options.
# For the reason, we add "${CMAKE_CXX_FLAGS}" instead of "--compiler-options ${CMAKE_CXX_FLAGS}" here.
# ==> We changed to use "${CMAKE_CUDA_FLAGS}" instead of "${CMAKE_CXX_FLAGS}" ${CMAKE_CXX_FLAGS} can have wrong options such as '-march=nocona'
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -Xptxas=-v")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo")

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
option(CUCIM_SUPPORT_GDS "Support cufile library" OFF)
option(CUCIM_STATIC_GDS "Use static cufile library" OFF)

# Setup CXX11 ABI
# : Adds CXX11 ABI definition to the compiler command line for targets in the current directory,
#   whether added before or after this command is invoked, and for the ones in sub-directories added after.
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) # TODO: create two library, one with CXX11 ABI and one without it.

################################################################################
# Define dependencies
################################################################################
superbuild_depend(fmt)
#superbuild_depend(boost)
superbuild_depend(abseil)
superbuild_depend(rmm) # this imports googletest internally
#superbuild_depend(googletest)
superbuild_depend(googlebenchmark)
superbuild_depend(openslide)
superbuild_depend(catch2)
superbuild_depend(cli11)
superbuild_depend(json)
superbuild_depend(libcuckoo)
superbuild_depend(boost-header-only)

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
#        ${CUCIM_PACKAGE_NAME}-header-only
#        rmm
        fmt-header-only
#        spdlog_header_only # required by rmm
        cucim_benchmarks
#        cufile_stub
#        cucim_tests
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
#            ${deps-rmm_SOURCE_DIR}/include/
#            ${deps-rmm_SOURCE_DIR}/../thrust-src/thrust
#            ${deps-rmm_SOURCE_DIR}/../spdlog-src/spdlog
#            ${THRUST_INCLUDE_DIR}/thrust # thrust needs to be installed because rmm depends on thrust
#            ${SPDLOG_INCLUDE_DIR}/spdlog # spdlog needs to be installed because rmm depends on spdlog
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR})

# Copy 3rdparty headers
install(DIRECTORY
            ${deps-fmt_SOURCE_DIR}/include/
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