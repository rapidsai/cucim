# Apache License, Version 2.0
# Copyright 2021-2024 NVIDIA Corporation
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

cmake_minimum_required(VERSION 3.26.4)

################################################################################
# Prerequisite statements
################################################################################

# Set VERSION
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/../../../VERSION VERSION)
# strip alpha version info
string(REGEX REPLACE "a.*$" "" VERSION ${VERSION})

# Append local cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

project(cumed VERSION ${VERSION} DESCRIPTION "cumed" LANGUAGES CXX)
set(CUCIM_PLUGIN_NAME "cucim.kit.cumed")

################################################################################
# Include utilities
################################################################################
include(SuperBuildUtils)
include(CuCIMUtils)

################################################################################
# Set cmake policy
################################################################################
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19")
    cmake_policy(SET CMP0110 NEW) # For add_test() to support arbitrary characters in test name
endif()

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


################################################################################
# Define dependencies
################################################################################
superbuild_depend(fmt)
superbuild_depend(catch2)
superbuild_depend(googletest)
superbuild_depend(googlebenchmark)
superbuild_depend(cli11)

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

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

################################################################################
# Add library: cucim
################################################################################

# Add library
add_library(${CUCIM_PLUGIN_NAME}
    src/cumed/cumed.cpp
    src/cumed/cumed.h
    )

# Compile options
set_target_properties(${CUCIM_PLUGIN_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
)
target_compile_features(${CUCIM_PLUGIN_NAME} PRIVATE cxx_std_17)

# Link libraries
target_link_libraries(${CUCIM_PLUGIN_NAME}
        PRIVATE
            CUDA::cudart
            deps::fmt
            cucim::cucim
        )

target_include_directories(${CUCIM_PLUGIN_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/src
            # turbojpeg.h is not included in 'turbojpeg-static' so manually include
            ${deps-libjpeg-turbo_SOURCE_DIR}
        )

# Do not generate SONAME as this would be used as plugin
# Need to use IMPORTED_NO_SONAME when using this .so file.
set_target_properties(${CUCIM_PLUGIN_NAME} PROPERTIES NO_SONAME 1)
# Prevent relative path problem of .so with no DT_SONAME.
# : https://stackoverflow.com/questions/27261288/cmake-linking-shared-c-object-from-externalproject-produces-binaries-with-rel
target_link_options(${CUCIM_PLUGIN_NAME} PRIVATE "LINKER:-soname=${CUCIM_PLUGIN_NAME}@${PROJECT_VERSION}.so")

# Do not add 'lib' prefix for the library
set_target_properties(${CUCIM_PLUGIN_NAME} PROPERTIES PREFIX "")
# Postfix version
set_target_properties(${CUCIM_PLUGIN_NAME} PROPERTIES OUTPUT_NAME "${CUCIM_PLUGIN_NAME}@${PROJECT_VERSION}")

################################################################################
# Add tests
#########################################################std#######################
add_subdirectory(tests)
add_subdirectory(benchmarks)

################################################################################
# Install
################################################################################
set(INSTALL_TARGETS
        ${CUCIM_PLUGIN_NAME}
        cumed_tests
        cumed_benchmarks
        )

install(TARGETS ${INSTALL_TARGETS}
        EXPORT ${CUCIM_PLUGIN_NAME}-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${CUCIM_PLUGIN_NAME}_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT ${CUCIM_PLUGIN_NAME}_Runtime
        NAMELINK_COMPONENT ${CUCIM_PLUGIN_NAME}_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT ${CUCIM_PLUGIN_NAME}_Development
        )

# Currently cumed plugin doesn't have include path so comment out
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ${CUCIM_PLUGIN_NAME}-targets
        FILE
        ${CUCIM_PLUGIN_NAME}-targets.cmake
        NAMESPACE
        ${PROJECT_NAME}::
        DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME})

# Write package configs
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME}
)


set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE ${CUCIM_PLUGIN_NAME})


# Write package configs
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME}
)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON) # TODO: duplicate?
export(PACKAGE ${CUCIM_PLUGIN_NAME})

unset(BUILD_SHARED_LIBS CACHE)
