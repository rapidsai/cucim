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

# CUDA_STANDARD 17 is supported from CMAKE 3.18
# : https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_STANDARD.html
cmake_minimum_required(VERSION 3.18)

################################################################################
# Prerequisite statements
################################################################################

# Set VERSION
unset(VERSION CACHE)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/VERSION VERSION)

# Append local cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

project(cuslide VERSION ${VERSION} DESCRIPTION "cuslide" LANGUAGES C CXX)
set(CUCIM_PLUGIN_NAME "cucim.kit.cuslide")

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
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) # TODO: create two library, one with CXX11 ABI and one without it.

################################################################################
# Define dependencies
################################################################################
superbuild_depend(fmt)
superbuild_depend(libjpeg-turbo) # libjpeg-turbo should be located before libtiff as libtiff depends on libjpeg-turbo
superbuild_depend(libopenjpeg)
superbuild_depend(libtiff)
superbuild_depend(catch2)
superbuild_depend(openslide)
superbuild_depend(googletest)
superbuild_depend(googlebenchmark)
superbuild_depend(cli11)
superbuild_depend(pugixml)
superbuild_depend(json)
superbuild_depend(libdeflate)
superbuild_depend(nvjpeg)

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
    src/cuslide/cuslide.cpp
    src/cuslide/cuslide.h
    src/cuslide/deflate/deflate.cpp
    src/cuslide/deflate/deflate.h
    src/cuslide/jpeg/libjpeg_turbo.cpp
    src/cuslide/jpeg/libjpeg_turbo.h
    src/cuslide/jpeg/libnvjpeg.cpp
    src/cuslide/jpeg/libnvjpeg.h
    src/cuslide/jpeg2k/color_conversion.cpp
    src/cuslide/jpeg2k/color_conversion.h
    src/cuslide/jpeg2k/color_table.h
    src/cuslide/jpeg2k/libopenjpeg.cpp
    src/cuslide/jpeg2k/libopenjpeg.h
    src/cuslide/loader/nvjpeg_processor.cpp
    src/cuslide/loader/nvjpeg_processor.h
    ${deps-libopenjpeg_SOURCE_DIR}/src/bin/common/color.c  # for color_sycc_to_rgb() and color_apply_icc_profile()
    src/cuslide/lzw/lzw.cpp
    src/cuslide/lzw/lzw.h
    src/cuslide/lzw/lzw_libtiff.cpp
    src/cuslide/lzw/lzw_libtiff.h
    src/cuslide/raw/raw.cpp
    src/cuslide/raw/raw.h
    src/cuslide/tiff/ifd.cpp
    src/cuslide/tiff/ifd.h
    src/cuslide/tiff/tiff.cpp
    src/cuslide/tiff/tiff.h
    src/cuslide/tiff/types.h)

# compile color.c for libopenjpeg with c++
set_source_files_properties(${deps-libopenjpeg_SOURCE_DIR}/src/bin/common/color.c
    PROPERTIES
        LANGUAGE C
        CMAKE_CXX_VISIBILITY_PRESET default
        CMAKE_C_VISIBILITY_PRESET default
        CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

# Ignore warnings in existing source code from libjpeg-turbo
set_source_files_properties(src/cuslide/jpeg/libjpeg_turbo.cpp
    PROPERTIES
        COMPILE_OPTIONS "-Wno-error" # or, "-Wno-write-strings;-Wno-clobbered"
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
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(${CUCIM_PLUGIN_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)

# Link libraries
target_link_libraries(${CUCIM_PLUGIN_NAME}
        PRIVATE
            deps::nvjpeg_static # add this before cudart so that nvjpeg.h in static library takes precedence.
            CUDA::culibos       # for nvjpeg
            CUDA::cudart
            deps::fmt
            cucim::cucim
            deps::libtiff
            deps::libjpeg-turbo
            deps::libopenjpeg
            deps::libopenjpeg-lcms2
            deps::pugixml
            deps::json
            deps::libdeflate
        )

target_include_directories(${CUCIM_PLUGIN_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/src
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

#set_target_properties(${CUCIM_PLUGIN_NAME} PROPERTIES LINK_FLAGS
#                        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cuslide.map")

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
        cuslide_tests
        cuslide_benchmarks
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

# Currently cuslide plugin doesn't have include path so comment out
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
