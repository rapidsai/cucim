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

# Append local cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

project(cuslide2 VERSION ${VERSION} DESCRIPTION "cuslide2" LANGUAGES C CXX)
set(CUCIM_PLUGIN_NAME "cucim.kit.cuslide2")

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

# Setup CXX11 ABI
# : Adds CXX11 ABI definition to the compiler command line for targets in the current directory,
#   whether added before or after this command is invoked, and for the ones in sub-directories added after.
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) # TODO: create two library, one with CXX11 ABI and one without it.

################################################################################
# Define dependencies - PURE nvImageCodec (minimal dependencies)
################################################################################
superbuild_depend(fmt)
# REMOVED: CPU decoder dependencies (nvImageCodec handles all compression)
# superbuild_depend(libjpeg-turbo)
# superbuild_depend(libopenjpeg)
# superbuild_depend(libtiff)
# superbuild_depend(libdeflate)
superbuild_depend(openslide)
# Testing dependencies
superbuild_depend(catch2)
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

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

################################################################################
# Add library: cucim
################################################################################

# NOTE: Commented out for infrastructure-only PR. Will be enabled in follow-up PR
#       with actual implementation once source files are added.
# TODO: Uncomment this section when src/ directory is added with implementation

message(STATUS "=============================================================")
message(STATUS "cuslide2 PURE nvImageCodec - Dependencies:")
message(STATUS "  ✓ fmt (logging)")
message(STATUS "  ✓ nvImageCodec (GPU-accelerated JPEG/JPEG2000/deflate/LZW)")
message(STATUS "  ✓ pugixml (XML metadata parsing)")
message(STATUS "  ✓ json (JSON metadata)")
message(STATUS "  ✓ openslide (for testing)")
message(STATUS "  ✓ googletest, catch2, googlebenchmark, cli11 (testing)")
message(STATUS "")
message(STATUS "Build configured: Pure GPU-accelerated decoding with tests!")
message(STATUS "=============================================================")

# Add library - PURE nvImageCodec implementation (no CPU fallbacks)
add_library(${CUCIM_PLUGIN_NAME}
    # Main plugin interface
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/cuslide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/cuslide.h
    # TIFF structure management (uses nvImageCodec for parsing)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/tiff/ifd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/tiff/ifd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/tiff/tiff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/tiff/tiff.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/tiff/types.h
    # nvImageCodec decoding and TIFF parsing (GPU-accelerated)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/nvimgcodec/nvimgcodec_decoder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/nvimgcodec/nvimgcodec_tiff_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/nvimgcodec/nvimgcodec_tiff_parser.h)

# No special source file properties needed for pure nvImageCodec implementation

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

# Link libraries - Core dependencies (always required)
target_link_libraries(${CUCIM_PLUGIN_NAME}
        PRIVATE
            deps::fmt
            cucim::cucim
            deps::pugixml
            deps::json
        )

# Conditionally link nvImageCodec if available
if(TARGET deps::nvimgcodec)
    get_target_property(NVIMGCODEC_LOCATION deps::nvimgcodec IMPORTED_LOCATION)
    get_target_property(NVIMGCODEC_INTERFACE_LINK deps::nvimgcodec INTERFACE_LINK_LIBRARIES)

    # Check if it's a real target (has location or interface links) vs dummy target
    if(NVIMGCODEC_LOCATION OR NVIMGCODEC_INTERFACE_LINK OR TARGET nvimgcodec::nvimgcodec)
        target_link_libraries(${CUCIM_PLUGIN_NAME} PRIVATE deps::nvimgcodec)
        target_compile_definitions(${CUCIM_PLUGIN_NAME} PRIVATE CUCIM_HAS_NVIMGCODEC)
        message(STATUS "✓ nvImageCodec enabled - GPU-accelerated JPEG/JPEG2000 decoding available")
    else()
        message(STATUS "⚠ nvImageCodec target exists but is dummy - GPU acceleration disabled")
    endif()
else()
    message(STATUS "⚠ nvImageCodec target not found - GPU acceleration disabled")
endif()
if (TARGET CUDA::nvjpeg_static)
        target_link_libraries(${CUCIM_PLUGIN_NAME}
            PRIVATE
                # Add nvjpeg before cudart so that nvjpeg.h in static library takes precedence.
                CUDA::nvjpeg_static
                # Add CUDA::culibos to link necessary methods for 'deps::nvjpeg_static'
                CUDA::culibos
                CUDA::cudart
        )
else()
        target_link_libraries(${CUCIM_PLUGIN_NAME}
            PRIVATE
                CUDA::nvjpeg
                CUDA::cudart
        )
endif()

target_include_directories(${CUCIM_PLUGIN_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../cucim.kit.cuslide/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src  # Include cuslide2 src for nvimgcodec headers
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
################################################################################
add_subdirectory(tests)
add_subdirectory(benchmarks)

################################################################################
# Install
################################################################################
# NOTE: Disabled for infrastructure-only PR
# TODO: Uncomment when library target exists
set(INSTALL_TARGETS
        ${CUCIM_PLUGIN_NAME}
        # cuslide_tests  # Disabled for infrastructure-only PR
        # cuslide_benchmarks  # Disabled for infrastructure-only PR
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

# REMOVED: endif() - no longer needed since we removed the if(TRUE) wrapper

unset(BUILD_SHARED_LIBS CACHE)
