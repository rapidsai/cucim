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

# Include cuslide2's local cmake modules first (for custom SuperBuildUtils.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
# Reuse cmake modules from cuslide plugin (for shared deps)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cucim.kit.cuslide/cmake/modules")
# Include central CuCIM cmake modules (for CuCIMUtils.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cpp/cmake/modules")

project(cuslide2 VERSION ${VERSION} DESCRIPTION "cuslide2" LANGUAGES C CXX)
set(CUCIM_PLUGIN_NAME "cucim.kit.cuslide2")

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

# NOTE: Commented out for infrastructure-only PR. Will be enabled in follow-up PR
#       with actual implementation once source files are added.
# TODO: Uncomment this section when src/ directory is added with implementation

message(STATUS "=============================================================")
message(STATUS "cuslide2 infrastructure validation - Dependencies verified:")
message(STATUS "  ✓ fmt")
message(STATUS "  ✓ libjpeg-turbo")
message(STATUS "  ✓ libopenjpeg")
message(STATUS "  ✓ libtiff")
message(STATUS "  ✓ libdeflate")
message(STATUS "  ✓ nvImageCodec")
message(STATUS "  ✓ pugixml")
message(STATUS "  ✓ json")
message(STATUS "  ✓ openslide")
message(STATUS "")
message(STATUS "Infrastructure setup complete!")
message(STATUS "Next step: Add src/ directory with cuslide2 implementation")
message(STATUS "=============================================================")

# For infrastructure-only PR: Create a dummy target so build succeeds
# This will be replaced with actual library in follow-up PR
add_custom_target(${CUCIM_PLUGIN_NAME}
    COMMAND ${CMAKE_COMMAND} -E echo "cuslide2 infrastructure-only: No library to build - dependencies verified"
    COMMENT "Infrastructure-only build - no plugin binary generated"
)

# TODO: Replace dummy target with real library when source files exist
if(FALSE)  # Change to TRUE when source files exist
# Add library
# Use local src/ implementation with nvImageCodec support
set(CUSLIDE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide)
add_library(${CUCIM_PLUGIN_NAME} MODULE
    ${CUSLIDE_SRC_DIR}/cuslide.cpp
    ${CUSLIDE_SRC_DIR}/cuslide.h
    ${CUSLIDE_SRC_DIR}/deflate/deflate.cpp
    ${CUSLIDE_SRC_DIR}/deflate/deflate.h
    ${CUSLIDE_SRC_DIR}/jpeg/libjpeg_turbo.cpp
    ${CUSLIDE_SRC_DIR}/jpeg/libjpeg_turbo.h
    ${CUSLIDE_SRC_DIR}/jpeg/libnvjpeg.cpp
    ${CUSLIDE_SRC_DIR}/jpeg/libnvjpeg.h
    ${CUSLIDE_SRC_DIR}/jpeg2k/color_conversion.cpp
    ${CUSLIDE_SRC_DIR}/jpeg2k/color_conversion.h
    ${CUSLIDE_SRC_DIR}/jpeg2k/color_table.h
    ${CUSLIDE_SRC_DIR}/jpeg2k/libopenjpeg.cpp
    ${CUSLIDE_SRC_DIR}/jpeg2k/libopenjpeg.h
    ${CUSLIDE_SRC_DIR}/loader/nvjpeg_processor.cpp
    ${CUSLIDE_SRC_DIR}/loader/nvjpeg_processor.h
    ${deps-libopenjpeg_SOURCE_DIR}/src/bin/common/color.c  # for color_sycc_to_rgb() and color_apply_icc_profile()
    ${CUSLIDE_SRC_DIR}/lzw/lzw.cpp
    ${CUSLIDE_SRC_DIR}/lzw/lzw.h
    ${CUSLIDE_SRC_DIR}/lzw/lzw_libtiff.cpp
    ${CUSLIDE_SRC_DIR}/lzw/lzw_libtiff.h
    ${CUSLIDE_SRC_DIR}/raw/raw.cpp
    ${CUSLIDE_SRC_DIR}/raw/raw.h
    ${CUSLIDE_SRC_DIR}/tiff/ifd.cpp
    ${CUSLIDE_SRC_DIR}/tiff/ifd.h
    ${CUSLIDE_SRC_DIR}/tiff/tiff.cpp
    ${CUSLIDE_SRC_DIR}/tiff/tiff.h
    ${CUSLIDE_SRC_DIR}/tiff/types.h
    # nvImageCodec decoder implementation
    ${CUSLIDE_SRC_DIR}/nvimgcodec/nvimgcodec_decoder.cpp
    ${CUSLIDE_SRC_DIR}/nvimgcodec/nvimgcodec_decoder.h
    ${CUSLIDE_SRC_DIR}/nvimgcodec/nvimgcodec_tiff_parser.cpp
    ${CUSLIDE_SRC_DIR}/nvimgcodec/nvimgcodec_tiff_parser.h)

# compile color.c for libopenjpeg with c++
set_source_files_properties(${deps-libopenjpeg_SOURCE_DIR}/src/bin/common/color.c
    PROPERTIES
        LANGUAGE C
        CMAKE_CXX_VISIBILITY_PRESET default
        CMAKE_C_VISIBILITY_PRESET default
        CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

# Ignore warnings in existing source code from libjpeg-turbo
set_source_files_properties(${CUSLIDE_SRC_DIR}/jpeg/libjpeg_turbo.cpp
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
            deps::fmt
            cucim::cucim
            deps::libtiff
            deps::libjpeg-turbo
            deps::libopenjpeg
            deps::libopenjpeg-lcms2
            deps::pugixml
            deps::json
            deps::libdeflate
            $<TARGET_NAME_IF_EXISTS:deps::nvimgcodec>
        )

# Add nvImageCodec compile definition if the target exists
if(TARGET deps::nvimgcodec)
    target_compile_definitions(${CUCIM_PLUGIN_NAME} PRIVATE CUCIM_HAS_NVIMGCODEC)
    message(STATUS "✓ nvImageCodec enabled - GPU-accelerated JPEG/JPEG2000 decoding available")
endif()
if (TARGET CUDA::nvjpeg_static)
        target_link_libraries(${CUCIM_PLUGIN_NAME}
            PRIVATE
                CUDA::nvjpeg_static
                # culibos provides OS abstraction functions needed by nvjpeg_static
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
            ${CMAKE_CURRENT_SOURCE_DIR}/src
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
# NOTE: Commented out for infrastructure-only PR. Will be enabled in follow-up PR
#       with actual implementation.
# add_subdirectory(tests)
# add_subdirectory(benchmarks)

################################################################################
# Install
################################################################################
# NOTE: Disabled for infrastructure-only PR
# TODO: Uncomment when library target exists
# set(INSTALL_TARGETS
#         ${CUCIM_PLUGIN_NAME}
#         # cuslide_tests  # Disabled for infrastructure-only PR
#         # cuslide_benchmarks  # Disabled for infrastructure-only PR
#         )
#
# install(TARGETS ${INSTALL_TARGETS}
#         EXPORT ${CUCIM_PLUGIN_NAME}-targets
#         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#         COMPONENT ${CUCIM_PLUGIN_NAME}_Runtime
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         COMPONENT ${CUCIM_PLUGIN_NAME}_Runtime
#         NAMELINK_COMPONENT ${CUCIM_PLUGIN_NAME}_Development
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         COMPONENT ${CUCIM_PLUGIN_NAME}_Development
#         )
#
# # Currently cuslide plugin doesn't have include path so comment out
# # install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# install(EXPORT ${CUCIM_PLUGIN_NAME}-targets
#         FILE
#         ${CUCIM_PLUGIN_NAME}-targets.cmake
#         NAMESPACE
#         ${PROJECT_NAME}::
#         DESTINATION
#         ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME})
#
# # Write package configs
# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME}
# )
# write_basic_package_version_file(
#     ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config-version.cmake
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )
# install(
#     FILES
#         ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config.cmake
#         ${CMAKE_CURRENT_BINARY_DIR}/cmake/${CUCIM_PLUGIN_NAME}-config-version.cmake
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CUCIM_PLUGIN_NAME}
# )
#
#
# set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
# export(PACKAGE ${CUCIM_PLUGIN_NAME})

endif()  # End of if(FALSE) block for infrastructure-only PR

unset(BUILD_SHARED_LIBS CACHE)
