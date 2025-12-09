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

message(STATUS "=============================================================")
message(STATUS "cuslide2 PURE nvImageCodec - Dependencies:")
message(STATUS "  ✓ fmt (logging)")
message(STATUS "  ✓ nvImageCodec (GPU JPEG/JPEG2000/deflate/LZW decoding)")
message(STATUS "  ✓ pugixml (XML metadata)")
message(STATUS "  ✓ json (JSON metadata)")
message(STATUS "  ✓ openslide (testing)")
message(STATUS "  ✓ googletest, catch2, googlebenchmark, cli11 (testing)")
message(STATUS "")
message(STATUS "NOTE: No codec libraries needed (libjpeg/libtiff/libdeflate)")
message(STATUS "      All decoding handled by nvImageCodec GPU acceleration!")
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

# Link libraries
target_link_libraries(${CUCIM_PLUGIN_NAME}
        PRIVATE
            deps::fmt
            cucim::cucim
            deps::pugixml
            deps::json
        )

# nvImageCodec linking: dynamic (runtime) vs static (compile-time)
option(WITH_DYNAMIC_NVIMGCODEC "Use dynamic loading for nvImageCodec (dlopen at runtime)" ON)

if(WITH_DYNAMIC_NVIMGCODEC)

    message(STATUS "============== Dynamic nvImageCodec Loading ==============")

    set(nvimgcodec_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/nvimgcodec_dynlink)
    include_directories(AFTER SYSTEM "${nvimgcodec_INCLUDE_DIR}")


    # Use pre-generated stubs (faster builds, no Python clang dependency)
    # To regenerate: cd build-release && make dynlink_nvimgcodec_gen.cc
    #                then copy to src/nvimgcodec_dynlink/nvimgcodec_stubs_generated.cc

    # Create dynamic loading wrapper library
    add_library(nvimgcodec_dynlink STATIC
        src/nvimgcodec_dynlink/nvimgcodec_wrap.cc
        src/nvimgcodec_dynlink/nvimgcodec_stubs_generated.cc  # Pre-generated stubs
    )
    set_target_properties(nvimgcodec_dynlink PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nvimgcodec_dynlink SYSTEM PRIVATE
        ${nvimgcodec_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(nvimgcodec_dynlink PRIVATE ${CMAKE_DL_LIBS})

    # Link cuslide2 against dynlink wrapper (instead of libnvimgcodec.so)
    target_link_libraries(${CUCIM_PLUGIN_NAME} PRIVATE nvimgcodec_dynlink)
    target_compile_definitions(${CUCIM_PLUGIN_NAME} PRIVATE CUCIM_HAS_NVIMGCODEC WITH_DYNAMIC_NVIMGCODEC)

    # Set rpath for runtime library search
    foreach(rpath IN ITEMS
        "$ORIGIN"
        "$ORIGIN/../nvimgcodec"
        "$ORIGIN/../lib"
        "$ORIGIN/../../lib")
        message(STATUS "  Adding to rpath: ${rpath}")
        list(APPEND CMAKE_BUILD_RPATH ${rpath})
        list(APPEND CMAKE_INSTALL_RPATH ${rpath})
    endforeach()

    message(STATUS "✓ Dynamic nvImageCodec loading enabled")
    message(STATUS "  Runtime: libnvimgcodec.so loaded via dlopen")
    message(STATUS "  Python wheels: Place libnvimgcodec.so in lib/ directory")
    message(STATUS "==========================================================")
else()
    # Static linking: link directly against nvImageCodec (build-time dependency)
    if(TARGET deps::nvimgcodec)
        target_link_libraries(${CUCIM_PLUGIN_NAME} PRIVATE deps::nvimgcodec)
        target_compile_definitions(${CUCIM_PLUGIN_NAME} PRIVATE CUCIM_HAS_NVIMGCODEC)
        message(STATUS "✓ nvImageCodec enabled (static linking) - GPU-accelerated decoding available")
    else()
        message(STATUS "⚠ nvImageCodec target not found - GPU acceleration disabled")
    endif()
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
            ${CMAKE_CURRENT_SOURCE_DIR}/../cucim.kit.cuslide/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src  # Include cuslide2 src for nvimgcodec headers
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
################################################################################
add_subdirectory(tests)
add_subdirectory(benchmarks)

################################################################################
# Install
################################################################################

set(INSTALL_TARGETS
        ${CUCIM_PLUGIN_NAME}
        cuslide2_tests
        cuslide2_benchmarks
        )

# Add nvimgcodec_dynlink to install targets if dynamic loading is enabled
if(WITH_DYNAMIC_NVIMGCODEC)
    list(APPEND INSTALL_TARGETS nvimgcodec_dynlink)
endif()

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
