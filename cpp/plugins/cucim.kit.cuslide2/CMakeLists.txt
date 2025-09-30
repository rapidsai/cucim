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
    src/cuslide/tiff/types.h
    src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp
    src/cuslide/nvimgcodec/nvimgcodec_decoder.h)

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

# Add nvImageCodec support if available
# Option to automatically install nvImageCodec via conda
option(AUTO_INSTALL_NVIMGCODEC "Automatically install nvImageCodec via conda" ON)
set(NVIMGCODEC_VERSION "0.6.0" CACHE STRING "nvImageCodec version to install")

if(AUTO_INSTALL_NVIMGCODEC)
    message(STATUS "Configuring automatic nvImageCodec installation...")
    
    # Try to find micromamba or conda in various locations
    find_program(MICROMAMBA_EXECUTABLE 
        NAMES micromamba
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin
              ${CMAKE_CURRENT_SOURCE_DIR}/../../bin
              ${CMAKE_CURRENT_SOURCE_DIR}/bin
              $ENV{HOME}/micromamba/bin
              $ENV{HOME}/.local/bin
              /usr/local/bin
              /opt/conda/bin
              /opt/miniconda/bin
        DOC "Path to micromamba executable"
    )
    
    find_program(CONDA_EXECUTABLE 
        NAMES conda mamba
        PATHS $ENV{HOME}/miniconda3/bin
              $ENV{HOME}/anaconda3/bin
              /opt/conda/bin
              /opt/miniconda/bin
              /usr/local/bin
        DOC "Path to conda/mamba executable"
    )
    
    # Determine which conda tool to use
    set(CONDA_CMD "")
    set(CONDA_TYPE "")
    if(MICROMAMBA_EXECUTABLE)
        set(CONDA_CMD ${MICROMAMBA_EXECUTABLE})
        set(CONDA_TYPE "micromamba")
        message(STATUS "Found micromamba: ${MICROMAMBA_EXECUTABLE}")
    elseif(CONDA_EXECUTABLE)
        set(CONDA_CMD ${CONDA_EXECUTABLE})
        set(CONDA_TYPE "conda")
        message(STATUS "Found conda/mamba: ${CONDA_EXECUTABLE}")
    endif()
    
    if(CONDA_CMD)
        # Get conda environment info
        execute_process(
            COMMAND ${CONDA_CMD} info --base
            OUTPUT_VARIABLE CONDA_BASE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        
        # Check current environment
        if(DEFINED ENV{CONDA_PREFIX})
            set(CONDA_ENV_PATH "$ENV{CONDA_PREFIX}")
            message(STATUS "Using conda environment: ${CONDA_ENV_PATH}")
        else()
            set(CONDA_ENV_PATH "${CONDA_BASE_PATH}")
            message(STATUS "Using conda base environment: ${CONDA_ENV_PATH}")
        endif()
        
        # Check if nvImageCodec is already installed
        message(STATUS "Checking for existing nvImageCodec installation...")
        execute_process(
            COMMAND ${CONDA_CMD} list libnvimgcodec-dev
            RESULT_VARIABLE NVIMGCODEC_CHECK_RESULT
            OUTPUT_VARIABLE NVIMGCODEC_CHECK_OUTPUT
            ERROR_QUIET
        )
        
        # Parse version from output if installed
        set(NVIMGCODEC_INSTALLED_VERSION "")
        if(NVIMGCODEC_CHECK_RESULT EQUAL 0)
            string(REGEX MATCH "libnvimgcodec-dev[ ]+([0-9]+\\.[0-9]+\\.[0-9]+)" 
                   VERSION_MATCH "${NVIMGCODEC_CHECK_OUTPUT}")
            if(CMAKE_MATCH_1)
                set(NVIMGCODEC_INSTALLED_VERSION ${CMAKE_MATCH_1})
            endif()
        endif()
        
        # Install or upgrade if needed
        set(NEED_INSTALL FALSE)
        if(NOT NVIMGCODEC_CHECK_RESULT EQUAL 0)
            message(STATUS "nvImageCodec not found - installing version ${NVIMGCODEC_VERSION}")
            set(NEED_INSTALL TRUE)
        elseif(NVIMGCODEC_INSTALLED_VERSION AND NVIMGCODEC_INSTALLED_VERSION VERSION_LESS NVIMGCODEC_VERSION)
            message(STATUS "nvImageCodec ${NVIMGCODEC_INSTALLED_VERSION} found - upgrading to ${NVIMGCODEC_VERSION}")
            set(NEED_INSTALL TRUE)
        else()
            message(STATUS "nvImageCodec ${NVIMGCODEC_INSTALLED_VERSION} already installed (>= ${NVIMGCODEC_VERSION})")
        endif()
        
        if(NEED_INSTALL)
            # Install nvImageCodec with specific version
            message(STATUS "Installing nvImageCodec ${NVIMGCODEC_VERSION} via ${CONDA_TYPE}...")
            execute_process(
                COMMAND ${CONDA_CMD} install 
                    libnvimgcodec-dev=${NVIMGCODEC_VERSION} 
                    libnvimgcodec0=${NVIMGCODEC_VERSION} 
                    -c conda-forge -y
                RESULT_VARIABLE CONDA_INSTALL_RESULT
                OUTPUT_VARIABLE CONDA_INSTALL_OUTPUT
                ERROR_VARIABLE CONDA_INSTALL_ERROR
                TIMEOUT 300  # 5 minute timeout
            )
            
            if(CONDA_INSTALL_RESULT EQUAL 0)
                message(STATUS "✓ Successfully installed nvImageCodec ${NVIMGCODEC_VERSION}")
                
                # Verify installation
                execute_process(
                    COMMAND ${CONDA_CMD} list libnvimgcodec-dev
                    OUTPUT_VARIABLE VERIFY_OUTPUT
                    ERROR_QUIET
                )
                message(STATUS "Verification: ${VERIFY_OUTPUT}")
            else()
                message(WARNING "✗ Failed to install nvImageCodec via ${CONDA_TYPE}")
                message(WARNING "Error: ${CONDA_INSTALL_ERROR}")
                message(STATUS "Falling back to manual detection...")
                
                # Try alternative installation without version constraint
                message(STATUS "Attempting installation without version constraint...")
                execute_process(
                    COMMAND ${CONDA_CMD} install libnvimgcodec-dev libnvimgcodec0 -c conda-forge -y
                    RESULT_VARIABLE CONDA_FALLBACK_RESULT
                    OUTPUT_QUIET
                    ERROR_QUIET
                )
                
                if(CONDA_FALLBACK_RESULT EQUAL 0)
                    message(STATUS "✓ Fallback installation successful")
                else()
                    message(WARNING "✗ Fallback installation also failed")
                endif()
            endif()
        endif()
        
        # Set conda environment variables for library detection
        set(ENV{CONDA_PREFIX} ${CONDA_ENV_PATH})
        
    else()
        message(STATUS "No conda/micromamba found in standard locations")
        message(STATUS "Searched paths:")
        message(STATUS "  - ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin")
        message(STATUS "  - $ENV{HOME}/miniconda3/bin")
        message(STATUS "  - /opt/conda/bin")
        message(STATUS "Skipping automatic installation - will attempt manual detection")
    endif()
endif()

# First try to find it as a package
find_package(nvimgcodec QUIET)

if(NOT nvimgcodec_FOUND)
    # Try to find it manually in pip/conda environment
    # Get Python site-packages directory dynamically
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        
        if(PYTHON_SITE_PACKAGES)
            set(NVIMGCODEC_ROOT "${PYTHON_SITE_PACKAGES}/nvidia/nvimgcodec")
        endif()
    endif()
    
    # Try conda environment detection (both Python packages and native packages)
    if(NOT NVIMGCODEC_ROOT AND DEFINED ENV{CONDA_PREFIX})
        # First try native conda package installation (libnvimgcodec-dev)
        set(CONDA_NATIVE_ROOT "$ENV{CONDA_PREFIX}")
        if(EXISTS "${CONDA_NATIVE_ROOT}/include/nvimgcodec.h" AND 
           (EXISTS "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so.0" OR 
            EXISTS "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so"))
            set(NVIMGCODEC_ROOT "${CONDA_NATIVE_ROOT}")
            message(STATUS "Found nvImageCodec native conda package: ${NVIMGCODEC_ROOT}")
        else()
            # Fallback: try Python site-packages in conda environment
            foreach(PY_VER "3.13" "3.12" "3.11" "3.10" "3.9")
                set(CONDA_PYTHON_ROOT "$ENV{CONDA_PREFIX}/lib/python${PY_VER}/site-packages/nvidia/nvimgcodec")
                if(EXISTS "${CONDA_PYTHON_ROOT}/include/nvimgcodec.h")
                    set(NVIMGCODEC_ROOT "${CONDA_PYTHON_ROOT}")
                    message(STATUS "Found nvImageCodec Python package in conda: ${NVIMGCODEC_ROOT}")
                    break()
                endif()
            endforeach()
        endif()
    endif()
    
    # Check if nvImageCodec was found and determine library path
    set(NVIMGCODEC_LIBRARY_PATH "")
    if(NVIMGCODEC_ROOT AND EXISTS "${NVIMGCODEC_ROOT}/include/nvimgcodec.h")
        # Try different library naming conventions
        if(EXISTS "${NVIMGCODEC_ROOT}/lib/libnvimgcodec.so.0")
            set(NVIMGCODEC_LIBRARY_PATH "${NVIMGCODEC_ROOT}/lib/libnvimgcodec.so.0")
        elseif(EXISTS "${NVIMGCODEC_ROOT}/lib/libnvimgcodec.so")
            set(NVIMGCODEC_LIBRARY_PATH "${NVIMGCODEC_ROOT}/lib/libnvimgcodec.so")
        elseif(EXISTS "${NVIMGCODEC_ROOT}/libnvimgcodec.so.0")
            set(NVIMGCODEC_LIBRARY_PATH "${NVIMGCODEC_ROOT}/libnvimgcodec.so.0")
        elseif(EXISTS "${NVIMGCODEC_ROOT}/libnvimgcodec.so")
            set(NVIMGCODEC_LIBRARY_PATH "${NVIMGCODEC_ROOT}/libnvimgcodec.so")
        endif()
        
        if(NVIMGCODEC_LIBRARY_PATH)
            message(STATUS "✓ Found nvImageCodec installation:")
            message(STATUS "  Headers: ${NVIMGCODEC_ROOT}/include")
            message(STATUS "  Library: ${NVIMGCODEC_LIBRARY_PATH}")
            
            # Set up include directories and library
            target_include_directories(${CUCIM_PLUGIN_NAME} PRIVATE "${NVIMGCODEC_ROOT}/include")
            target_link_libraries(${CUCIM_PLUGIN_NAME} PRIVATE "${NVIMGCODEC_LIBRARY_PATH}")
            target_compile_definitions(${CUCIM_PLUGIN_NAME} PRIVATE CUCIM_HAS_NVIMGCODEC)
            
            message(STATUS "✓ nvImageCodec enabled - GPU-accelerated JPEG/JPEG2000 decoding available")
        else()
            message(STATUS "✗ nvImageCodec headers found but library missing")
            message(STATUS "  Headers: ${NVIMGCODEC_ROOT}/include/nvimgcodec.h")
            message(STATUS "  Expected library locations:")
            message(STATUS "    - ${NVIMGCODEC_ROOT}/lib/libnvimgcodec.so.0")
            message(STATUS "    - ${NVIMGCODEC_ROOT}/lib/libnvimgcodec.so")
        endif()
    else()
        message(STATUS "✗ nvImageCodec not found - using fallback decoders")
        message(STATUS "To enable nvImageCodec support:")
        message(STATUS "  Option 1 (conda): ./bin/micromamba install libnvimgcodec-dev -c conda-forge")
        message(STATUS "  Option 2 (pip):   pip install nvidia-nvimgcodec-cu12[all]")
        message(STATUS "  Option 3 (cmake): cmake -DAUTO_INSTALL_NVIMGCODEC=ON ..")
    endif()
else()
    target_link_libraries(${CUCIM_PLUGIN_NAME}
        PRIVATE
            nvimgcodec::nvimgcodec
    )
    target_compile_definitions(${CUCIM_PLUGIN_NAME} PRIVATE CUCIM_HAS_NVIMGCODEC)
    message(STATUS "nvImageCodec found via find_package - enabling GPU-accelerated JPEG/JPEG2000 decoding")
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
