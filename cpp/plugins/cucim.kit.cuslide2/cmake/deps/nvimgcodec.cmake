#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

# nvImageCodec v0.7.0 internal release configuration
# ===================================================
#
# This cmake module configures nvImageCodec for cuslide2. It supports:
#   1. Pre-installed packages via NVIMGCODEC_ROOT
#   2. Download from URL via NVIMGCODEC_URL  
#   3. Auto-detection in conda/pip/system paths
#
# For internal release v0.7.0, use one of these options:
#
# Option A - Specify local installation path (CUDA 12):
#   cmake -DNVIMGCODEC_ROOT=/home/cdinea/Downloads/cucim_pr3/nvimgcodec/12 ..
#
# Option B - Specify local installation path (CUDA 13):
#   cmake -DNVIMGCODEC_ROOT=/home/cdinea/Downloads/cucim_pr3/nvimgcodec/13 ..
#
# Option C - Use auto-detection with NVIMGCODEC_CUDA_VERSION:
#   cmake -DNVIMGCODEC_DIR=/home/cdinea/Downloads/cucim_pr3/nvimgcodec -DNVIMGCODEC_CUDA_VERSION=12 ..
#
# Available packages for v0.7.0 Build 11:
#   C Packages (CUDA 12.9 and 13.0):
#     - linux-x86_64, linux-sbsa, linux-aarch64 (12.9 only), windows-x86_64
#   Python Packages:
#     - CUDA 12 (12.9): linux-aarch64, linux-sbsa, linux-x86_64, windows-x86_64
#     - CUDA 13 (13.0): linux-sbsa, linux-x86_64, windows-x86_64

set(NVIMGCODEC_VERSION "0.7.0" CACHE STRING "nvImageCodec version to use")
set(NVIMGCODEC_ROOT "" CACHE PATH "Path to nvImageCodec installation directory (e.g., /path/to/nvimgcodec/12)")
set(NVIMGCODEC_DIR "" CACHE PATH "Path to nvImageCodec parent directory containing CUDA version subdirs")
set(NVIMGCODEC_CUDA_VERSION "" CACHE STRING "CUDA version to use (12 or 13) when NVIMGCODEC_DIR is set")
set(NVIMGCODEC_URL "" CACHE STRING "URL to download nvImageCodec tarball from internal release")

# Default nvimgcodec location for this machine
set(NVIMGCODEC_DEFAULT_DIR "/home/cdinea/Downloads/cucim_pr3/nvimgcodec")

if (NOT TARGET deps::nvimgcodec)
    set(NVIMGCODEC_LIB_PATH "")
    set(NVIMGCODEC_INCLUDE_PATH "")
    set(NVIMGCODEC_EXTENSIONS_PATH "")
    set(NVIMGCODEC_FOUND FALSE)

    message(STATUS "")
    message(STATUS "=== nvImageCodec v${NVIMGCODEC_VERSION} Configuration ===")

    # =========================================================================
    # Determine the actual root path to use
    # =========================================================================
    set(NVIMGCODEC_ACTUAL_ROOT "")
    
    # Priority 1: Direct NVIMGCODEC_ROOT specification
    if(NVIMGCODEC_ROOT AND EXISTS "${NVIMGCODEC_ROOT}")
        set(NVIMGCODEC_ACTUAL_ROOT "${NVIMGCODEC_ROOT}")
        message(STATUS "Using NVIMGCODEC_ROOT: ${NVIMGCODEC_ACTUAL_ROOT}")
    
    # Priority 2: NVIMGCODEC_DIR + CUDA version
    elseif(NVIMGCODEC_DIR AND EXISTS "${NVIMGCODEC_DIR}")
        # Auto-detect CUDA version if not specified
        if(NOT NVIMGCODEC_CUDA_VERSION)
            # Try to detect from CUDAToolkit
            if(CUDAToolkit_VERSION_MAJOR)
                set(NVIMGCODEC_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}")
                message(STATUS "Auto-detected CUDA version: ${NVIMGCODEC_CUDA_VERSION}")
            else()
                set(NVIMGCODEC_CUDA_VERSION "12")
                message(STATUS "Defaulting to CUDA version: ${NVIMGCODEC_CUDA_VERSION}")
            endif()
        endif()
        
        if(EXISTS "${NVIMGCODEC_DIR}/${NVIMGCODEC_CUDA_VERSION}")
            set(NVIMGCODEC_ACTUAL_ROOT "${NVIMGCODEC_DIR}/${NVIMGCODEC_CUDA_VERSION}")
            message(STATUS "Using NVIMGCODEC_DIR with CUDA ${NVIMGCODEC_CUDA_VERSION}: ${NVIMGCODEC_ACTUAL_ROOT}")
        endif()
    
    # Priority 3: Check default location
    elseif(EXISTS "${NVIMGCODEC_DEFAULT_DIR}")
        # Auto-detect CUDA version
        if(NOT NVIMGCODEC_CUDA_VERSION)
            if(CUDAToolkit_VERSION_MAJOR)
                set(NVIMGCODEC_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}")
            else()
                set(NVIMGCODEC_CUDA_VERSION "12")
            endif()
        endif()
        
        if(EXISTS "${NVIMGCODEC_DEFAULT_DIR}/${NVIMGCODEC_CUDA_VERSION}")
            set(NVIMGCODEC_ACTUAL_ROOT "${NVIMGCODEC_DEFAULT_DIR}/${NVIMGCODEC_CUDA_VERSION}")
            message(STATUS "Using default location with CUDA ${NVIMGCODEC_CUDA_VERSION}: ${NVIMGCODEC_ACTUAL_ROOT}")
        endif()
    endif()

    # =========================================================================
    # Method 1: Use CMake config files from the package (preferred)
    # =========================================================================
    if(NVIMGCODEC_ACTUAL_ROOT AND EXISTS "${NVIMGCODEC_ACTUAL_ROOT}/cmake/nvimgcodec/nvimgcodecConfig.cmake")
        message(STATUS "Found nvImageCodec CMake config at: ${NVIMGCODEC_ACTUAL_ROOT}/cmake/nvimgcodec")
        
        # Add to CMAKE_PREFIX_PATH for find_package
        list(APPEND CMAKE_PREFIX_PATH "${NVIMGCODEC_ACTUAL_ROOT}/cmake")
        
        find_package(nvimgcodec CONFIG QUIET
            PATHS "${NVIMGCODEC_ACTUAL_ROOT}/cmake"
            NO_DEFAULT_PATH
        )
        
        if(nvimgcodec_FOUND)
            # The nvimgcodec CMake config sets these variables:
            #   nvimgcodec_INCLUDE_DIR, nvimgcodec_LIB_DIR, nvimgcodec_EXTENSIONS_DIR
            # But it doesn't set INTERFACE_INCLUDE_DIRECTORIES on the target, so we must do it
            
            # Get library path from target
            get_target_property(_nvimgcodec_loc nvimgcodec::nvimgcodec IMPORTED_LOCATION_RELEASE)
            if(NOT _nvimgcodec_loc)
                get_target_property(_nvimgcodec_loc nvimgcodec::nvimgcodec IMPORTED_LOCATION)
            endif()
            
            # Use nvimgcodec_INCLUDE_DIR from the config (not from target property)
            set(NVIMGCODEC_LIB_PATH "${_nvimgcodec_loc}")
            set(NVIMGCODEC_INCLUDE_PATH "${nvimgcodec_INCLUDE_DIR}")
            set(NVIMGCODEC_EXTENSIONS_PATH "${nvimgcodec_EXTENSIONS_DIR}")
            
            # Add include directory to the nvimgcodec target (it's missing from the CMake config)
            set_target_properties(nvimgcodec::nvimgcodec PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${nvimgcodec_INCLUDE_DIR}"
            )
            
            # Create our wrapper target
            add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
            target_link_libraries(deps::nvimgcodec INTERFACE nvimgcodec::nvimgcodec)
            
            set(NVIMGCODEC_FOUND TRUE)
            
            message(STATUS "✓ nvImageCodec v${NVIMGCODEC_VERSION} found via CMake config")
            message(STATUS "  Include dir: ${nvimgcodec_INCLUDE_DIR}")
        endif()
    endif()

    # =========================================================================
    # Method 2: Manual detection in specified root
    # =========================================================================
    if(NOT NVIMGCODEC_FOUND AND NVIMGCODEC_ACTUAL_ROOT)
        message(STATUS "Searching for nvImageCodec in: ${NVIMGCODEC_ACTUAL_ROOT}")
        
        # Check for header file
        if(EXISTS "${NVIMGCODEC_ACTUAL_ROOT}/include/nvimgcodec.h")
            set(NVIMGCODEC_INCLUDE_PATH "${NVIMGCODEC_ACTUAL_ROOT}/include")
        endif()
        
        # Check for library file (lib64 first, then lib)
        foreach(LIB_NAME "libnvimgcodec.so.0" "libnvimgcodec.so" "libnvimgcodec.so.${NVIMGCODEC_VERSION}")
            foreach(LIB_DIR "lib64" "lib" "")
                if(LIB_DIR)
                    set(LIB_CHECK_PATH "${NVIMGCODEC_ACTUAL_ROOT}/${LIB_DIR}/${LIB_NAME}")
                else()
                    set(LIB_CHECK_PATH "${NVIMGCODEC_ACTUAL_ROOT}/${LIB_NAME}")
                endif()
                if(EXISTS "${LIB_CHECK_PATH}")
                    set(NVIMGCODEC_LIB_PATH "${LIB_CHECK_PATH}")
                    break()
                endif()
            endforeach()
            if(NVIMGCODEC_LIB_PATH)
                break()
            endif()
        endforeach()
        
        # Check for extensions directory
        if(EXISTS "${NVIMGCODEC_ACTUAL_ROOT}/extensions")
            set(NVIMGCODEC_EXTENSIONS_PATH "${NVIMGCODEC_ACTUAL_ROOT}/extensions")
        endif()
        
        if(NVIMGCODEC_INCLUDE_PATH AND NVIMGCODEC_LIB_PATH)
            set(NVIMGCODEC_FOUND TRUE)
            message(STATUS "✓ nvImageCodec v${NVIMGCODEC_VERSION} found via manual detection")
        endif()
    endif()

    # =========================================================================
    # Method 3: Download from URL (internal release)
    # =========================================================================
    if(NOT NVIMGCODEC_FOUND AND NVIMGCODEC_URL)
        message(STATUS "Downloading nvImageCodec from: ${NVIMGCODEC_URL}")
        
        include(FetchContent)
        
        FetchContent_Declare(
            deps-nvimgcodec
            URL ${NVIMGCODEC_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        
        FetchContent_GetProperties(deps-nvimgcodec)
        if(NOT deps-nvimgcodec_POPULATED)
            message(STATUS "Fetching nvImageCodec v${NVIMGCODEC_VERSION}...")
            FetchContent_Populate(deps-nvimgcodec)
            message(STATUS "Fetching nvImageCodec v${NVIMGCODEC_VERSION} - done")
        endif()
        
        set(NVIMGCODEC_DOWNLOAD_DIR "${deps-nvimgcodec_SOURCE_DIR}")
        
        # Search for headers and library in downloaded content
        if(EXISTS "${NVIMGCODEC_DOWNLOAD_DIR}/include/nvimgcodec.h")
            set(NVIMGCODEC_INCLUDE_PATH "${NVIMGCODEC_DOWNLOAD_DIR}/include")
        endif()
        
        foreach(LIB_NAME "libnvimgcodec.so.0" "libnvimgcodec.so")
            foreach(LIB_DIR "lib64" "lib" "")
                if(LIB_DIR)
                    set(LIB_CHECK_PATH "${NVIMGCODEC_DOWNLOAD_DIR}/${LIB_DIR}/${LIB_NAME}")
                else()
                    set(LIB_CHECK_PATH "${NVIMGCODEC_DOWNLOAD_DIR}/${LIB_NAME}")
                endif()
                if(EXISTS "${LIB_CHECK_PATH}")
                    set(NVIMGCODEC_LIB_PATH "${LIB_CHECK_PATH}")
                    break()
                endif()
            endforeach()
            if(NVIMGCODEC_LIB_PATH)
                break()
            endif()
        endforeach()
        
        if(EXISTS "${NVIMGCODEC_DOWNLOAD_DIR}/extensions")
            set(NVIMGCODEC_EXTENSIONS_PATH "${NVIMGCODEC_DOWNLOAD_DIR}/extensions")
        endif()
        
        if(NVIMGCODEC_INCLUDE_PATH AND NVIMGCODEC_LIB_PATH)
            set(NVIMGCODEC_FOUND TRUE)
            message(STATUS "✓ nvImageCodec v${NVIMGCODEC_VERSION} downloaded and extracted")
        else()
            message(WARNING "Downloaded nvImageCodec but couldn't find library or headers")
            message(WARNING "  Download dir: ${NVIMGCODEC_DOWNLOAD_DIR}")
        endif()
    endif()

    # =========================================================================
    # Method 4: Try find_package (works in conda and system installations)
    # =========================================================================
    if(NOT NVIMGCODEC_FOUND)
        find_package(nvimgcodec ${NVIMGCODEC_VERSION} QUIET CONFIG)
        
        if(nvimgcodec_FOUND)
            add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
            target_link_libraries(deps::nvimgcodec INTERFACE nvimgcodec::nvimgcodec)
            message(STATUS "✓ nvImageCodec found via find_package (version: ${nvimgcodec_VERSION})")
            set(NVIMGCODEC_FOUND TRUE)
        endif()
    endif()

    # =========================================================================
    # Method 5: Manual detection in conda/pip/system paths
    # =========================================================================
    if(NOT NVIMGCODEC_FOUND)
        # Try conda environment
        if(DEFINED ENV{CONDA_PREFIX})
            # Try native conda package (libnvimgcodec-dev)
            set(CONDA_NATIVE_ROOT "$ENV{CONDA_PREFIX}")
            if(EXISTS "${CONDA_NATIVE_ROOT}/include/nvimgcodec.h")
                set(NVIMGCODEC_INCLUDE_PATH "${CONDA_NATIVE_ROOT}/include")
                if(EXISTS "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so.0")
                    set(NVIMGCODEC_LIB_PATH "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so.0")
                elseif(EXISTS "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so")
                    set(NVIMGCODEC_LIB_PATH "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so")
                endif()
            endif()

            # Fallback: try Python site-packages in conda environment
            if(NOT NVIMGCODEC_LIB_PATH)
                foreach(PY_VER "3.13" "3.12" "3.11" "3.10" "3.9")
                    set(CONDA_PYTHON_ROOT "$ENV{CONDA_PREFIX}/lib/python${PY_VER}/site-packages/nvidia/nvimgcodec")
                    if(EXISTS "${CONDA_PYTHON_ROOT}/include/nvimgcodec.h")
                        set(NVIMGCODEC_INCLUDE_PATH "${CONDA_PYTHON_ROOT}/include")
                        if(EXISTS "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                        elseif(EXISTS "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so")
                        elseif(EXISTS "${CONDA_PYTHON_ROOT}/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/libnvimgcodec.so.0")
                        elseif(EXISTS "${CONDA_PYTHON_ROOT}/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/libnvimgcodec.so")
                        endif()
                        if(NVIMGCODEC_LIB_PATH)
                            break()
                        endif()
                    endif()
                endforeach()
            endif()
        endif()

        # Try Python site-packages (outside conda or as additional fallback)
        if(NOT NVIMGCODEC_LIB_PATH)
            find_package(Python3 COMPONENTS Interpreter QUIET)
            if(Python3_FOUND)
                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getusersitepackages())"
                    OUTPUT_VARIABLE PYTHON_USER_SITE_PACKAGES
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )
                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
                    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )

                foreach(SITE_PKG_DIR ${PYTHON_USER_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES})
                    if(SITE_PKG_DIR)
                        set(NVIMGCODEC_PYTHON_ROOT "${SITE_PKG_DIR}/nvidia/nvimgcodec")
                        if(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/include/nvimgcodec.h")
                            set(NVIMGCODEC_INCLUDE_PATH "${NVIMGCODEC_PYTHON_ROOT}/include")
                            foreach(LIB_SUBDIR "lib" "")
                                if(LIB_SUBDIR)
                                    set(LIB_BASE "${NVIMGCODEC_PYTHON_ROOT}/${LIB_SUBDIR}")
                                else()
                                    set(LIB_BASE "${NVIMGCODEC_PYTHON_ROOT}")
                                endif()
                                if(EXISTS "${LIB_BASE}/libnvimgcodec.so.0")
                                    set(NVIMGCODEC_LIB_PATH "${LIB_BASE}/libnvimgcodec.so.0")
                                    break()
                                elseif(EXISTS "${LIB_BASE}/libnvimgcodec.so")
                                    set(NVIMGCODEC_LIB_PATH "${LIB_BASE}/libnvimgcodec.so")
                                    break()
                                endif()
                            endforeach()
                            if(NVIMGCODEC_LIB_PATH)
                                break()
                            endif()
                        endif()
                    endif()
                endforeach()
            endif()
        endif()

        # System-wide installation fallback
        if(NOT NVIMGCODEC_LIB_PATH)
            foreach(SYS_LIB_DIR "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu" "/usr/lib64")
                if(EXISTS "${SYS_LIB_DIR}/libnvimgcodec.so.0")
                    set(NVIMGCODEC_LIB_PATH "${SYS_LIB_DIR}/libnvimgcodec.so.0")
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include")
                    break()
                endif()
            endforeach()
        endif()

        if(NVIMGCODEC_LIB_PATH AND EXISTS "${NVIMGCODEC_LIB_PATH}")
            set(NVIMGCODEC_FOUND TRUE)
        endif()
    endif()

    # =========================================================================
    # Create the target if nvImageCodec was found
    # =========================================================================
    if(NVIMGCODEC_FOUND AND NOT TARGET deps::nvimgcodec)
        if(NVIMGCODEC_LIB_PATH AND EXISTS "${NVIMGCODEC_LIB_PATH}")
            add_library(deps::nvimgcodec SHARED IMPORTED GLOBAL)
            set_target_properties(deps::nvimgcodec PROPERTIES
                IMPORTED_LOCATION "${NVIMGCODEC_LIB_PATH}"
                INTERFACE_INCLUDE_DIRECTORIES "${NVIMGCODEC_INCLUDE_PATH}"
            )
        endif()
    endif()
    
    if(NVIMGCODEC_FOUND)
        message(STATUS "✓ nvImageCodec v${NVIMGCODEC_VERSION} configured successfully:")
        message(STATUS "  Library:    ${NVIMGCODEC_LIB_PATH}")
        message(STATUS "  Headers:    ${NVIMGCODEC_INCLUDE_PATH}")
        if(NVIMGCODEC_EXTENSIONS_PATH)
            message(STATUS "  Extensions: ${NVIMGCODEC_EXTENSIONS_PATH}")
        endif()

        # Cache the paths
        set(NVIMGCODEC_INCLUDE_PATH ${NVIMGCODEC_INCLUDE_PATH} CACHE INTERNAL "" FORCE)
        set(NVIMGCODEC_LIB_PATH ${NVIMGCODEC_LIB_PATH} CACHE INTERNAL "" FORCE)
        set(NVIMGCODEC_EXTENSIONS_PATH ${NVIMGCODEC_EXTENSIONS_PATH} CACHE INTERNAL "" FORCE)
        mark_as_advanced(NVIMGCODEC_INCLUDE_PATH NVIMGCODEC_LIB_PATH NVIMGCODEC_EXTENSIONS_PATH)
        
        # Export extensions path as compile definition (useful at runtime)
        if(NVIMGCODEC_EXTENSIONS_PATH)
            add_compile_definitions(NVIMGCODEC_EXTENSIONS_DIR="${NVIMGCODEC_EXTENSIONS_PATH}")
        endif()
    else()
        message(STATUS "")
        message(STATUS "✗ nvImageCodec v${NVIMGCODEC_VERSION} not found - GPU acceleration disabled")
        message(STATUS "")
        message(STATUS "To install nvImageCodec v${NVIMGCODEC_VERSION} (internal release Build 11):")
        message(STATUS "")
        message(STATUS "  Option 1 - Use downloaded packages (CUDA 12):")
        message(STATUS "    cmake -DNVIMGCODEC_ROOT=/home/cdinea/Downloads/cucim_pr3/nvimgcodec/12 ..")
        message(STATUS "")
        message(STATUS "  Option 2 - Use downloaded packages (CUDA 13):")
        message(STATUS "    cmake -DNVIMGCODEC_ROOT=/home/cdinea/Downloads/cucim_pr3/nvimgcodec/13 ..")
        message(STATUS "")
        message(STATUS "  Option 3 - Auto-detect CUDA version:")
        message(STATUS "    cmake -DNVIMGCODEC_DIR=/home/cdinea/Downloads/cucim_pr3/nvimgcodec ..")
        message(STATUS "")
        message(STATUS "  Available platforms for v${NVIMGCODEC_VERSION}:")
        message(STATUS "    C Packages:      linux-x86_64, linux-sbsa, linux-aarch64 (12.9), windows-x86_64")
        message(STATUS "    Python (CUDA 12): linux-x86_64, linux-sbsa, linux-aarch64, windows-x86_64")
        message(STATUS "    Python (CUDA 13): linux-x86_64, linux-sbsa, windows-x86_64")
        message(STATUS "")
    endif()
    
    message(STATUS "=== End nvImageCodec Configuration ===")
    message(STATUS "")
endif()
