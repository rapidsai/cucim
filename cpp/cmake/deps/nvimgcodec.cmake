#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::nvimgcodec)
    # Packaging pins nvImageCodec to >=0.9.0,<0.10.0 and the vendored headers
    # declare 0.9, so only 0.9 is supported here.  The unversioned entries are
    # kept for wheel and system layouts that ship just the symlinks.
    set(NVIMGCODEC_SONAME_CANDIDATES
        "libnvimgcodec.so.0.9.0"
        "libnvimgcodec.so.0"
        "libnvimgcodec.so")

    function(_cucim_find_nvimgcodec_library out_var root_dir)
        foreach(_soname IN LISTS NVIMGCODEC_SONAME_CANDIDATES)
            if(EXISTS "${root_dir}/${_soname}")
                set(${out_var} "${root_dir}/${_soname}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
        set(${out_var} "" PARENT_SCOPE)
    endfunction()

    # The unversioned "libnvimgcodec.so.0" / "libnvimgcodec.so" candidates above
    # resolve to whatever major-0 build is installed, so a version below the
    # required minimum would otherwise be picked up silently.  That API is not
    # compatible and there are no compile-time version guards, so fail at
    # configure time instead of at runtime.
    function(_cucim_check_nvimgcodec_version include_dir)
        set(_version_header "${include_dir}/nvimgcodec_version.h")
        if(NOT EXISTS "${_version_header}")
            message(WARNING
                "nvImageCodec: nvimgcodec_version.h not found under '${include_dir}' - "
                "skipping version check. cuCIM requires >=0.9.0,<0.10.0.")
            return()
        endif()

        file(STRINGS "${_version_header}" _major_line
             REGEX "^#define[ \t]+NVIMGCODEC_VER_MAJOR[ \t]+[0-9]+")
        file(STRINGS "${_version_header}" _minor_line
             REGEX "^#define[ \t]+NVIMGCODEC_VER_MINOR[ \t]+[0-9]+")
        string(REGEX REPLACE ".*NVIMGCODEC_VER_MAJOR[ \t]+([0-9]+).*" "\\1" _major "${_major_line}")
        string(REGEX REPLACE ".*NVIMGCODEC_VER_MINOR[ \t]+([0-9]+).*" "\\1" _minor "${_minor_line}")

        if(NOT _major MATCHES "^[0-9]+$" OR NOT _minor MATCHES "^[0-9]+$")
            message(WARNING
                "nvImageCodec: could not parse version from '${_version_header}' - "
                "skipping version check.")
            return()
        endif()

        if(_major EQUAL 0 AND _minor LESS 9)
            message(FATAL_ERROR
                "nvImageCodec ${_major}.${_minor} found at '${include_dir}', but cuCIM "
                "requires >=0.9.0 (packaging pins >=0.9.0,<0.10.0). The 0.9 API is not "
                "backward compatible and this build has no compile-time version guards, "
                "so it would fail at runtime. Install libnvimgcodec-dev >=0.9.")
        elseif(_major GREATER 0 OR _minor GREATER_EQUAL 10)
            message(WARNING
                "nvImageCodec ${_major}.${_minor} is outside the pinned range "
                "(>=0.9.0,<0.10.0) and is untested with this cuCIM build.")
        endif()
    endfunction()

    # First try to find it as a package
    find_package(nvimgcodec QUIET)

    if(nvimgcodec_FOUND)
        # Use the found package
        if(DEFINED nvimgcodec_VERSION AND nvimgcodec_VERSION VERSION_LESS "0.9.0")
            message(FATAL_ERROR
                "nvImageCodec ${nvimgcodec_VERSION} found via find_package, but cuCIM "
                "requires >=0.9.0 (packaging pins >=0.9.0,<0.10.0).")
        endif()
        add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::nvimgcodec INTERFACE nvimgcodec::nvimgcodec)
        message(STATUS "✓ nvImageCodec found via find_package (version: ${nvimgcodec_VERSION})")
    else()
        # Manual detection in various environments
        set(NVIMGCODEC_LIB_PATH "")
        set(NVIMGCODEC_INCLUDE_PATH "")

        # Try conda environment detection (both Python packages and native packages)
        if(DEFINED ENV{CONDA_BUILD})
            # Conda build environment
            _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "$ENV{PREFIX}/lib")
            set(NVIMGCODEC_INCLUDE_PATH "$ENV{PREFIX}/include/")
        elseif(DEFINED ENV{CONDA_PREFIX})
            # Active conda environment - try native package first
            set(CONDA_NATIVE_ROOT "$ENV{CONDA_PREFIX}")
            if(EXISTS "${CONDA_NATIVE_ROOT}/include/nvimgcodec.h")
                set(NVIMGCODEC_INCLUDE_PATH "${CONDA_NATIVE_ROOT}/include/")
                _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${CONDA_NATIVE_ROOT}/lib")
            else()
                # Fallback: try Python site-packages in conda environment
                foreach(PY_VER "3.13" "3.12" "3.11" "3.10" "3.9")
                    set(CONDA_PYTHON_ROOT "$ENV{CONDA_PREFIX}/lib/python${PY_VER}/site-packages/nvidia/nvimgcodec")
                    if(EXISTS "${CONDA_PYTHON_ROOT}/include/nvimgcodec.h")
                        set(NVIMGCODEC_INCLUDE_PATH "${CONDA_PYTHON_ROOT}/include/")
                        _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib")
                        if(NOT NVIMGCODEC_LIB_PATH)
                            _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}")
                        endif()
                        break()
                    endif()
                endforeach()
            endif()
        else()
            # Try Python site-packages detection
            find_package(Python3 COMPONENTS Interpreter)
            if(Python3_FOUND)
                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
                    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )

                if(PYTHON_SITE_PACKAGES)
                    set(NVIMGCODEC_PYTHON_ROOT "${PYTHON_SITE_PACKAGES}/nvidia/nvimgcodec")
                    if(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/include/nvimgcodec.h")
                        set(NVIMGCODEC_INCLUDE_PATH "${NVIMGCODEC_PYTHON_ROOT}/include/")
                        _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib")
                        if(NOT NVIMGCODEC_LIB_PATH)
                            _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}")
                        endif()
                    endif()
                endif()
            endif()

            # System-wide installation fallback
            if(NOT NVIMGCODEC_LIB_PATH)
                _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "/usr/lib/x86_64-linux-gnu")
                if(NVIMGCODEC_LIB_PATH)
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
                else()
                    _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "/usr/lib/aarch64-linux-gnu")
                endif()
                if(NVIMGCODEC_LIB_PATH)
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
                else()
                    _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "/usr/lib64") # CentOS (x86_64)
                endif()
                if(NVIMGCODEC_LIB_PATH)
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
                endif()
            endif()
        endif()

        # Create the target if we found the library
        if(NVIMGCODEC_LIB_PATH AND EXISTS "${NVIMGCODEC_LIB_PATH}")
            _cucim_check_nvimgcodec_version("${NVIMGCODEC_INCLUDE_PATH}")
            add_library(deps::nvimgcodec SHARED IMPORTED GLOBAL)
            set_target_properties(deps::nvimgcodec PROPERTIES
                IMPORTED_LOCATION "${NVIMGCODEC_LIB_PATH}"
                INTERFACE_INCLUDE_DIRECTORIES "${NVIMGCODEC_INCLUDE_PATH}"
            )
            message(STATUS "✓ nvImageCodec found:")
            message(STATUS "  Library: ${NVIMGCODEC_LIB_PATH}")
            message(STATUS "  Headers: ${NVIMGCODEC_INCLUDE_PATH}")
        else()
            # Create a dummy target to prevent build failures
            add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
            message(STATUS "✗ nvImageCodec not found - GPU acceleration disabled")
            message(STATUS "To enable nvImageCodec support:")
            message(STATUS "  Option 1 (conda): conda install libnvimgcodec-dev -c conda-forge")
            message(STATUS "  Option 2 (pip):   pip install nvidia-nvimgcodec-cu12[all]")
        endif()
    endif()
endif()
