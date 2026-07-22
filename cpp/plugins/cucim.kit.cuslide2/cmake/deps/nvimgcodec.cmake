#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::nvimgcodec)
    set(NVIMGCODEC_LIB_PATH "")
    set(NVIMGCODEC_INCLUDE_PATH "")

    set(NVIMGCODEC_SONAME_CANDIDATES
        "libnvimgcodec.so.0.9.0"
        "libnvimgcodec.so.0.8.0"
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

    # Try find_package first (works in conda and system installations)
    find_package(nvimgcodec QUIET)

    if (nvimgcodec_FOUND)
        add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::nvimgcodec INTERFACE nvimgcodec::nvimgcodec)
        message(STATUS "✓ nvImageCodec found via find_package")
    else()
        # Manual detection: try conda environment, Python site-packages, and system paths
        if (DEFINED ENV{CONDA_PREFIX})
            # Try native conda package (libnvimgcodec-dev)
            set(CONDA_NATIVE_ROOT "$ENV{CONDA_PREFIX}")
            if(EXISTS "${CONDA_NATIVE_ROOT}/include/nvimgcodec.h")
                set(NVIMGCODEC_INCLUDE_PATH "${CONDA_NATIVE_ROOT}/include/")
                _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${CONDA_NATIVE_ROOT}/lib")
            endif()

            # Fallback: try Python site-packages in conda environment
            if(NOT NVIMGCODEC_LIB_PATH)
                foreach(PY_VER "3.13" "3.12" "3.11" "3.10" "3.9")
                    set(CONDA_PYTHON_ROOT "$ENV{CONDA_PREFIX}/lib/python${PY_VER}/site-packages/nvidia/nvimgcodec")
                    if(EXISTS "${CONDA_PYTHON_ROOT}/include/nvimgcodec.h")
                        set(NVIMGCODEC_INCLUDE_PATH "${CONDA_PYTHON_ROOT}/include/")
                        # Check conda package layout first, then pip-style layout.
                        _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib")
                        if(NOT NVIMGCODEC_LIB_PATH)
                            _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}")
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
                # Try user site-packages first (pip install --user)
                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getusersitepackages())"
                    OUTPUT_VARIABLE PYTHON_USER_SITE_PACKAGES
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )

                # Then try system site-packages
                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
                    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )

                # Check user site-packages first
                if(PYTHON_USER_SITE_PACKAGES)
                    set(NVIMGCODEC_PYTHON_ROOT "${PYTHON_USER_SITE_PACKAGES}/nvidia/nvimgcodec")
                    if(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/include/nvimgcodec.h")
                        set(NVIMGCODEC_INCLUDE_PATH "${NVIMGCODEC_PYTHON_ROOT}/include/")
                        _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib")
                        if(NOT NVIMGCODEC_LIB_PATH)
                            _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}")
                        endif()
                    endif()
                endif()

                # If not found in user site-packages, check system site-packages
                if(NOT NVIMGCODEC_LIB_PATH AND PYTHON_SITE_PACKAGES)
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
                _cucim_find_nvimgcodec_library(NVIMGCODEC_LIB_PATH "/usr/lib64")
            endif()
            if(NVIMGCODEC_LIB_PATH)
                set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
            endif()
        endif()

        # Create target only if nvImageCodec was found
        if(NVIMGCODEC_LIB_PATH AND EXISTS "${NVIMGCODEC_LIB_PATH}")
            add_library(deps::nvimgcodec SHARED IMPORTED GLOBAL)
            set_target_properties(deps::nvimgcodec PROPERTIES
                IMPORTED_LOCATION "${NVIMGCODEC_LIB_PATH}"
                INTERFACE_INCLUDE_DIRECTORIES "${NVIMGCODEC_INCLUDE_PATH}"
            )
            message(STATUS "✓ nvImageCodec found:")
            message(STATUS "  Library: ${NVIMGCODEC_LIB_PATH}")
            message(STATUS "  Headers: ${NVIMGCODEC_INCLUDE_PATH}")

            set(NVIMGCODEC_INCLUDE_PATH ${NVIMGCODEC_INCLUDE_PATH} CACHE INTERNAL "" FORCE)
            set(NVIMGCODEC_LIB_PATH ${NVIMGCODEC_LIB_PATH} CACHE INTERNAL "" FORCE)
            mark_as_advanced(NVIMGCODEC_INCLUDE_PATH NVIMGCODEC_LIB_PATH)
        else()
            message(STATUS "✗ nvImageCodec not found - GPU acceleration disabled")
            message(STATUS "To enable nvImageCodec support:")
            message(STATUS "  Option 1 (conda): conda install libnvimgcodec-dev -c conda-forge")
            message(STATUS "  Option 2 (pip):   pip install nvidia-nvimgcodec-cu12[all]")
        endif()
    endif()
endif ()
