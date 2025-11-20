#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::nvimgcodec)
    if (DEFINED ENV{CONDA_PREFIX})
        # Try to find nvImageCodec in conda environment first
        find_package(nvimgcodec QUIET)

        if (nvimgcodec_FOUND)
            add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
            target_link_libraries(deps::nvimgcodec INTERFACE nvimgcodec::nvimgcodec)
            message(STATUS "✓ nvImageCodec found via find_package in conda environment")
        else()
            # Manual detection in conda environment
            set(NVIMGCODEC_LIB_PATH "")
            set(NVIMGCODEC_INCLUDE_PATH "")

            # Try native conda package first (libnvimgcodec-dev)
            set(CONDA_NATIVE_ROOT "$ENV{CONDA_PREFIX}")
            if(EXISTS "${CONDA_NATIVE_ROOT}/include/nvimgcodec.h")
                set(NVIMGCODEC_INCLUDE_PATH "${CONDA_NATIVE_ROOT}/include/")
                if(EXISTS "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so.0")
                    set(NVIMGCODEC_LIB_PATH "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so.0")
                elseif(EXISTS "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so")
                    set(NVIMGCODEC_LIB_PATH "${CONDA_NATIVE_ROOT}/lib/libnvimgcodec.so")
                endif()
            else()
                # Fallback: try Python site-packages in conda environment
                foreach(PY_VER "3.13" "3.12" "3.11" "3.10" "3.9")
                    set(CONDA_PYTHON_ROOT "$ENV{CONDA_PREFIX}/lib/python${PY_VER}/site-packages/nvidia/nvimgcodec")
                    if(EXISTS "${CONDA_PYTHON_ROOT}/include/nvimgcodec.h")
                        set(NVIMGCODEC_INCLUDE_PATH "${CONDA_PYTHON_ROOT}/include/")
                        # Check for library in lib/ subdirectory first (conda package structure)
                        if(EXISTS "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                        elseif(EXISTS "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so")
                        # Check for library directly in nvimgcodec directory (pip package structure)
                        elseif(EXISTS "${CONDA_PYTHON_ROOT}/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/libnvimgcodec.so.0")
                        elseif(EXISTS "${CONDA_PYTHON_ROOT}/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/libnvimgcodec.so")
                        endif()
                        break()
                    endif()
                endforeach()
            endif()

            if(NVIMGCODEC_LIB_PATH AND EXISTS "${NVIMGCODEC_LIB_PATH}")
                add_library(deps::nvimgcodec SHARED IMPORTED GLOBAL)
                set_target_properties(deps::nvimgcodec PROPERTIES
                    IMPORTED_LOCATION "${NVIMGCODEC_LIB_PATH}"
                    INTERFACE_INCLUDE_DIRECTORIES "${NVIMGCODEC_INCLUDE_PATH}"
                )
                message(STATUS "✓ nvImageCodec found in conda environment:")
                message(STATUS "  Library: ${NVIMGCODEC_LIB_PATH}")
                message(STATUS "  Headers: ${NVIMGCODEC_INCLUDE_PATH}")

                set(NVIMGCODEC_INCLUDE_PATH ${NVIMGCODEC_INCLUDE_PATH} CACHE INTERNAL "" FORCE)
                set(NVIMGCODEC_LIB_PATH ${NVIMGCODEC_LIB_PATH} CACHE INTERNAL "" FORCE)
                mark_as_advanced(NVIMGCODEC_INCLUDE_PATH NVIMGCODEC_LIB_PATH)
            else()
                # Create dummy target
                add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
                message(STATUS "✗ nvImageCodec not found in conda environment - GPU acceleration disabled")
                message(STATUS "To enable nvImageCodec support:")
                message(STATUS "  Option 1 (conda): conda install libnvimgcodec-dev -c conda-forge")
                message(STATUS "  Option 2 (pip):   pip install nvidia-nvimgcodec-cu12[all]")
            endif()
        endif()
    else ()
        # Fallback to manual detection outside conda environment
        message(STATUS "Not in conda environment - attempting manual nvImageCodec detection...")

        # Try find_package first
        find_package(nvimgcodec QUIET)

        if(nvimgcodec_FOUND)
            add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
            target_link_libraries(deps::nvimgcodec INTERFACE nvimgcodec::nvimgcodec)
            message(STATUS "✓ nvImageCodec found via find_package")
        else()
            # Try Python site-packages detection
            find_package(Python3 COMPONENTS Interpreter)
            set(NVIMGCODEC_LIB_PATH "")
            set(NVIMGCODEC_INCLUDE_PATH "")

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
                        # Check for library in lib/ subdirectory first (conda package structure)
                        if(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                        elseif(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so")
                        # Check for library directly in nvimgcodec directory (pip package structure)
                        elseif(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so.0")
                        elseif(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so")
                        endif()
                    endif()
                endif()

                # If not found in user site-packages, check system site-packages
                if(NOT NVIMGCODEC_LIB_PATH AND PYTHON_SITE_PACKAGES)
                    set(NVIMGCODEC_PYTHON_ROOT "${PYTHON_SITE_PACKAGES}/nvidia/nvimgcodec")
                    if(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/include/nvimgcodec.h")
                        set(NVIMGCODEC_INCLUDE_PATH "${NVIMGCODEC_PYTHON_ROOT}/include/")
                        # Check for library in lib/ subdirectory first (conda package structure)
                        if(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                        elseif(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so")
                        # Check for library directly in nvimgcodec directory (pip package structure)
                        elseif(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so.0")
                        elseif(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/libnvimgcodec.so")
                        endif()
                    endif()
                endif()
            endif()

            # System-wide installation fallback
            if(NOT NVIMGCODEC_LIB_PATH)
                if(EXISTS /usr/lib/x86_64-linux-gnu/libnvimgcodec.so.0)
                    set(NVIMGCODEC_LIB_PATH /usr/lib/x86_64-linux-gnu/libnvimgcodec.so.0)
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
                elseif(EXISTS /usr/lib/aarch64-linux-gnu/libnvimgcodec.so.0)
                    set(NVIMGCODEC_LIB_PATH /usr/lib/aarch64-linux-gnu/libnvimgcodec.so.0)
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
                elseif(EXISTS /usr/lib64/libnvimgcodec.so.0)
                    set(NVIMGCODEC_LIB_PATH /usr/lib64/libnvimgcodec.so.0)
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
                endif()
            endif()

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
                # Create dummy target
                add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
                message(STATUS "✗ nvImageCodec not found - GPU acceleration disabled")
                message(STATUS "To enable nvImageCodec support:")
                message(STATUS "  Option 1 (conda): conda install libnvimgcodec-dev -c conda-forge")
                message(STATUS "  Option 2 (pip):   pip install nvidia-nvimgcodec-cu12[all]")
            endif()
        endif()
    endif ()
endif ()
