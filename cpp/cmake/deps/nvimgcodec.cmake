#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::nvimgcodec)
    # Option to automatically install nvImageCodec via conda
    option(AUTO_INSTALL_NVIMGCODEC "Automatically install nvImageCodec via conda" ON)
    set(NVIMGCODEC_VERSION "0.6.0" CACHE STRING "nvImageCodec version to install")

    # Automatic installation logic
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
                else()
                    message(WARNING "✗ Failed to install nvImageCodec via ${CONDA_TYPE}")
                    message(WARNING "Error: ${CONDA_INSTALL_ERROR}")

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
        else()
            message(STATUS "No conda/micromamba found - skipping automatic installation")
        endif()
    endif()

    # First try to find it as a package
    find_package(nvimgcodec QUIET)

    if(nvimgcodec_FOUND)
        # Use the found package
        add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::nvimgcodec INTERFACE nvimgcodec::nvimgcodec)
        message(STATUS "✓ nvImageCodec found via find_package")
    else()
        # Manual detection in various environments
        set(NVIMGCODEC_LIB_PATH "")
        set(NVIMGCODEC_INCLUDE_PATH "")

        # Try conda environment detection (both Python packages and native packages)
        if(DEFINED ENV{CONDA_BUILD})
            # Conda build environment
            set(NVIMGCODEC_LIB_PATH "$ENV{PREFIX}/lib/libnvimgcodec.so.0")
            set(NVIMGCODEC_INCLUDE_PATH "$ENV{PREFIX}/include/")
            if(NOT EXISTS "${NVIMGCODEC_LIB_PATH}")
                set(NVIMGCODEC_LIB_PATH "$ENV{PREFIX}/lib/libnvimgcodec.so")
            endif()
        elseif(DEFINED ENV{CONDA_PREFIX})
            # Active conda environment - try native package first
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
                        if(EXISTS "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                        elseif(EXISTS "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${CONDA_PYTHON_ROOT}/lib/libnvimgcodec.so")
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
                        if(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so.0")
                        elseif(EXISTS "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so")
                            set(NVIMGCODEC_LIB_PATH "${NVIMGCODEC_PYTHON_ROOT}/lib/libnvimgcodec.so")
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
                elseif(EXISTS /usr/lib64/libnvimgcodec.so.0) # CentOS (x86_64)
                    set(NVIMGCODEC_LIB_PATH /usr/lib64/libnvimgcodec.so.0)
                    set(NVIMGCODEC_INCLUDE_PATH "/usr/include/")
                endif()
            endif()
        endif()

        # Create the target if we found the library
        if(NVIMGCODEC_LIB_PATH AND EXISTS "${NVIMGCODEC_LIB_PATH}")
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
            message(STATUS "  Option 1 (conda): micromamba install libnvimgcodec-dev -c conda-forge")
            message(STATUS "  Option 2 (pip):   pip install nvidia-nvimgcodec-cu12[all]")
            message(STATUS "  Option 3 (cmake): cmake -DAUTO_INSTALL_NVIMGCODEC=ON ..")
        endif()
    endif()
endif()
