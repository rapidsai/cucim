#
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if (NOT TARGET deps::nvimgcodec)
    # Option to automatically install nvImageCodec via conda
    option(AUTO_INSTALL_NVIMGCODEC "Automatically install nvImageCodec via conda" ON)
    set(NVIMGCODEC_VERSION "0.6.0" CACHE STRING "nvImageCodec version to install")

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
                # Auto-install if enabled and not found
                if(AUTO_INSTALL_NVIMGCODEC)
                    message(STATUS "nvImageCodec not found in conda environment - attempting automatic installation...")
                    
                    # Find conda executable
                    find_program(MICROMAMBA_EXECUTABLE 
                        NAMES micromamba
                        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin
                              ${CMAKE_CURRENT_SOURCE_DIR}/../../bin
                              $ENV{HOME}/micromamba/bin
                              /usr/local/bin
                              /opt/conda/bin
                        DOC "Path to micromamba executable"
                    )
                    
                    find_program(CONDA_EXECUTABLE 
                        NAMES conda mamba
                        PATHS $ENV{HOME}/miniconda3/bin
                              $ENV{HOME}/anaconda3/bin
                              /opt/conda/bin
                              /usr/local/bin
                        DOC "Path to conda/mamba executable"
                    )
                    
                    set(CONDA_CMD "")
                    if(MICROMAMBA_EXECUTABLE)
                        set(CONDA_CMD ${MICROMAMBA_EXECUTABLE})
                        message(STATUS "Using micromamba: ${MICROMAMBA_EXECUTABLE}")
                    elseif(CONDA_EXECUTABLE)
                        set(CONDA_CMD ${CONDA_EXECUTABLE})
                        message(STATUS "Using conda/mamba: ${CONDA_EXECUTABLE}")
                    endif()
                    
                    if(CONDA_CMD)
                        message(STATUS "Installing nvImageCodec ${NVIMGCODEC_VERSION}...")
                        execute_process(
                            COMMAND ${CONDA_CMD} install 
                                libnvimgcodec-dev=${NVIMGCODEC_VERSION} 
                                libnvimgcodec0=${NVIMGCODEC_VERSION} 
                                -c conda-forge -y
                            RESULT_VARIABLE CONDA_INSTALL_RESULT
                            OUTPUT_QUIET
                            ERROR_QUIET
                            TIMEOUT 300
                        )
                        
                        if(CONDA_INSTALL_RESULT EQUAL 0)
                            message(STATUS "✓ Successfully installed nvImageCodec ${NVIMGCODEC_VERSION}")
                            # Retry detection after installation
                            if(EXISTS "$ENV{CONDA_PREFIX}/include/nvimgcodec.h" AND 
                               EXISTS "$ENV{CONDA_PREFIX}/lib/libnvimgcodec.so.0")
                                add_library(deps::nvimgcodec SHARED IMPORTED GLOBAL)
                                set_target_properties(deps::nvimgcodec PROPERTIES
                                    IMPORTED_LOCATION "$ENV{CONDA_PREFIX}/lib/libnvimgcodec.so.0"
                                    INTERFACE_INCLUDE_DIRECTORIES "$ENV{CONDA_PREFIX}/include/"
                                )
                                message(STATUS "✓ nvImageCodec configured after installation")
                            endif()
                        else()
                            message(WARNING "✗ Failed to install nvImageCodec - creating dummy target")
                            add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
                        endif()
                    else()
                        message(WARNING "No conda manager found - creating dummy target")
                        add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
                    endif()
                else()
                    message(STATUS "nvImageCodec not found and auto-install disabled - creating dummy target")
                    add_library(deps::nvimgcodec INTERFACE IMPORTED GLOBAL)
                endif()
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
                message(STATUS "  Option 1 (conda): micromamba install libnvimgcodec-dev -c conda-forge")
                message(STATUS "  Option 2 (pip):   pip install nvidia-nvimgcodec-cu12[all]")
                message(STATUS "  Option 3 (cmake): cmake -DAUTO_INSTALL_NVIMGCODEC=ON ..")
            endif()
        endif()
    endif ()
endif ()
