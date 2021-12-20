#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

################################################################################
# Set cmake policy
################################################################################
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19")
    cmake_policy(SET CMP0110 NEW) # For add_test() to support arbitrary characters in test name
endif()

################################################################################
# Define compile options
################################################################################

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

################################################################################
# Set definitions
################################################################################

################################################################################
# Add library: cucim
################################################################################
add_library(${CUCIM_PACKAGE_NAME}
        src/core/framework.cpp
        include/cucim/cuimage.h
        include/cucim/cache/image_cache.h
        include/cucim/cache/image_cache_config.h
        include/cucim/cache/image_cache_manager.h
        include/cucim/codec/base64.h
        include/cucim/codec/hash_function.h
        include/cucim/codec/methods.h
        include/cucim/concurrent/threadpool.h
        include/cucim/config/config.h
        include/cucim/core/framework.h
        include/cucim/core/plugin.h
        include/cucim/core/plugin_util.h
        include/cucim/core/interface.h
        include/cucim/core/version.h
        include/cucim/cpp20/find_if.h
        include/cucim/dynlib/helper.h
        include/cucim/filesystem/cufile_driver.h
        include/cucim/filesystem/file_handle.h
        include/cucim/filesystem/file_path.h
        include/cucim/io/device.h
        include/cucim/io/device_type.h
        include/cucim/io/format/image_format.h
        include/cucim/loader/batch_data_processor.h
        include/cucim/loader/thread_batch_data_loader.h
        include/cucim/loader/tile_info.h
        include/cucim/logger/logger.h
        include/cucim/logger/timer.h
        include/cucim/macros/defines.h
        include/cucim/memory/dlpack.h
        include/cucim/memory/memory_manager.h
        include/cucim/plugin/image_format.h
        include/cucim/plugin/plugin_config.h
        include/cucim/profiler/nvtx3.h
        include/cucim/profiler/profiler.h
        include/cucim/profiler/profiler_config.h
        include/cucim/util/cuda.h
        include/cucim/util/file.h
        include/cucim/util/platform.h
        include/cucim/3rdparty/dlpack/dlpack.h
        include/cucim/3rdparty/dlpack/dlpackcpp.h
        src/cuimage.cpp
        src/cache/cache_type.cpp
        src/cache/image_cache.cpp
        src/cache/image_cache_config.cpp
        src/cache/image_cache_empty.h
        src/cache/image_cache_empty.cpp
        src/cache/image_cache_manager.cpp
        src/cache/image_cache_per_process.h
        src/cache/image_cache_per_process.cpp
        src/cache/image_cache_shared_memory.h
        src/cache/image_cache_shared_memory.cpp
        src/codec/base64.cpp
        src/concurrent/threadpool.cpp
        src/config/config.cpp
        src/core/cucim_framework.h
        src/core/cucim_framework.cpp
        src/core/cucim_plugin.h
        src/core/cucim_plugin.cpp
        src/core/plugin_manager.h
        src/core/plugin_manager.cpp
        src/core/version.inl
        src/filesystem/cufile_driver.cpp
        src/filesystem/file_handle.cpp
        src/io/device.cpp
        src/io/device_type.cpp
        src/io/format/image_format.cpp
        src/loader/batch_data_processor.cpp
        src/loader/thread_batch_data_loader.cpp
        src/logger/logger.cpp
        src/logger/timer.cpp
        src/memory/memory_manager.cpp
        src/plugin/image_format.cpp
        src/plugin/plugin_config.cpp
        src/profiler/profiler.cpp
        src/profiler/profiler_config.cpp
        src/util/file.cpp
        src/util/platform.cpp)

# Compile options
set_target_properties(${CUCIM_PACKAGE_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
)

# Note: Looks like the following line causes error on CMake 3.18.4 (it is working on 3.18.2). Keeping it for now.
set(CUCIM_REQUIRED_FEATURES cxx_std_17)
target_compile_features(${CUCIM_PACKAGE_NAME} PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(${CUCIM_PACKAGE_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>
        )
target_compile_definitions(${CUCIM_PACKAGE_NAME}
    PUBLIC
        CUCIM_VERSION=${PROJECT_VERSION}
        CUCIM_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUCIM_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUCIM_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUCIM_VERSION_BUILD=${PROJECT_VERSION_BUILD}
        CUCIM_SUPPORT_GDS=$<BOOL:${CUCIM_SUPPORT_GDS}>
        CUCIM_STATIC_GDS=$<BOOL:${CUCIM_STATIC_GDS}>
        CUCIM_SUPPORT_CUDA=$<BOOL:${CUCIM_SUPPORT_CUDA}>
        CUCIM_SUPPORT_NVTX=$<BOOL:${CUCIM_SUPPORT_NVTX}>
        _GLIBCXX_USE_CXX11_ABI=0  # TODO: create two library, one with CXX11 ABI and one without it.
)

# Link libraries
target_link_libraries(${CUCIM_PACKAGE_NAME}
        PUBLIC
            ${CMAKE_DL_LIBS}
            Threads::Threads # -lpthread
            $<BUILD_INTERFACE:deps::fmt>
            $<INSTALL_INTERFACE:cucim::fmt-header-only>
        PRIVATE
            CUDA::cudart
            deps::abseil
            deps::gds
            deps::libcuckoo
            deps::boost-header-only
            deps::json
            deps::nvtx3
            deps::taskflow
        )

if (CUCIM_STATIC_GDS)
    target_link_libraries(${CUCIM_PACKAGE_NAME}
        PUBLIC
            CUDA::cuda_driver # this may not be needed
            $<BUILD_INTERFACE:deps::gds_static>
        )
endif ()

target_include_directories(${CUCIM_PACKAGE_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${CUCIM_PACKAGE_NAME}/3rdparty>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${CUCIM_PACKAGE_NAME}/3rdparty> # for 3rdparty libraries such as dlpack, nvtx3, and fmt
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/src
        )

add_library(${CUCIM_PACKAGE_NAME}::${CUCIM_PACKAGE_NAME} ALIAS ${CUCIM_PACKAGE_NAME})

################################################################################
# Add library: cucim-header-only
################################################################################
add_library(${CUCIM_PACKAGE_NAME}-header-only INTERFACE)
target_compile_definitions(${CUCIM_PACKAGE_NAME}-header-only
    INTERFACE
        CUCIM_VERSION=${PROJECT_VERSION}
        CUCIM_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUCIM_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUCIM_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUCIM_VERSION_BUILD=${PROJECT_VERSION_BUILD}
        CUCIM_SUPPORT_GDS=$<BOOL:${CUCIM_SUPPORT_GDS}>
        CUCIM_STATIC_GDS=$<BOOL:${CUCIM_STATIC_GDS}>
        CUCIM_SUPPORT_CUDA=$<BOOL:${CUCIM_SUPPORT_CUDA}>
        CUCIM_SUPPORT_NVTX=$<BOOL:${CUCIM_SUPPORT_NVTX}>
        CUCIM_HEADER_ONLY=1
        )
target_compile_features(${CUCIM_PACKAGE_NAME}-header-only INTERFACE ${CUCIM_REQUIRED_FEATURES})
target_include_directories(${CUCIM_PACKAGE_NAME}-header-only
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${CUCIM_PACKAGE_NAME}/3rdparty>
        )
add_library(${CUCIM_PACKAGE_NAME}::${CUCIM_PACKAGE_NAME}-header-only ALIAS ${CUCIM_PACKAGE_NAME}-header-only)

################################################################################
# Add tests
################################################################################
add_subdirectory(tests)

#################################################################################
## Add bindings
#################################################################################
#add_subdirectory(bindings/python)

unset(BUILD_SHARED_LIBS CACHE)
