#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

include(CTest)
enable_testing()

################################################################################
# Add executable: cuslide_tests
################################################################################
add_executable(cuslide_tests
        config.h
        main.cpp
        test_read_region.cpp
        test_read_rawtiff.cpp
        test_philips_tiff.cpp
        )
set_target_properties(cuslide_tests
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(cuslide_tests PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cuslide_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_compile_definitions(cuslide_tests
    PUBLIC
        CUSLIDE_VERSION=${PROJECT_VERSION}
        CUSLIDE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUSLIDE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUSLIDE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUSLIDE_VERSION_BUILD=${PROJECT_VERSION_BUILD}
)
target_link_libraries(cuslide_tests
        PRIVATE
            CUDA::cudart
            cucim::cucim
            ${CUCIM_PLUGIN_NAME}
            deps::catch2
            deps::openslide
            deps::cli11
            deps::fmt
        )

# Add headers in src
target_include_directories(cuslide_tests
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>
        )

include(Catch)
# See https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#catchcmake-and-catchaddtestscmake for other options
# Do not use catch_discover_tests() since it causes a test to be run at build time
# and somehow it causes a deadlock during the build.
# catch_discover_tests(cuslide_tests)
