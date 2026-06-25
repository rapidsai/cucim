#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

include(CTest)
enable_testing()

################################################################################
# Add executable: cumed_tests
################################################################################
add_executable(cumed_tests
        config.h
        main.cpp
        test_basic.cpp
        )

set_target_properties(cumed_tests
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(cumed_tests PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cumed_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_compile_definitions(cumed_tests
    PUBLIC
        CUMED_VERSION=${PROJECT_VERSION}
        CUMED_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUMED_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUMED_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUMED_VERSION_BUILD=${PROJECT_VERSION_BUILD}
)
target_link_libraries(cumed_tests
        PRIVATE
            cucim::cucim
            ${CUCIM_PLUGIN_NAME}
            deps::catch2
            deps::cli11
            deps::fmt
        )

# Add headers in src
target_include_directories(cumed_tests
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>
        )

include(Catch)
# See https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#catchcmake-and-catchaddtestscmake for other options
# Do not use catch_discover_tests() since it causes a test to be run at build time
# and somehow it causes a deadlock during the build.
# catch_discover_tests(cumed_tests)
