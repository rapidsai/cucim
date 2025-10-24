#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

################################################################################
# Add executable: cuslide_benchmarks
################################################################################
add_executable(cuslide_benchmarks main.cpp config.h)
#set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA) # failed with CLI11 library

set_target_properties(cuslide_benchmarks
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(cuslide_benchmarks PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cuslide_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_compile_definitions(cuslide_benchmarks
    PUBLIC
        CUSLIDE_VERSION=${PROJECT_VERSION}
        CUSLIDE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUSLIDE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUSLIDE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUSLIDE_VERSION_BUILD=${PROJECT_VERSION_BUILD}
)
target_link_libraries(cuslide_benchmarks
        PRIVATE
            cucim::cucim
            deps::googlebenchmark
            deps::openslide
            deps::cli11
        )
