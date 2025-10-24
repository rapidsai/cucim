#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

################################################################################
# Add executable: cucim_benchmarks
################################################################################

add_executable(cucim_benchmarks main.cpp config.h)

set_target_properties(cucim_benchmarks
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(cucim_benchmarks PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cucim_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_compile_definitions(cucim_benchmarks
    PUBLIC
        CUCIM_VERSION=${PROJECT_VERSION}
        CUCIM_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUCIM_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUCIM_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUCIM_VERSION_BUILD=${PROJECT_VERSION_BUILD}
)
target_link_libraries(cucim_benchmarks
        PRIVATE
            ${CUCIM_PACKAGE_NAME}
            deps::googlebenchmark
            deps::openslide
            deps::cli11
        )


################################################################################
# Add executable: cucim_primitives_benchmarks
################################################################################

add_executable(cucim_primitives_benchmarks primitives.cpp)
#set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA) # failed with CLI11 library

set_target_properties(cucim_primitives_benchmarks
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(cucim_primitives_benchmarks PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cucim_primitives_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_link_libraries(cucim_primitives_benchmarks
        PRIVATE
            ${CUCIM_PACKAGE_NAME}
            deps::googlebenchmark
        )
