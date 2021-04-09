#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
set_source_files_properties(test_read_rawtiff.cpp PROPERTIES LANGUAGE CUDA) # failed with CLI11 library

set_target_properties(cuslide_tests
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared
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

include(ParseAndAddCatchTests)
# See https://github.com/catchorg/Catch2/blob/master/docs/cmake-integration.md#parseandaddcatchtestscmake for other options
ParseAndAddCatchTests(cuslide_tests)
