#
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
catch_discover_tests(cumed_tests)
