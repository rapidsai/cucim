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
# Add executable: cucim_tests
################################################################################
add_executable(cucim_tests
        main.cpp
        test_read_region.cpp
        test_cufile.cpp
        test_metadata.cpp
        )

set_target_properties(cucim_tests
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(cucim_tests PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cucim_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_compile_definitions(cucim_tests
    PUBLIC
        CUCIM_VERSION=${PROJECT_VERSION}
        CUCIM_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUCIM_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUCIM_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUCIM_VERSION_BUILD=${PROJECT_VERSION_BUILD}
)

# Include pthread
# (https://cmake.org/cmake/help/v3.18/module/FindThreads.html)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

target_link_libraries(cucim_tests
        PRIVATE
            CUDA::cudart
            ${CUCIM_PACKAGE_NAME}
            deps::catch2
            deps::openslide
            deps::taskflow
            Threads::Threads # -lpthread
        )

include(ParseAndAddCatchTests)
# See https://github.com/catchorg/Catch2/blob/master/docs/cmake-integration.md#parseandaddcatchtestscmake for other options
ParseAndAddCatchTests(cucim_tests)
