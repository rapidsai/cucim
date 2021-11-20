#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
# Add executable: cumed_benchmarks
################################################################################
add_executable(cumed_benchmarks main.cpp config.h)
#set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA) # failed with CLI11 library

set_target_properties(cumed_benchmarks
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
target_compile_features(cumed_benchmarks PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cumed_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_compile_definitions(cumed_benchmarks
    PUBLIC
        CUMED_VERSION=${PROJECT_VERSION}
        CUMED_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUMED_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUMED_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        CUMED_VERSION_BUILD=${PROJECT_VERSION_BUILD}
)
target_link_libraries(cumed_benchmarks
        PRIVATE
            cucim::cucim
            deps::googlebenchmark
            deps::cli11
        )
