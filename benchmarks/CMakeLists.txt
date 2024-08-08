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
target_link_libraries(cucim_primitives_benchmarks
        PRIVATE
            ${CUCIM_PACKAGE_NAME}
            deps::googlebenchmark
        )
