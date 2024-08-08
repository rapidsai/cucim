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
