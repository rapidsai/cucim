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

if (NOT TARGET deps::googletest)
    FetchContent_Declare(
            deps-googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.10.0
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-googletest)
    if (NOT deps-googletest_POPULATED)
        message(STATUS "Fetching googletest sources")
        # TODO: use FetchContent_MakeAvailable (with EXCLUDE_FROM_ALL option in FetchContent_Declare) when CMake 3.30 is minimum required
        #       (https://cmake.org/cmake/help/latest/policy/CMP0169.html#policy:CMP0169)
        FetchContent_Populate(deps-googletest)
        message(STATUS "Fetching googletest sources - done")
    endif ()

    # Create static library
    cucim_set_build_shared_libs(OFF)
    add_subdirectory(${deps-googletest_SOURCE_DIR} ${deps-googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    cucim_restore_build_shared_libs()

    add_library(deps::googletest INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::googletest INTERFACE googletest)
    set(deps-googletest_SOURCE_DIR ${deps-googletest_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-googletest_SOURCE_DIR)
endif ()

#CMake Warning (dev) in cmake-build-debug/_deps/deps-googletest-src/googlemock/CMakeLists.txt:
#  Policy CMP0082 is not set: Install rules from add_subdirectory() are
#  interleaved with those in caller.  Run "cmake --help-policy CMP0082" for
#  policy details.  Use the cmake_policy command to set the policy and
#  suppress this warning.
