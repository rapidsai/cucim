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

if (NOT TARGET deps::abseil)
    set(Abseil_VERSION 20240116.2)

    if (DEFINED ENV{CONDA_PREFIX})
        find_package(absl REQUIRED)
        if (NOT absl_FOUND)
            message(FATAL_ERROR "fmt package not found in conda environment")
        endif()

        add_library(deps::abseil ALIAS absl::strings)
        get_target_property(abseil_INCLUDE_DIR absl::strings INTERFACE_INCLUDE_DIRECTORIES)

        set(abseil_INCLUDE_DIR ${abseil_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(abseil_INCLUDE_DIR)
    else()
        # Fallback to fetching Abseil sources
        FetchContent_Declare(
                deps-abseil
                GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
                GIT_TAG ${Abseil_VERSION}
                GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(deps-abseil)
        if (NOT deps-abseil_POPULATED)
            message(STATUS "Fetching abseil sources")
            # TODO: use FetchContent_MakeAvailable (with EXCLUDE_FROM_ALL option in FetchContent_Declare) when CMake 3.30 is minimum required
            #       (https://cmake.org/cmake/help/latest/policy/CMP0169.html#policy:CMP0169)
            FetchContent_Populate(deps-abseil)
            message(STATUS "Fetching abseil sources - done")
        endif ()

        # Create static library
        cucim_set_build_shared_libs(OFF)
        set(BUILD_TESTING FALSE) # Disable BUILD_TESTING (cmake-build-debug/_deps/deps-abseil-src/CMakeLists.txt:97)
        add_subdirectory(${deps-abseil_SOURCE_DIR} ${deps-abseil_BINARY_DIR} EXCLUDE_FROM_ALL)

        # Set PIC to prevent the following error message
        # : /usr/bin/ld: ../lib/libabsl_strings.a(escaping.cc.o): relocation R_X86_64_PC32 against symbol `_ZN4absl14lts_2020_02_2516numbers_internal8kHexCharE' can not be used when making a shared object; recompile with -fPIC
        set_target_properties(absl_strings absl_strings_internal absl_int128 absl_raw_logging_internal PROPERTIES POSITION_INDEPENDENT_CODE ON)
        cucim_restore_build_shared_libs()

        add_library(deps::abseil INTERFACE IMPORTED GLOBAL)
        target_link_libraries(deps::abseil INTERFACE absl::strings)
        set(abseil_INCLUDE_DIR ${deps-deps-abseil_SOURCE_DIR} CACHE INTERNAL "" FORCE)
        mark_as_advanced(abseil_INCLUDE_DIR)
    endif()
endif ()
