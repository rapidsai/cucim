#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::googletest)
    FetchContent_Declare(
            deps-googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.16.0
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching googletest sources")
    # Create static library
    cucim_set_build_shared_libs(OFF)
    FetchContent_MakeAvailable(deps-googletest)
    message(STATUS "Fetching googletest sources - done")
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
