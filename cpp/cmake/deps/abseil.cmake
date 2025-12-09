#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::abseil)
    FetchContent_Declare(
            deps-abseil
            GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
            GIT_TAG 20200225.2
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    # Create static library
    cucim_set_build_shared_libs(OFF)
         # Disable BUILD_TESTING (cmake-build-debug/_deps/deps-abseil-src/CMakeLists.txt:97)
    set(BUILD_TESTING FALSE)
    message(STATUS "Fetching abseil sources")
    FetchContent_MakeAvailable(deps-abseil)
    message(STATUS "Fetching abseil sources - done")

    # Set PIC to prevent the following error message
    # : /usr/bin/ld: ../lib/libabsl_strings.a(escaping.cc.o): relocation R_X86_64_PC32 against symbol `_ZN4absl14lts_2020_02_2516numbers_internal8kHexCharE' can not be used when making a shared object; recompile with -fPIC
    set_target_properties(absl_strings absl_strings_internal absl_int128 absl_raw_logging_internal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cucim_restore_build_shared_libs()

    add_library(deps::abseil INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::abseil INTERFACE absl::strings)
    set(deps-abseil_SOURCE_DIR ${deps-abseil_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-abseil_SOURCE_DIR)
endif ()
