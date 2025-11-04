#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::fmt)
    FetchContent_Declare(
            deps-fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG 11.2.0
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching fmt sources")

    # Create static library
    cucim_set_build_shared_libs(OFF)

    FetchContent_MakeAvailable(deps-fmt)
    message(STATUS "Fetching fmt sources - done")

    # Set PIC to prevent the following error message
    # : /usr/bin/ld: ../lib/libfmtd.a(format.cc.o): relocation R_X86_64_PC32 against symbol `stderr@@GLIBC_2.2.5' can not be used when making a shared object; recompile with -fPIC
    set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cucim_restore_build_shared_libs()

    add_library(deps::fmt INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::fmt INTERFACE fmt::fmt-header-only)
    set(deps-fmt_SOURCE_DIR ${deps-fmt_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-fmt_SOURCE_DIR)
endif ()
