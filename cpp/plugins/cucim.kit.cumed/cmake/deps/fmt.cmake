# Apache License, Version 2.0
# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (NOT TARGET deps::fmt)
    FetchContent_Declare(
            deps-fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG 7.0.1
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-fmt)
    if (NOT deps-fmt_POPULATED)
        message(STATUS "Fetching fmt sources")
        FetchContent_Populate(deps-fmt)
        message(STATUS "Fetching fmt sources - done")
    endif ()

    # Create static library
    cucim_set_build_shared_libs(OFF)
    add_subdirectory(${deps-fmt_SOURCE_DIR} ${deps-fmt_BINARY_DIR} EXCLUDE_FROM_ALL)
    # Set PIC to prevent the following error message
    # : /usr/bin/ld: ../lib/libfmtd.a(format.cc.o): relocation R_X86_64_PC32 against symbol `stderr@@GLIBC_2.2.5' can not be used when making a shared object; recompile with -fPIC
    set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cucim_restore_build_shared_libs()

    add_library(deps::fmt INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::fmt INTERFACE fmt::fmt-header-only)
    set(deps-fmt_SOURCE_DIR ${deps-fmt_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-fmt_SOURCE_DIR)
endif ()
