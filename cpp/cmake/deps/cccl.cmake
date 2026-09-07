#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::cccl)
    FetchContent_Declare(
            deps-cccl
            GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
            GIT_TAG v3.4.2
            GIT_SHALLOW TRUE
            # `libcudacxx/include` holds no CMakeLists.txt, so this downloads the
            # sources without configuring CCCL's own project, which would build
            # Thrust, CUB and their test suites. Only the header-only
            # `cuda::mr` types are needed here.
            SOURCE_SUBDIR libcudacxx/include
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching CCCL sources")

    FetchContent_MakeAvailable(deps-cccl)
    message(STATUS "Fetching CCCL sources - done")

    # Note for consumers: linking this target is not by itself enough for any
    # target that also pulls in the CUDA toolkit include directory. The toolkit
    # bundles an older libcudacxx, and whichever include directory is searched
    # first wins for cuda/std/*, which decides whether cuda::mr exists at all.
    # Targets in that situation add this directory with
    # target_include_directories(... BEFORE ...); see cpp/CMakeLists.txt.
    add_library(deps::cccl INTERFACE IMPORTED GLOBAL)
    # SYSTEM so that CCCL's own headers do not trip cuCIM's -Werror settings.
    set_target_properties(deps::cccl PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
            "${deps-cccl_SOURCE_DIR}/libcudacxx/include"
        INTERFACE_INCLUDE_DIRECTORIES
            "${deps-cccl_SOURCE_DIR}/libcudacxx/include"
    )

    set(deps-cccl_SOURCE_DIR ${deps-cccl_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-cccl_SOURCE_DIR)
endif ()
