#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::openslide)
    add_library(deps::openslide SHARED IMPORTED GLOBAL)

    if (DEFINED ENV{CONDA_BUILD})
        set(OPENSLIDE_LIB_PATH "$ENV{PREFIX}/lib/libopenslide.so")
    elseif (DEFINED ENV{CONDA_PREFIX})
        set(OPENSLIDE_LIB_PATH "$ENV{CONDA_PREFIX}/lib/libopenslide.so")
    elseif (EXISTS /usr/lib/x86_64-linux-gnu/libopenslide.so)
        set(OPENSLIDE_LIB_PATH /usr/lib/x86_64-linux-gnu/libopenslide.so)
    elseif (EXISTS /usr/lib/aarch64-linux-gnu/libopenslide.so)
        set(OPENSLIDE_LIB_PATH /usr/lib/aarch64-linux-gnu/libopenslide.so)
    else () # CentOS (x86_64)
        set(OPENSLIDE_LIB_PATH /usr/lib64/libopenslide.so)
    endif ()

    if (DEFINED ENV{CONDA_BUILD})
        set(OPENSLIDE_INCLUDE_PATH "$ENV{PREFIX}/include/")
    elseif (DEFINED ENV{CONDA_PREFIX})
        set(OPENSLIDE_INCLUDE_PATH "$ENV{CONDA_PREFIX}/include/")
    else ()
        set(OPENSLIDE_INCLUDE_PATH "/usr/include/")
    endif ()

    set_target_properties(deps::openslide PROPERTIES
        IMPORTED_LOCATION "${OPENSLIDE_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSLIDE_INCLUDE_PATH}"
    )
endif ()
