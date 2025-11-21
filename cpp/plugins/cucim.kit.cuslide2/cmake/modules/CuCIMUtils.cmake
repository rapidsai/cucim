#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

# Store current BUILD_SHARED_LIBS setting in CUCIM_OLD_BUILD_SHARED_LIBS
if(NOT COMMAND cucim_set_build_shared_libs)
    macro(cucim_set_build_shared_libs new_value)
        set(CUCIM_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}})
        if (DEFINED CACHE{BUILD_SHARED_LIBS})
            set(CUCIM_OLD_BUILD_SHARED_LIBS_CACHED TRUE)
        else()
            set(CUCIM_OLD_BUILD_SHARED_LIBS_CACHED FALSE)
        endif()
        set(BUILD_SHARED_LIBS ${new_value} CACHE BOOL "" FORCE)
    endmacro()
endif()

# Restore BUILD_SHARED_LIBS setting from CUCIM_OLD_BUILD_SHARED_LIBS
if(NOT COMMAND cucim_restore_build_shared_libs)
    macro(cucim_restore_build_shared_libs)
        if (CUCIM_OLD_BUILD_SHARED_LIBS_CACHED)
            set(BUILD_SHARED_LIBS ${CUCIM_OLD_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
        else()
            unset(BUILD_SHARED_LIBS CACHE)
            set(BUILD_SHARED_LIBS ${CUCIM_OLD_BUILD_SHARED_LIBS})
        endif()
    endmacro()
endif()
