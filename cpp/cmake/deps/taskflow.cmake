#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::taskflow)
    FetchContent_Declare(
            deps-taskflow
            GIT_REPOSITORY https://github.com/taskflow/taskflow.git
            GIT_TAG v3.2.0
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching taskflow sources")
    set(TF_BUILD_TESTS OFF)
    set(TF_BUILD_EXAMPLES OFF)

    FetchContent_MakeAvailable(deps-taskflow)
    message(STATUS "Fetching taskflow sources - done")

    add_library(deps::taskflow INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::taskflow INTERFACE Taskflow)
    set(deps-taskflow_SOURCE_DIR ${deps-taskflow_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-taskflow_SOURCE_DIR)
endif ()
