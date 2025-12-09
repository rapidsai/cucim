#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::pybind11_json)
    FetchContent_Declare(
            deps-pybind11_json
            GIT_REPOSITORY https://github.com/pybind/pybind11_json.git
            GIT_TAG 0.2.15
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching pybind11_json sources")
    FetchContent_MakeAvailable(deps-pybind11_json)
    message(STATUS "Fetching pybind11_json sources - done")

    add_library(deps::pybind11_json INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::pybind11_json INTERFACE pybind11_json)
    set(deps-pybind11_json_SOURCE_DIR ${deps-pybind11_json_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-pybind11_json_SOURCE_DIR)
endif ()
