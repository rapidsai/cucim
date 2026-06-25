#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::pybind11)
    find_package(Git REQUIRED)

    FetchContent_Declare(
            deps-pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
            GIT_SHALLOW TRUE
            PATCH_COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_LIST_DIR}/pybind11_pr4857_4877.patch"
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching pybind11 sources")
    FetchContent_MakeAvailable(deps-pybind11)
    message(STATUS "Fetching pybind11 sources - done")

    # https://pybind11.readthedocs.io/en/stable/compiling.html#configuration-variables
    add_library(deps::pybind11 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::pybind11 INTERFACE pybind11::module)
    set(deps-pybind11_SOURCE_DIR ${deps-pybind11_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-pybind11_SOURCE_DIR)
endif ()
