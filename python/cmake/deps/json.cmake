#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::json)
    FetchContent_Declare(
            deps-json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.11.3
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching json sources")

    # Typically you don't care so much for a third party library's tests to be
    # run from your own project's code.
    option(JSON_BuildTests OFF)

    FetchContent_MakeAvailable(deps-json)
    message(STATUS "Fetching json sources - done")

    add_library(deps::json INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::json INTERFACE nlohmann_json::nlohmann_json)
    set(deps-json_SOURCE_DIR ${deps-json_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-json_SOURCE_DIR)
endif ()
