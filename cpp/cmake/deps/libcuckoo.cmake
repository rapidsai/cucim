# cmake-format: off
# SPDX-FileCopyrightText: Copyright 2021-2025 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

if (NOT TARGET deps::libcuckoo)
    find_package(Git REQUIRED)

    FetchContent_Declare(
            deps-libcuckoo
            GIT_REPOSITORY https://github.com/efficient/libcuckoo
            GIT_TAG v0.3
            GIT_SHALLOW TRUE
            PATCH_COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_LIST_DIR}/libcuckoo.patch"
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching libcuckoo sources")
    # Create static library
    cucim_set_build_shared_libs(OFF)
    FetchContent_MakeAvailable(deps-libcuckoo)
    message(STATUS "Fetching libcuckoo sources - done")

    # libcuckoo's CMakeLists.txt is not compatible with `add_subdirectory` method (it uses ${CMAKE_SOURCE_DIR} instead of ${CMAKE_CURRENT_SOURCE_DIR})
    # so add include directories explicitly.
    target_include_directories(libcuckoo INTERFACE
        $<BUILD_INTERFACE:${deps-libcuckoo_SOURCE_DIR}>
    )

    cucim_restore_build_shared_libs()

    add_library(deps::libcuckoo INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::libcuckoo INTERFACE libcuckoo)
    set(deps-libcuckoo_SOURCE_DIR ${deps-libcuckoo_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libcuckoo_SOURCE_DIR)
endif ()
