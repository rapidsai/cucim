#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

if (NOT TARGET deps::boost-header-only)
    find_package(Git REQUIRED)
    set(Boost_VERSION 1.75.0)
    set(boost_component_list "interprocess" "config" "intrusive" "move" "assert" "static_assert" "container" "core" "date_time" "smart_ptr" "throw_exception" "utility" "type_traits" "numeric/conversion" "mpl" "preprocessor" "container_hash" "integer" "detail")

    message(STATUS "Fetching boost-header-only sources")
    FetchContent_Populate(deps-boost-header-only
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-${Boost_VERSION}
        GIT_SHALLOW TRUE
        PATCH_COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_LIST_DIR}/boost-header-only.patch" --directory=libs/interprocess)
    message(STATUS "Fetching boost-header-only sources - done")

    FetchContent_GetProperties(deps-boost-header-only)
    add_library(deps::boost-header-only INTERFACE IMPORTED GLOBAL)

    unset(boost_include_string)
    # Create a list of components
    foreach(item IN LISTS boost_component_list)
        set(boost_include_string "${boost_include_string}" "${deps-boost-header-only_SOURCE_DIR}/libs/${item}/include")
    endforeach(item)
    # https://www.boost.org/doc/libs/1_75_0/doc/html/interprocess.html#interprocess.intro.introduction_building_interprocess
    set_target_properties(deps::boost-header-only PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "${boost_include_string}"
        INTERFACE_COMPILE_DEFINITIONS
            BOOST_DATE_TIME_NO_LIB=1
    )

    set(deps-boost-header-only_SOURCE_DIR ${deps-boost-header-only_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-boost-header-only_SOURCE_DIR)
endif ()
