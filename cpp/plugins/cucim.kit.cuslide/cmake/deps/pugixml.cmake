# cmake-format: off
# SPDX-FileCopyrightText: Copyright 2020-2025 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

if (NOT TARGET deps::pugixml)
    FetchContent_Declare(
            deps-pugixml
            GIT_REPOSITORY https://github.com/zeux/pugixml.git
            GIT_TAG v1.15
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )
    message(STATUS "Fetching pugixml sources")

    # Create static library
    cucim_set_build_shared_libs(OFF)
    FetchContent_MakeAvailable(deps-pugixml)

    message(STATUS "Fetching pugixml sources - done")
    # Disable visibility to not expose unnecessary symbols
    set_target_properties(pugixml-static
    PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)

    cucim_restore_build_shared_libs()

    add_library(deps::pugixml INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::pugixml INTERFACE pugixml-static)
    set(deps-pugixml_SOURCE_DIR ${deps-pugixml_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-pugixml_SOURCE_DIR)
endif ()
