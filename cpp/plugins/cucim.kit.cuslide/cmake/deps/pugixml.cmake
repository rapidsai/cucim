# Apache License, Version 2.0
# Copyright 2020-2025 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (NOT TARGET deps::pugixml)
    FetchContent_Declare(
            deps-pugixml
            GIT_REPOSITORY https://github.com/zeux/pugixml.git
            GIT_TAG v1.11.1
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-pugixml)
    if (NOT deps-pugixml_POPULATED)
        message(STATUS "Fetching pugixml sources")
        FetchContent_Populate(deps-pugixml)
        message(STATUS "Fetching pugixml sources - done")
    endif ()

    # Create static library
    cucim_set_build_shared_libs(OFF)

    add_subdirectory(${deps-pugixml_SOURCE_DIR} ${deps-pugixml_BINARY_DIR} EXCLUDE_FROM_ALL)

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
