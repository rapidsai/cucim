#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if (NOT TARGET deps::openslide)
    add_library(deps::openslide SHARED IMPORTED GLOBAL)

    if (DEFINED ENV{CONDA_BUILD})
        set(OPENSLIDE_LIB_PATH "$ENV{PREFIX}/lib/libopenslide.so")
    elseif (DEFINED ENV{CONDA_PREFIX})
        set(OPENSLIDE_LIB_PATH "$ENV{CONDA_PREFIX}/lib/libopenslide.so")
    elseif (EXISTS /usr/lib/x86_64-linux-gnu/libopenslide.so)
        set(OPENSLIDE_LIB_PATH /usr/lib/x86_64-linux-gnu/libopenslide.so)
    else () # CentOS 6
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
