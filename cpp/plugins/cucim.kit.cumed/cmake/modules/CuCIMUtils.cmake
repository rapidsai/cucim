#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

# Define CMAKE_CUDA_ARCHITECTURES for the given architecture values
#
# Params:
#   arch_list - architecture value list (e.g., '60;70;75;80;86')
if(NOT COMMAND cucim_define_cuda_architectures)
    function(cucim_define_cuda_architectures arch_list)
        set(arch_string "")
        # Create SASS for all architectures in the list
        foreach(arch IN LISTS arch_list)
            set(arch_string "${arch_string}" "${arch}-real")
        endforeach(arch)

        # Create PTX for the latest architecture for forward-compatibility.
        list(GET arch_list -1 latest_arch)
        foreach(arch IN LISTS arch_list)
            set(arch_string "${arch_string}" "${latest_arch}-virtual")
        endforeach(arch)
        set(CMAKE_CUDA_ARCHITECTURES ${arch_string} PARENT_SCOPE)
    endfunction()
endif()
