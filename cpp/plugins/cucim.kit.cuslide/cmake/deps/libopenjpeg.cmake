# Apache License, Version 2.0
# Copyright 2020-2021 NVIDIA Corporation
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

if (NOT TARGET deps::libopenjpeg)

    FetchContent_Declare(
            deps-libopenjpeg
            GIT_REPOSITORY https://github.com/uclouvain/openjpeg.git
            GIT_TAG v2.4.0
            GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(deps-libopenjpeg)
    if (NOT deps-libopenjpeg_POPULATED)
        message(STATUS "Fetching libopenjpeg sources")
        FetchContent_Populate(deps-libopenjpeg)
        message(STATUS "Fetching libopenjpeg sources - done")
    endif ()

    # Create static library
    # It build a static library when both BUILD_SHARED_LIBS and BUILD_STATIC_LIBS are ON
    # (build-debug/_deps/deps-libopenjpeg-src/src/lib/openjp2/CMakeLists.txt:94)
    #
    #     if(BUILD_SHARED_LIBS AND BUILD_STATIC_LIBS)
    cucim_set_build_shared_libs(ON)

    ###########################################################################
    # Build liblcms2 with the source in libopenjpeg
    ###########################################################################
    message(" ##GG ${deps-libopenjpeg_SOURCE_DIR}")

    add_subdirectory(${deps-libopenjpeg_SOURCE_DIR}/thirdparty/liblcms2 ${deps-libopenjpeg_BINARY_DIR}/thirdparty/liblcms2)

    # Override the output library folder path
    set_target_properties(lcms2
        PROPERTIES
        OUTPUT_NAME "lcms2"
        ARCHIVE_OUTPUT_DIRECTORY ${deps-libopenjpeg_BINARY_DIR}/thirdparty/lib)

    # Override definition of OPJ_HAVE_LIBLCMS2 to build color_apply_icc_profile() method
    target_compile_definitions(lcms2
        PUBLIC
            OPJ_HAVE_LIBLCMS2=1
    )

    # Set PIC to prevent the following error message
    # : /usr/bin/ld: _deps/deps-libopenjpeg-build/thirdparty/lib/liblcms2.a(cmserr.c.o): relocation R_X86_64_PC32 against symbol `_cmsMemPluginChunk' can not be used when making a shared object; recompile with -fPIC
    #   /usr/bin/ld: final link failed: bad value
    set_target_properties(lcms2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(deps::libopenjpeg-lcms2 INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::libopenjpeg-lcms2 INTERFACE lcms2)
    target_include_directories(deps::libopenjpeg-lcms2
        INTERFACE
            # lcms2.h is not included in 'lcms2' so manually include
            ${deps-libopenjpeg_SOURCE_DIR}/thirdparty/liblcms2/include
    )

    set(deps-libopenjpeg-lcms2_SOURCE_DIR ${deps-libopenjpeg_SOURCE_DIR}/thirdparty/liblcms2 CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libopenjpeg-lcms2_SOURCE_DIR)
    set(deps-libopenjpeg-lcms2_BINARY_DIR ${deps-libopenjpeg_BINARY_DIR}/thirdparty/liblcms2 CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libopenjpeg-lcms2_BINARY_DIR)

    ###########################################################################

    add_subdirectory(${deps-libopenjpeg_SOURCE_DIR} ${deps-libopenjpeg_BINARY_DIR} EXCLUDE_FROM_ALL)

    # Disable visibility to not expose unnecessary symbols
    set_target_properties(openjp2_static
        PROPERTIES
            C_VISIBILITY_PRESET hidden
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
    )
    # target_compile_options(openjp2_static PRIVATE $<$<CXX_COMPILER_ID:GNU>:-march=core-avx2>)

    # Set PIC to prevent the following error message
    # : /usr/bin/ld: lib/libopenjp2.a(cio.c.o): relocation R_X86_64_PC32 against symbol `opj_stream_read_skip' can not be used when making a shared object; recompile with -fPIC
    #   /usr/bin/ld: final link failed: bad value
    set_target_properties(openjp2_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cucim_restore_build_shared_libs()

    add_library(deps::libopenjpeg INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::libopenjpeg INTERFACE openjp2_static)
    target_include_directories(deps::libopenjpeg
        INTERFACE
            # openjpeg.h is not included in 'openjp2_static' so manually include
            ${deps-libopenjpeg_SOURCE_DIR}/src/lib/openjp2
            # opj_config.h is not included in openjp2_static so manually include
            ${deps-libopenjpeg_BINARY_DIR}/src/lib/openjp2
            # color.h is not included in 'openjp2_static' so manually include
            ${deps-libopenjpeg_SOURCE_DIR}/src/bin/common
            # opj_apps_config.h is not included in 'openjp2_static' so manually include
            ${deps-libopenjpeg_BINARY_DIR}/src/bin/common
    )

    set(deps-libopenjpeg_SOURCE_DIR ${deps-libopenjpeg_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libopenjpeg_SOURCE_DIR)
    set(deps-libopenjpeg_BINARY_DIR ${deps-libopenjpeg_BINARY_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libopenjpeg_BINARY_DIR)
endif ()
