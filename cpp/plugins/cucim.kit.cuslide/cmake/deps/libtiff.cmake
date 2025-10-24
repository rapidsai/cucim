# cmake-format: off
# SPDX-FileCopyrightText: Copyright 2020-2025 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

if (NOT TARGET deps::libtiff)
#    add_library(deps::libtiff SHARED IMPORTED GLOBAL)
#
#    set_target_properties(deps::libtiff PROPERTIES
#        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libtiff.so"
#        INTERFACE_INCLUDE_DIRECTORIES "/usr/include/x86_64-linux-gnu"
#    )

    FetchContent_Declare(
            deps-libtiff
            GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
            GIT_TAG v4.1.0
            GIT_SHALLOW TRUE
            PATCH_COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_LIST_DIR}/libtiff.patch"
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching libtiff sources")

    # Set policies for libtiff
    set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_LIST_DIR}/libtiff-policies-fix.cmake")

    # Create static library
    cucim_set_build_shared_libs(OFF)

    # The following does some tricks so that libtiff uses libjpeg-turbo instead of system's libjpeg.
    # - set jpeg to OFF so that we can manually specify LIBRARIES and INCLUDES
    #   (status message in cmake shows jpeg is OFF but it actually use libjpeg)
    # - set TIFF_INCLUDES instead of JPEG_INCLUDE_DIR to set libjpeg-turbo's include folder with higher priority
    #   (otherwise, jpeg's include dir wouldn't be the first of TIFF_INCLUDES)
    # Otherwise, libtiff would use system's shared libjpeg(8.0) whereas libjpeg turbo uses static libjpeg(6.2)
    # so symbol conflict(such as jpeg_CreateDecompress) happens.
    # See 'cmake-build-debug/_deps/deps-libtiff-src/CMakeLists.txt' for existing libtiff's logic.
    set(jpeg OFF)
    set(JPEG_FOUND TRUE)
    set(JPEG_LIBRARIES deps::libjpeg-turbo)
    # for jpeglib.h and jconfig.h/jconfigint.h
    set(TIFF_INCLUDES ${deps-libjpeg-turbo_SOURCE_DIR} ${deps-libjpeg-turbo_BINARY_DIR} )

    # Explicitly disable external codecs
    set(zlib OFF)
    set(pixarlog OFF)
    set(lzma OFF)
    set(old-jpeg OFF)
    set(jpeg12 OFF)
    set(zstd OFF)
    set(jbig OFF)
    set(webp OFF)

    FetchContent_MakeAvailable(deps-libtiff)
    message(STATUS "Fetching libtiff sources - done")

    # Disable visibility to not expose unnecessary symbols
    set_target_properties(tiff tiffxx
    PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)

    # Set PIC to prevent the following error message
    # : /usr/bin/ld: lib/libtiff.a(tif_close.c.o): relocation R_X86_64_PC32 against symbol `TIFFCleanup' can not be used when making a shared object; recompile with -fPIC
    set_target_properties(tiff PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cucim_restore_build_shared_libs()

    add_library(deps::libtiff INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::libtiff INTERFACE tiffxx)
    set(deps-libtiff_SOURCE_DIR ${deps-libtiff_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-libtiff_SOURCE_DIR)
endif ()
