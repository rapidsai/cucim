/*
 * Apache License, Version 2.0
 * Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Code below is based on libtiff library which is under BSD-like license,
 * for providing lzw_decoder implementation.
 * The code is a port of the following file:
 *    https://gitlab.com/libtiff/libtiff/-/blob/8546f7ee994eacff0a563918096f16e0a6078fa2/libtiff/tif_lzw.c
 * , which is after v4.3.0.
 * Please see LICENSE-3rdparty.md for the detail.
 **/

#ifndef CUSLIDE_LZW_LIBTIFF_H
#define CUSLIDE_LZW_LIBTIFF_H

#include <cucim/io/device.h>

namespace cuslide::lzw
{

/****************************************************************************
 * Define missing types for libtiff's lzw decoder implementation
 ****************************************************************************/

// Forward declaration
struct TIFF;

#define COMPRESSION_LZW 5 /* Lempel-Ziv  & Welch */

/* Signed size type */
#define TIFF_SSIZE_T signed long
typedef TIFF_SSIZE_T tmsize_t;
typedef tmsize_t tsize_t; /* i/o size in bytes */

typedef void (*TIFFVoidMethod)(TIFF*);
typedef int (*TIFFBoolMethod)(TIFF*);
typedef int (*TIFFPreMethod)(TIFF*, uint16_t);
typedef int (*TIFFCodeMethod)(TIFF* tif, uint8_t* buf, tmsize_t size, uint16_t sample);
typedef int (*TIFFSeekMethod)(TIFF*, uint32_t);
typedef void (*TIFFPostMethod)(TIFF* tif, uint8_t* buf, tmsize_t size);
typedef uint32_t (*TIFFStripMethod)(TIFF*, uint32_t);
typedef void (*TIFFTileMethod)(TIFF*, uint32_t*, uint32_t*);

struct TIFF
{
    // Pointer to the buffer to be lzw-compressed/decompressed.
    uint8_t* tif_rawdata; /* raw data buffer */

    // Same with tif_rawcp
    uint8_t* tif_rawcp = nullptr; /* current spot in raw buffer */
    // Size of the buffer to be compressed/decompressed.
    tmsize_t tif_rawcc = 0; /* bytes unread from raw buffer */

    // Codec state initialized by tif->tif_setupdecode which is LZWSetupDecode
    uint8_t* tif_data = nullptr; /* compression scheme private data */

    TIFFBoolMethod tif_setupdecode = nullptr; /* called once before predecode */
    TIFFPreMethod tif_predecode = nullptr; /* pre- row/strip/tile decoding */
    TIFFCodeMethod tif_decoderow = nullptr; /* scanline decoding routine */
    TIFFCodeMethod tif_decodestrip = nullptr; /* strip decoding routine */
    TIFFCodeMethod tif_decodetile = nullptr; /* tile decoding routine */
    TIFFVoidMethod tif_cleanup = nullptr; /* cleanup state routine */

    // Additional method for predictor decoding
    TIFFPostMethod decodepfunc = nullptr; /* horizontal accumulator */

    // Not used in the implementation
    char* tif_name; /* name of open file */
    // Not used in the implementation
    uint32_t tif_row = 0; /* current scanline */
    // Not used in the implementation
    void* tif_clientdata; /* callback parameter */
};

int TIFFInitLZW(TIFF* tif, int scheme = COMPRESSION_LZW);

void horAcc8(uint8_t* cp0, tmsize_t cc, tmsize_t row_size);

} // namespace cuslide::lzw
#endif // CUSLIDE_LZW_LIBTIFF_H
