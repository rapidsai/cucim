/*
 * Apache License, Version 2.0
 * Copyright 2020 NVIDIA Corporation
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
 * Code below is derived from the libjpeg-turbo's example code (tjexample.c) which is under three compatible
 * BSD-style open source licenses
 * - The IJG (Independent JPEG Group) License
 * - The Modified (3-clause) BSD License
 * - The zlib License
 * Please see LICENSE-3rdparty.md for the detail.
 * (https://github.com/libjpeg-turbo/libjpeg-turbo/blob/00607ec260efa4cfe10f9b36d6e3d3590ae92d79/tjexample.c)
 */

#include "libjpeg_turbo.h"

#include <cstring>
#include <jpeglib.h>
#include <setjmp.h>
#include <turbojpeg.h>
#include <unistd.h>

struct my_error_mgr
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
    void (*emit_message)(j_common_ptr, int);
    boolean warning, stopOnWarning;
};

typedef struct _tjinstance
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_decompress_struct dinfo;
    struct my_error_mgr jerr;
    int init, headerRead;
    char errStr[JMSG_LENGTH_MAX];
    boolean isInstanceError;
} tjinstance;

extern "C" void jpeg_mem_src_tj(j_decompress_ptr, const unsigned char*, unsigned long);

namespace cuslide::jpeg
{


#define THROW(action, message)                                                                                         \
    {                                                                                                                  \
        printf("ERROR in line %d while %s:\n%s\n", __LINE__, action, message);                                         \
        retval = -1;                                                                                                   \
        goto bailout;                                                                                                  \
    }

#define THROW_TJ(action) THROW(action, tjGetErrorStr2(tjInstance))

#define THROW_UNIX(action) THROW(action, strerror(errno))

#define DEFAULT_SUBSAMP TJSAMP_444
#define DEFAULT_QUALITY 95

// static const char* subsampName[TJ_NUMSAMP] = { "4:4:4", "4:2:2", "4:2:0", "Grayscale", "4:4:0", "4:1:1" };

// static const char* colorspaceName[TJ_NUMCS] = { "RGB", "YCbCr", "GRAY", "CMYK", "YCCK" };

bool decode_libjpeg(int fd,
                    unsigned char* jpeg_buf,
                    uint64_t offset,
                    uint64_t size,
                    const void* jpegtable_data,
                    uint32_t jpegtable_count,
                    uint8_t** dest,
                    const cucim::io::Device& out_device)
{
    (void)out_device;

    //    tjscalingfactor scalingFactor = { 1, 1 };
    tjtransform xform;
    int flags = 0;
    //    flags |= TJFLAG_FASTUPSAMPLE;
    //    flags |= TJFLAG_FASTDCT;
    //    flags |= TJFLAG_ACCURATEDCT;
    int width, height;
    int retval = 0, pixelFormat = TJPF_RGB;
    (void)retval; // retval is used by macro THROW
    tjhandle tjInstance = nullptr;

    memset(&xform, 0, sizeof(tjtransform));

    /* Input image is a JPEG image.  Decompress and/or transform it. */

    int inSubsamp, inColorspace;
    //    int doTransform = (xform.op != TJXOP_NONE || xform.options != 0 || xform.customFilter != NULL);

    /* Read the JPEG file/buffer into memory. */
    if (size == 0)
        THROW("determining input file size", "Input file contains no data");

    if (dest == nullptr)
    {
        THROW("checking dest ptr", "'dest' shouldn't be nullptr in decode_libjpeg()");
    }

    if (jpeg_buf == nullptr)
    {
        if ((jpeg_buf = (unsigned char*)tjAlloc(size)) == nullptr)
            THROW_UNIX("allocating JPEG buffer");

        if (pread(fd, jpeg_buf, size, offset) < 1)
            THROW_UNIX("reading input file");
    }
    else
    {
        fd = -1;
        jpeg_buf += offset;
    }

    if ((tjInstance = tjInitDecompress()) == nullptr)
        THROW_TJ("initializing decompressor");

    // Read jpeg tables if exists
    if (jpegtable_count)
    {
        if (!read_jpeg_header_tables(tjInstance, jpegtable_data, jpegtable_count))
        {
            THROW_TJ("reading JPEG header tables");
        }
    }

    if (tjDecompressHeader3(tjInstance, jpeg_buf, size, &width, &height, &inSubsamp, &inColorspace) < 0)
        THROW_TJ("reading JPEG header");

    //    printf("%s Image:  %d x %d pixels, %s subsampling, %s colorspace\n", (doTransform ? "Transformed" : "Input"),
    //    width,
    //           height, subsampName[inSubsamp], colorspaceName[inColorspace]);

    // Allocate memory only when dest is not null
    if (*dest == nullptr)
    {
        if ((*dest = (unsigned char*)tjAlloc(width * height * tjPixelSize[pixelFormat])) == nullptr)
            THROW_UNIX("Unable to allocate uncompressed image buffer");
    }

    if (tjDecompress2(tjInstance, jpeg_buf, size, (unsigned char*)*dest, width, 0, height, pixelFormat, flags) < 0)
        THROW_TJ("decompressing JPEG image");

    if (fd != -1)
    {
        tjFree(jpeg_buf);
    }
    tjDestroy(tjInstance);
    return true;

bailout:
    if (tjInstance)
        tjDestroy(tjInstance);
    if (fd != -1)
    {
        tjFree(jpeg_buf);
    }
    return false;
}

bool read_jpeg_header_tables(const void* handle, const void* jpeg_buf, unsigned long jpeg_size)
{
    tjinstance* instance = (tjinstance*)handle;
    j_decompress_ptr dinfo = NULL;
    dinfo = &instance->dinfo;
    instance->jerr.warning = FALSE;
    instance->isInstanceError = FALSE;

    if (setjmp(instance->jerr.setjmp_buffer))
    {
        /* If we get here, the JPEG code has signaled an error. */
        return false;
    }

    jpeg_mem_src_tj(dinfo, static_cast<const unsigned char*>(jpeg_buf), jpeg_size);
    if (jpeg_read_header(dinfo, FALSE) != JPEG_HEADER_TABLES_ONLY)
    {
        return false;
    }

    return true;
}

bool get_dimension(const void* image_buf, uint64_t offset, uint64_t size, int* out_width, int* out_height)
{
    int retval = 0;
    (void)retval; // retval is used by macro THROW
    tjhandle tjInstance = nullptr;

    int inSubsamp, inColorspace;

    if (image_buf == nullptr || size == 0)
        THROW("determining input buffer size", "Input buffer contains no data");

    if ((tjInstance = tjInitDecompress()) == nullptr)
        THROW_TJ("initializing decompressor");

    if (tjDecompressHeader3(tjInstance, static_cast<const unsigned char*>(image_buf) + offset, size, out_width,
                            out_height, &inSubsamp, &inColorspace) < 0)
        THROW_TJ("reading JPEG header");

    tjDestroy(tjInstance);
    return true;

bailout:
    if (tjInstance)
        tjDestroy(tjInstance);
    return false;
}

} // namespace cuslide::jpeg
