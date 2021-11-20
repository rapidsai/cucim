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

#include "libopenjpeg.h"

#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <fmt/format.h>
#include <openjpeg.h>

#include "color_conversion.h"

#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))
#define ALIGN_SIZE 16

// Extern methods from 'deps-libopenjpeg-src/src/bin/common/color.h'
extern "C"
{
    void color_sycc_to_rgb(opj_image_t* img);
    void color_apply_icc_profile(opj_image_t* image);
}


namespace cuslide::jpeg2k
{
/**
 * Code below is derived from the openjpeg's code which is under BSD-2-Clause License
 * Please see LICENSE-3rdparty.md for the detail.
 * - https://github.com/uclouvain/openjpeg/blob/v2.4.0/tests/test_decode_area.c
 * - https://github.com/uclouvain/openjpeg/blob/v2.4.0/src/lib/openjpip/jp2k_decoder.c#L46
 */

struct UserData
{
    uint8_t* buf = nullptr;
    uint64_t size = 0;
    uint64_t offset = 0;
};

static void error_callback(const char* msg, void* client_data)
{
    (void)client_data;
    fprintf(stderr, "[Error] %s\n", msg);
}

static void warning_callback(const char* msg, void* client_data)
{
    (void)client_data;
    fprintf(stderr, "[Warning] %s\n", msg);
}

static OPJ_SIZE_T read_callback(void* p_buffer, OPJ_SIZE_T p_nb_bytes, void* p_user_data)
{
    auto data = static_cast<UserData*>(p_user_data);
    if (data->offset >= data->size)
    {
        return -1;
    }
    if (data->offset + p_nb_bytes >= data->size)
    {
        size_t nb_bytes_to_read = data->size - data->offset;
        memcpy(p_buffer, data->buf + data->offset, nb_bytes_to_read);
        data->offset = data->size;
        return nb_bytes_to_read;
    }
    if (p_nb_bytes == 0)
    {
        return -1;
    }
    memcpy(p_buffer, data->buf + data->offset, p_nb_bytes);
    data->offset += p_nb_bytes;
    return p_nb_bytes;
}

static OPJ_OFF_T skip_callback(OPJ_OFF_T p_nb_bytes, void* p_user_data)
{
    auto data = static_cast<UserData*>(p_user_data);
    if (data->offset + p_nb_bytes >= data->size)
    {
        uint64_t skip_count = data->size - data->offset;
        data->offset = data->size;
        return skip_count;
    }
    data->offset += p_nb_bytes;
    return p_nb_bytes;
}

static OPJ_BOOL seek_callback(OPJ_OFF_T p_nb_bytes, void* p_user_data)
{
    auto data = static_cast<UserData*>(p_user_data);
    if (p_nb_bytes < 0)
    {
        data->offset = 0;
        return OPJ_FALSE;
    }
    if (static_cast<uint64_t>(p_nb_bytes) >= data->size)
    {
        data->offset = data->size;
        return OPJ_FALSE;
    }
    data->offset = p_nb_bytes;
    return OPJ_TRUE;
}

bool decode_libopenjpeg(int fd,
                        unsigned char* jpeg_buf,
                        uint64_t offset,
                        uint64_t size,
                        uint8_t** dest,
                        uint64_t dest_nbytes,
                        const cucim::io::Device& out_device,
                        ColorSpace color_space)
{
    (void)out_device;

    if (dest == nullptr)
    {
        throw std::runtime_error("'dest' shouldn't be nullptr in decode_libopenjpeg()");
    }

    // Allocate memory only when dest is not null
    if (*dest == nullptr)
    {
        if ((*dest = (unsigned char*)malloc(dest_nbytes)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate uncompressed image buffer");
        }
    }

    if (jpeg_buf == nullptr)
    {

        if ((jpeg_buf = (unsigned char*)malloc(size)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate buffer for libopenjpeg!");
        }

        if (pread(fd, jpeg_buf, size, offset) < 1)
        {
            throw std::runtime_error("Unable to read file for libopenjpeg!");
        }
    }
    else
    {
        fd = -1;
        jpeg_buf += offset;
    }

    opj_stream_t* stream = opj_stream_create(size, OPJ_TRUE);
    if (!stream)
    {
        throw std::runtime_error("[Error] Failed to create stream\n");
    }

    UserData data{ jpeg_buf, size, 0 };
    opj_stream_set_user_data(stream, &data, nullptr);
    opj_stream_set_user_data_length(stream, size);
    opj_stream_set_read_function(stream, read_callback);
    opj_stream_set_skip_function(stream, skip_callback);
    opj_stream_set_seek_function(stream, seek_callback);

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    if (!codec)
    {
        throw std::runtime_error("[Error] Failed to create codec\n");
    }

    // Register the event callbacks
    opj_set_warning_handler(codec, warning_callback, nullptr);
    opj_set_error_handler(codec, error_callback, nullptr);

    opj_dparameters_t parameters;
    opj_set_default_decoder_parameters(&parameters);
    opj_setup_decoder(codec, &parameters);

    opj_image_t* image = nullptr;

    try
    {
        if (!opj_read_header(stream, codec, &image))
        {
            throw std::runtime_error("[Error] Failed to read header from OpenJpeg stream\n");
        }

        if (image->numcomps != 3)
        {
            throw std::runtime_error("[Error] Only RGB images are supported\n");
        }

        if (!opj_decode(codec, stream, image))
        {
            throw std::runtime_error("[Error] Failed to decode image\n");
        }
        if (image->color_space != OPJ_CLRSPC_SYCC)
        {
            if (color_space == ColorSpace::kSYCC)
            {
                image->color_space = OPJ_CLRSPC_SYCC;
            }
            else if (color_space == ColorSpace::kRGB)
            {
                image->color_space = OPJ_CLRSPC_SRGB;
            }
        }

        // YCbCr 4:2:2 or 4:2:0 or 4:4:4
        if ((image->color_space == OPJ_CLRSPC_SYCC) && (image->icc_profile_buf == nullptr))
        {
            uint32_t& comp0_dx = image->comps[0].dx;
            uint32_t& comp0_dy = image->comps[0].dy;
            uint32_t& comp1_dx = image->comps[1].dx;
            uint32_t& comp1_dy = image->comps[1].dy;
            uint32_t& comp2_dx = image->comps[2].dx;
            uint32_t& comp2_dy = image->comps[2].dy;

            if ((comp0_dx == 1) && (comp1_dx == 2) && (comp2_dx == 2) && (comp0_dy == 1) && (comp1_dy == 1) &&
                (comp2_dy == 1))
            {
                fast_sycc422_to_rgb(image, *dest); // horizontal sub-sample only
            }
            else if ((comp0_dx == 1) && (comp1_dx == 2) && (comp2_dx == 2) && (comp0_dy == 1) && (comp1_dy == 2) &&
                     (comp2_dy == 2))
            {
                fast_sycc420_to_rgb(image, *dest); // horizontal and vertical sub-sample
            }
            else if ((comp0_dx == 1) && (comp1_dx == 1) && (comp2_dx == 1) && (comp0_dy == 1) && (comp1_dy == 1) &&
                     (comp2_dy == 1))
            {
                fast_sycc444_to_rgb(image, *dest); // no sub-sample
            }
            else
            {
                throw std::runtime_error(fmt::format(
                    "[Error] decode_libopenjpeg cannot convert the image (comp0_dx:{}, comp0_dy:{}, comp1_dx:{}, comp1_dy:{}, comp2_dx:{}, comp2_dy:{})\n",
                    comp0_dx, comp0_dy, comp1_dx, comp1_dy, comp2_dx, comp2_dy));
            }
        }
        else
        {
            if (image->color_space == OPJ_CLRSPC_SYCC)
            {
                color_sycc_to_rgb(image);
            }
            if (image->icc_profile_buf)
            {
                color_apply_icc_profile(image);
                image->icc_profile_len = 0;
                free(image->icc_profile_buf);
                image->icc_profile_buf = nullptr;
            }
            if (image->comps)
            {
                fast_image_to_rgb(image, *dest);
            }
        }
    }
    catch (const std::runtime_error& e)
    {
        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
        opj_image_destroy(image);
        if (fd != -1)
        {
            free(jpeg_buf);
        }
        throw e;
    }

    opj_destroy_codec(codec);
    opj_stream_destroy(stream);
    opj_image_destroy(image);
    if (fd != -1)
    {
        free(jpeg_buf);
    }

    return true;
}

} // namespace cuslide::jpeg2k
