/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#ifndef CUSLIDE2_NVIMGCODEC_DECODER_H
#define CUSLIDE2_NVIMGCODEC_DECODER_H

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <cucim/io/device.h>
#include <cstdint>

namespace cuslide2::nvimgcodec
{

/**
 * Decode JPEG using nvImageCodec
 * 
 * @param fd File descriptor
 * @param jpeg_buf JPEG buffer (if nullptr, read from fd at offset)
 * @param offset File offset to read from
 * @param size Size of compressed data
 * @param jpegtable_data JPEG tables data (for TIFF JPEG)
 * @param jpegtable_count Size of JPEG tables
 * @param dest Output buffer pointer
 * @param out_device Output device ("cpu" or "cuda")
 * @param jpeg_color_space JPEG color space hint
 * @return true if successful
 */
bool decode_jpeg_nvimgcodec(int fd,
                            unsigned char* jpeg_buf,
                            uint64_t offset,
                            uint64_t size,
                            const void* jpegtable_data,
                            uint32_t jpegtable_count,
                            uint8_t** dest,
                            const cucim::io::Device& out_device,
                            int jpeg_color_space = 0);

/**
 * Decode JPEG2000 using nvImageCodec
 * 
 * @param fd File descriptor
 * @param jpeg2k_buf JPEG2000 buffer (if nullptr, read from fd at offset)
 * @param offset File offset to read from
 * @param size Size of compressed data
 * @param dest Output buffer pointer
 * @param dest_size Expected output size
 * @param out_device Output device ("cpu" or "cuda")
 * @param color_space Color space hint (RGB, YCbCr, etc.)
 * @return true if successful
 */
bool decode_jpeg2k_nvimgcodec(int fd,
                              unsigned char* jpeg2k_buf,
                              uint64_t offset,
                              uint64_t size,
                              uint8_t** dest,
                              size_t dest_size,
                              const cucim::io::Device& out_device,
                              int color_space = 0);

#ifdef CUCIM_HAS_NVIMGCODEC
// Forward declaration
struct IfdInfo;

/**
 * Decode an entire IFD using nvImageCodec
 * 
 * This function uses the parsed IfdInfo (from TiffFileParser) to decode
 * a full resolution level. It separates parsing from decoding.
 * 
 * @param ifd_info Parsed IFD information with sub_code_stream
 * @param output_buffer Pointer to receive allocated buffer (caller must free)
 * @param out_device Output device ("cpu" or "cuda")
 * @return true if successful, false otherwise
 */
bool decode_ifd_nvimgcodec(const IfdInfo& ifd_info,
                           uint8_t** output_buffer,
                           const cucim::io::Device& out_device);

/**
 * Decode a region of interest (ROI) from an IFD using nvImageCodec
 * 
 * Uses nvImageCodec's CodeStreamView with region specification for
 * memory-efficient decoding of specific image areas.
 * 
 * @param ifd_info Parsed IFD information with sub_code_stream
 * @param main_code_stream Main TIFF code stream (for creating ROI views)
 * @param x Starting x coordinate (column)
 * @param y Starting y coordinate (row)
 * @param width Width of region in pixels
 * @param height Height of region in pixels
 * @param output_buffer Pointer to receive allocated buffer (caller must free)
 * @param out_device Output device ("cpu" or "cuda")
 * @return true if successful, false otherwise
 */
bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height,
                                  uint8_t** output_buffer,
                                  const cucim::io::Device& out_device);
#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

#endif // CUSLIDE2_NVIMGCODEC_DECODER_H
