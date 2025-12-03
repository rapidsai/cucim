/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

// Forward declaration (needed in both cases)
struct IfdInfo;

#ifdef CUCIM_HAS_NVIMGCODEC
// nvImageCodec types are only available when CUCIM_HAS_NVIMGCODEC is defined
// (nvimgcodecCodeStream_t is defined in nvimgcodec.h)
#else
// When nvImageCodec is not available, provide a dummy type for the signature
typedef void* nvimgcodecCodeStream_t;
#endif

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
 * @param output_buffer Reference to pointer to receive allocated buffer (caller must free)
 * @param out_device Output device ("cpu" or "cuda")
 * @return true if successful, false otherwise
 *
 * @note When CUCIM_HAS_NVIMGCODEC is false, this function throws a runtime error.
 */
bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height,
                                  uint8_t*& output_buffer,
                                  const cucim::io::Device& out_device);

} // namespace cuslide2::nvimgcodec

#endif // CUSLIDE2_NVIMGCODEC_DECODER_H
