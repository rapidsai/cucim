/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUSLIDE2_NVIMGCODEC_DECODER_H
#define CUSLIDE2_NVIMGCODEC_DECODER_H

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <cucim/io/device.h>
#include <cstdint>
#include <vector>

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
 * Supports optional buffer allocation:
 * - If output_buffer is nullptr, the function allocates a buffer and sets it
 * - If output_buffer is non-null, the function decodes into the provided buffer
 *
 * @param ifd_info Parsed IFD information with sub_code_stream
 * @param main_code_stream Main TIFF code stream (for creating ROI views)
 * @param x Starting x coordinate (column)
 * @param y Starting y coordinate (row)
 * @param width Width of region in pixels
 * @param height Height of region in pixels
 * @param output_buffer Reference to buffer pointer; if null, will be allocated
 * @param out_device Output device ("cpu" or "cuda")
 * @return true if successful, false otherwise
 *
 * @note Caller is responsible for freeing the buffer (use cucim_free for CPU,
 *       cudaFree for CUDA)
 * @note When CUCIM_HAS_NVIMGCODEC is false, this function throws a runtime error.
 */
bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height,
                                  uint8_t*& output_buffer,
                                  const cucim::io::Device& out_device);

#ifdef CUCIM_HAS_NVIMGCODEC

/**
 * @brief ROI (Region of Interest) specification for batch decoding
 */
struct RoiRegion
{
    uint32_t x;          // Starting x coordinate (column)
    uint32_t y;          // Starting y coordinate (row)
    uint32_t width;      // Width of region in pixels
    uint32_t height;     // Height of region in pixels
};

/**
 * @brief Result of batch decoding for a single region
 */
struct BatchDecodeResult
{
    uint8_t* buffer;     // Decoded pixel data (caller must free)
    size_t buffer_size;  // Size of buffer in bytes
    bool success;        // Whether decoding succeeded
};

/**
 * Decode multiple regions of interest (ROIs) from a single IFD in a TIFF file
 *
 * Uses nvImageCodec v0.7.0+ batch decoding API:
 * 1. Read image into CodeStream (main_code_stream)
 * 2. Call get_sub_code_stream() for each ROI with different regions
 * 3. Decode all ROIs in a single decoder.decode() call
 *
 * This provides significant performance improvement over sequential decoding
 * by amortizing GPU kernel launch overhead and enabling parallel decoding.
 *
 * @param ifd_info IFD information (resolution level to decode from)
 * @param main_code_stream Main TIFF code stream (from TiffFileParser)
 * @param regions Vector of ROI specifications (all from the same IFD)
 * @param out_device Output device ("cpu" or "cuda")
 * @return Vector of decode results (same order as input regions)
 *
 * @note Caller is responsible for freeing buffers in successful results
 * @note All regions must be from the same IFD (resolution level)
 */
std::vector<BatchDecodeResult> decode_batch_regions_nvimgcodec(
    const IfdInfo& ifd_info,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device);

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

#endif // CUSLIDE2_NVIMGCODEC_DECODER_H
