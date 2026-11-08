/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUSLIDE2_NVIMGCODEC_DECODER_H
#define CUSLIDE2_NVIMGCODEC_DECODER_H

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#include <cuda_runtime.h>
#endif

#include <cucim/io/device.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuslide2::nvimgcodec
{

// Forward declaration (needed in both cases)
struct IfdInfo;

/**
 * Backend class that actually executed a decode.
 *
 * nvImageCodec 0.9 exposes backend filters and processing status, but does
 * not expose the selected decoder plugin identifier after generic dispatch.
 * cuCIM therefore restricts each decode to one backend class and records that
 * class exactly; it never guesses the plugin identifier.
 */
enum class DecodeBackend
{
    kUnknown,
    kCpuOnly,
    kGpuOnly,
    kHybridCpuGpu,
    kHardwareGpuOnly,
};

struct DecodeProvenance
{
    std::string codec;
    std::string plugin_id; // Empty when the nvImageCodec API cannot disclose it.
    DecodeBackend backend = DecodeBackend::kUnknown;
    uint32_t processing_status = 0;
    int api_status = 0;
    bool success = false;
    std::string execution_detail;
    std::string failure_reason;
    std::string output_memory; // "host" or "cuda"
    std::string cuda_stream; // Output stream: "none", "default", or "dedicated".
    bool future_wait_attempted = false;
    bool future_wait_completed = false;
    bool cuda_stream_synchronize_attempted = false;
    bool cuda_stream_synchronize_completed = false;
};

struct DecodeProvenanceReport
{
    std::vector<DecodeProvenance> samples;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
};

struct BackendProbe
{
    DecodeBackend backend = DecodeBackend::kUnknown;
    uint32_t processing_status = 0;
    bool decoder_available = false;
};

/** Select the first successful backend probe, or kUnknown when none succeeds. */
DecodeBackend select_decode_backend(const std::vector<BackendProbe>& probes);

const char* decode_backend_name(DecodeBackend backend);
const char* decode_execution_name(DecodeBackend backend, bool success);
std::string processing_status_name(uint32_t status);
std::string decode_provenance_json(const std::vector<DecodeProvenance>& samples,
                                   size_t cache_hits = 0,
                                   size_t cache_misses = 0);

#ifdef CUCIM_HAS_NVIMGCODEC
// nvImageCodec types are only available when CUCIM_HAS_NVIMGCODEC is defined
// (nvimgcodecCodeStream_t is defined in nvimgcodec.h)
// (cudaStream_t is defined in cuda_runtime.h)
#else
// When nvImageCodec is not available, provide a dummy type for the signature
typedef void* nvimgcodecCodeStream_t;
typedef void* cudaStream_t;
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
 * @param cuda_stream Optional CUDA stream for asynchronous execution (nullptr = default stream)
 * @return true if successful, false otherwise
 *
 * @note Caller is responsible for freeing the buffer (use cucim_free for CPU,
 *       cudaFree for CUDA)
 * @note When CUCIM_HAS_NVIMGCODEC is false, this function throws a runtime error.
 * @note If cuda_stream is nullptr, the default stream (0) will be used
 */
bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x,
                                  uint32_t y,
                                  uint32_t width,
                                  uint32_t height,
                                  uint8_t*& output_buffer,
                                  const cucim::io::Device& out_device,
                                  cudaStream_t cuda_stream = nullptr,
                                  DecodeProvenance* provenance = nullptr);

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
    DecodeProvenance provenance;
};

// Forward declaration for implementation details
struct BatchDecodeStateImpl;

/**
 * @brief State for asynchronous batch decoding
 *
 * This structure holds all the state needed to track an in-flight batch decode operation.
 * It is created by schedule_batch_decode() and used by wait_batch_decode().
 *
 * Uses pimpl pattern to hide implementation details and manage RAII resources.
 */
struct BatchDecodeState
{
    BatchDecodeState();
    ~BatchDecodeState();
    BatchDecodeState(BatchDecodeState&& other) noexcept;
    BatchDecodeState& operator=(BatchDecodeState&& other) noexcept;
    BatchDecodeState(const BatchDecodeState&) = delete;
    BatchDecodeState& operator=(const BatchDecodeState&) = delete;

    /**
     * @brief Check whether the decode was successfully scheduled.
     *
     * BatchDecodeState always allocates impl, so testing impl alone is
     * not sufficient.  A state is valid only when impl exists AND the
     * nvImageCodec future was created (i.e. the decode call succeeded).
     */
    bool is_valid() const;

    BatchDecodeStateImpl* impl;  // Implementation details
};

/**
 * Decode multiple regions of interest (ROIs) from a single IFD in a TIFF file
 *
 * Uses the nvImageCodec batch decoding API:
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
 * @param cuda_stream Optional CUDA stream for asynchronous execution (nullptr = default stream)
 * @return Vector of decode results (same order as input regions)
 *
 * @note Caller is responsible for freeing buffers in successful results
 * @note All regions must be from the same IFD (resolution level)
 * @note If cuda_stream is nullptr, the default stream (0) will be used
 */
std::vector<BatchDecodeResult> decode_batch_regions_nvimgcodec(
    const IfdInfo& ifd_info,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device,
    cudaStream_t cuda_stream = nullptr);

/**
 * Schedule batch decoding of multiple regions asynchronously.
 *
 * Prepares and schedules a batch decode operation but does not wait for
 * completion. Use wait_batch_decode() to wait for completion and get results.
 *
 * When @p output_buffers is non-empty the decoder writes directly into those
 * buffers (one per region, same order). The caller retains ownership; the
 * returned BatchDecodeState will NOT free them on destruction.
 *
 * When @p output_buffers is empty (default), the function allocates its own
 * buffers, which are freed by wait_batch_decode() or ~BatchDecodeState.
 *
 * @param ifd_info IFD information (resolution level to decode from)
 * @param main_code_stream Main TIFF code stream (from TiffFileParser)
 * @param regions Vector of ROI specifications (all from the same IFD)
 * @param out_device Output device ("cpu" or "cuda")
 * @param cuda_stream Optional CUDA stream for asynchronous execution (nullptr = default stream)
 * @param output_buffers Optional pre-allocated output buffers, one per region
 * @return BatchDecodeState containing the future and all necessary state
 *
 * @note The returned state must be passed to wait_batch_decode() to complete the operation.
 * @note Caller is responsible for cleaning up resources via wait_batch_decode().
 */
BatchDecodeState schedule_batch_decode(
    const IfdInfo& ifd_info,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device,
    cudaStream_t cuda_stream = nullptr,
    const std::vector<uint8_t*>& output_buffers = {});

/**
 * Wait for batch decode completion and process results
 *
 * This function waits for the decode operation scheduled by schedule_batch_decode()
 * to complete, then processes and returns the results.
 *
 * @param state Batch decode state returned from schedule_batch_decode()
 * @return Vector of decode results (same order as input regions)
 *
 * @note This function will clean up all resources in the state structure
 * @note Caller is responsible for freeing buffers in successful results
 */
std::vector<BatchDecodeResult> wait_batch_decode(BatchDecodeState& state);

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

#endif // CUSLIDE2_NVIMGCODEC_DECODER_H
