/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvimgcodec_decoder.h"
#include "nvimgcodec_tiff_parser.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <memory>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include <mutex>
#include <fmt/format.h>

#include <cucim/memory/memory_manager.h>

#ifdef CUCIM_HAS_NVIMGCODEC
#include <cuda_runtime.h>
#endif

namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

// ============================================================================
// RAII Helpers for nvImageCodec Resources
// ============================================================================

// RAII wrapper for nvimgcodecCodeStream_t (including sub-code streams)
// Each code stream (parent or sub) must be explicitly destroyed. Sub-streams are
// not automatically cleaned up when the parent is destroyed.
struct CodeStreamDeleter
{
    void operator()(nvimgcodecCodeStream_t stream) const
    {
        if (stream)
        {
            nvimgcodecCodeStreamDestroy(stream);
        }
    }
};
using UniqueCodeStream = std::unique_ptr<std::remove_pointer_t<nvimgcodecCodeStream_t>, CodeStreamDeleter>;

// RAII wrapper for nvimgcodecImage_t
struct ImageDeleter
{
    void operator()(nvimgcodecImage_t image) const
    {
        if (image) { nvimgcodecImageDestroy(image); }
    }
};
using UniqueImage = std::unique_ptr<std::remove_pointer_t<nvimgcodecImage_t>, ImageDeleter>;

// ============================================================================
// BatchDecodeState Implementation (pimpl pattern)
// ============================================================================

struct BatchDecodeStateImpl
{
    nvimgcodecFuture_t future = nullptr;
    std::vector<UniqueCodeStream> roi_streams_raii;  // RAII wrappers for streams
    std::vector<UniqueImage> images_raii;            // RAII wrappers for images
    std::vector<nvimgcodecCodeStream_t> roi_streams; // Raw pointers for decode call
    std::vector<nvimgcodecImage_t> images;            // Raw pointers for decode call
    std::vector<void*> buffers;                       // Allocated buffers
    std::vector<size_t> buffer_sizes;                // Size of each buffer
    std::vector<size_t> valid_indices;                // Indices of valid regions
    size_t batch_size = 0;
    bool use_device_memory = false;
    cudaStream_t cuda_stream = nullptr;
};

BatchDecodeState::BatchDecodeState() : impl(new BatchDecodeStateImpl()) {}
BatchDecodeState::~BatchDecodeState() { delete impl; }

BatchDecodeState::BatchDecodeState(BatchDecodeState&& other) noexcept
    : impl(other.impl)
{
    other.impl = nullptr;
}

BatchDecodeState& BatchDecodeState::operator=(BatchDecodeState&& other) noexcept
{
    if (this != &other)
    {
        delete impl;
        impl = other.impl;
        other.impl = nullptr;
    }
    return *this;
}

// RAII wrapper for nvimgcodecFuture_t
struct FutureDeleter
{
    void operator()(nvimgcodecFuture_t future) const
    {
        if (future) { nvimgcodecFutureDestroy(future); }
    }
};
using UniqueFuture = std::unique_ptr<std::remove_pointer_t<nvimgcodecFuture_t>, FutureDeleter>;

// RAII wrapper for nvimgcodecCodeStream_t (for batch decode state)
struct CodeStreamPtrDeleter
{
    void operator()(nvimgcodecCodeStream_t* streams) const
    {
        if (streams)
        {
            // Note: Individual streams are managed by UniqueCodeStream in the state
            delete[] streams;
        }
    }
};

// RAII wrapper for nvimgcodecImage_t (for batch decode state)
struct ImagePtrDeleter
{
    void operator()(nvimgcodecImage_t* images) const
    {
        if (images)
        {
            // Note: Individual images are managed by UniqueImage in the state
            delete[] images;
        }
    }
};

// RAII wrapper for decode buffer (handles both CPU and GPU memory)
class DecodeBuffer
{
public:
    DecodeBuffer() = default;
    ~DecodeBuffer() { reset(); }

    // Non-copyable
    DecodeBuffer(const DecodeBuffer&) = delete;
    DecodeBuffer& operator=(const DecodeBuffer&) = delete;

    // Movable
    DecodeBuffer(DecodeBuffer&& other) noexcept
        : buffer_(other.buffer_), is_device_(other.is_device_)
    {
        other.buffer_ = nullptr;
    }

    DecodeBuffer& operator=(DecodeBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            buffer_ = other.buffer_;
            is_device_ = other.is_device_;
            other.buffer_ = nullptr;
        }
        return *this;
    }

    bool allocate(size_t size, bool device_memory)
    {
        reset();
        is_device_ = device_memory;
        if (device_memory)
        {
            cudaError_t status = cudaMalloc(&buffer_, size);
            return status == cudaSuccess;
        }
        else
        {
            // Use cucim_malloc for CPU memory (consistent with cuCIM memory management)
            // NOTE: Must NOT use cudaMallocHost() because cuCIM's cleanup expects
            // standard heap memory, which would cause "free(): invalid pointer" crash.
            buffer_ = cucim_malloc(size);
            return buffer_ != nullptr;
        }
    }

    void reset()
    {
        if (buffer_)
        {
            if (is_device_)
                cudaFree(buffer_);
            else
                cucim_free(buffer_);  // Matches cucim_malloc
            buffer_ = nullptr;
        }
    }

    void* get() const { return buffer_; }
    bool is_device() const { return is_device_; }

    // Release ownership (for passing to caller)
    void* release()
    {
        void* tmp = buffer_;
        buffer_ = nullptr;
        return tmp;
    }

private:
    void* buffer_ = nullptr;
    bool is_device_ = false;
};

// ============================================================================
// IFD-Level Region Decoding (Primary Decode Function)
// ============================================================================

bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height,
                                  uint8_t*& output_buffer,
                                  const cucim::io::Device& out_device,
                                  cudaStream_t cuda_stream)
{
    if (!main_code_stream)
    {
        #ifdef DEBUG
        fmt::print("❌ Invalid main_code_stream\n");
        #endif
        return false;
    }

    #ifdef DEBUG
    fmt::print("🚀 Decoding IFD[{}] region: [{},{}] {}x{}, codec: {}\n",
              ifd_info.index, x, y, width, height, ifd_info.codec);
    #endif

    try
    {
        // CRITICAL: Must use the same manager that created main_code_stream!
        // Using a decoder from a different nvImageCodec instance causes segfaults.
        auto& manager = NvImageCodecTiffParserManager::instance();
        if (!manager.is_available())
        {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec TIFF parser manager not initialized\n");
            #endif
            return false;
        }

        // Caller-provided output buffer (optional). If non-null, we decode into it.
        const bool caller_provided_buffer = (output_buffer != nullptr);

        // Determine target device for buffer placement
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);

        // Always use the primary (GPU-enabled) decoder.  For CPU output requests
        // the buffer_kind is set to STRIDED_HOST below, and nvImageCodec handles
        // the GPU decode → host copy internally.  The CPU-only decoder lacks the
        // GPU backends needed for JPEG-compressed TIFF, so we don't use it here.
        nvimgcodecDecoder_t decoder = manager.get_decoder();

        #ifdef DEBUG
        fmt::print("  💡 Using primary decoder (target: {})\n", target_is_cpu ? "CPU" : "GPU");
        #endif

        // Step 1: Create view with ROI for this IFD
        nvimgcodecRegion_t region{};
        region.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
        region.struct_size = sizeof(nvimgcodecRegion_t);
        region.struct_next = nullptr;
        region.ndim = 2;
        region.start[0] = y;  // row
        region.start[1] = x;  // col
        region.end[0] = y + height;
        region.end[1] = x + width;

        nvimgcodecCodeStreamView_t view{};
        view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
        view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
        view.struct_next = nullptr;
        view.image_idx = ifd_info.index;
        view.region = region;

        nvimgcodecCodeStream_t roi_stream_raw = nullptr;
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(main_code_stream,
                                                                         &roi_stream_raw,
                                                                         &view);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create ROI sub-stream (status: {})\n", static_cast<int>(status));
            #endif
            return false;
        }
        UniqueCodeStream roi_stream(roi_stream_raw);

        // Step 2: Determine buffer kind based on target device and availability
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);

        nvimgcodecImageBufferKind_t buffer_kind;
        if (target_is_cpu)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        else if (gpu_available)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        }
        else
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            #ifdef DEBUG
            fmt::print("  ⚠️  No GPU available, using CPU buffer\n");
            #endif
        }

        // Step 3: Prepare output image info for the region
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;

        // Use UNCHANGED to preserve original format (RGB or grayscale)
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;
        output_image_info.buffer_kind = buffer_kind;

        uint32_t num_channels = ifd_info.num_channels > 0 ? ifd_info.num_channels : 3;
        size_t row_stride = width * num_channels;
        size_t buffer_size = row_stride * height;

        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        // Note: buffer_size removed in nvImageCodec v0.7.0 - size is inferred from plane_info
        // Use provided CUDA stream, or default to 0 (default stream) if nullptr
        output_image_info.cuda_stream = (cuda_stream != nullptr) ? cuda_stream : cudaStream_t(0);

        // Step 4: Provide output buffer
        bool use_device_memory = (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE);
        DecodeBuffer decode_buffer;
        if (caller_provided_buffer)
        {
            // Query actual memory type of caller-provided buffer
            cudaPointerAttributes attrs{};
            cudaError_t ptr_err = cudaPointerGetAttributes(&attrs, output_buffer);
            if (ptr_err == cudaSuccess)
            {
                // Determine buffer kind based on actual memory type
                bool is_device_buffer = (attrs.type == cudaMemoryTypeDevice ||
                                         attrs.type == cudaMemoryTypeManaged);

                if (is_device_buffer)
                {
                    buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                    use_device_memory = true;
                }
                else
                {
                    buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
                    use_device_memory = false;
                }

                #ifdef DEBUG
                fmt::print("  ℹ️  Caller buffer detected as {} memory\n",
                          use_device_memory ? "device" : "host");
                #endif
            }
            // If cudaPointerGetAttributes fails, assume host memory (safe default)
            output_image_info.buffer = output_buffer;
            output_image_info.buffer_kind = buffer_kind;
        }
        else
        {
            if (!decode_buffer.allocate(buffer_size, use_device_memory))
            {
                #ifdef DEBUG
                fmt::print("❌ Failed to allocate {} memory\n", use_device_memory ? "GPU" : "host");
                #endif
                return false;
            }
            output_image_info.buffer = decode_buffer.get();
        }

        // Step 5: Create image
        nvimgcodecImage_t image_raw = nullptr;
        status = nvimgcodecImageCreate(manager.get_instance(), &image_raw, &output_image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create image object (status: {})\n", static_cast<int>(status));
            #endif
            return false;
        }
        UniqueImage image(image_raw);

        // Step 6: Decode
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;

        nvimgcodecCodeStream_t roi_stream_ptr = roi_stream.get();
        nvimgcodecImage_t image_ptr = image.get();
        nvimgcodecFuture_t decode_future_raw = nullptr;
        status = nvimgcodecDecoderDecode(decoder,
                                         &roi_stream_ptr,
                                         &image_ptr,
                                         1,
                                         &decode_params,
                                         &decode_future_raw);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule decoding (status: {})\n", static_cast<int>(status));
            #endif
            return false;
        }
        UniqueFuture decode_future(decode_future_raw);

        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        status = nvimgcodecFutureGetProcessingStatus(decode_future.get(), &decode_status, &status_size);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to get processing status (status: {})\n", static_cast<int>(status));
            #endif
            return false;
        }

        if (use_device_memory)
        {
            // Synchronize the stream used for decoding
            cudaStream_t stream_to_sync = (cuda_stream != nullptr) ? cuda_stream : cudaStream_t(0);
            cudaStreamSynchronize(stream_to_sync);
        }

        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Decoding failed (status: {})\n", static_cast<int>(decode_status));
            #endif
            return false;
        }

        // Success: return buffer (only if we allocated it)
        if (!caller_provided_buffer)
        {
            output_buffer = reinterpret_cast<uint8_t*>(decode_buffer.release());
        }
        return true;
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in ROI decoding: {}\n", e.what());
        #endif
        return false;
    }
}

// ============================================================================
// Batch ROI Decoding (nvImageCodec v0.7.0+)
// ============================================================================

std::vector<BatchDecodeResult> decode_batch_regions_nvimgcodec(
    const IfdInfo& ifd_info,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device,
    cudaStream_t cuda_stream)
{
    const size_t batch_size = regions.size();
    std::vector<BatchDecodeResult> results(batch_size);

    // Initialize all results to failure
    for (auto& result : results)
    {
        result.buffer = nullptr;
        result.buffer_size = 0;
        result.success = false;
    }

    if (batch_size == 0)
    {
        return results;
    }

    if (!main_code_stream)
    {
        #ifdef DEBUG
        fmt::print("❌ Batch decode: Invalid main_code_stream\n");
        #endif
        return results;
    }

    #ifdef DEBUG
    fmt::print("🚀 Batch decoding {} regions\n", batch_size);
    #endif

    try
    {
        auto& manager = NvImageCodecTiffParserManager::instance();
        if (!manager.is_available())
        {
            #ifdef DEBUG
            fmt::print("❌ Batch decode: nvImageCodec manager not initialized\n");
            #endif
            return results;
        }

        // Determine target device
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);

        // Check GPU availability
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);

        nvimgcodecImageBufferKind_t buffer_kind;
        bool use_device_memory;
        if (target_is_cpu)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            use_device_memory = false;
        }
        else if (gpu_available)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            use_device_memory = true;
        }
        else
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            use_device_memory = false;
            #ifdef DEBUG
            fmt::print("  ⚠️  No GPU available, using CPU buffers for batch\n");
            #endif
        }

        // Always use the primary GPU-enabled decoder (handles both GPU and
        // CPU output via buffer_kind).
        nvimgcodecDecoder_t decoder = manager.get_decoder();

        // Step 1: Create ROI sub-streams for each region

        std::vector<UniqueCodeStream> roi_streams;
        roi_streams.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i)
        {
            const auto& region = regions[i];

            nvimgcodecRegion_t nvregion{};
            nvregion.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
            nvregion.struct_size = sizeof(nvimgcodecRegion_t);
            nvregion.struct_next = nullptr;
            nvregion.ndim = 2;
            nvregion.start[0] = region.y;  // row
            nvregion.start[1] = region.x;  // col
            nvregion.end[0] = region.y + region.height;
            nvregion.end[1] = region.x + region.width;

            nvimgcodecCodeStreamView_t view{};
            view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
            view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
            view.struct_next = nullptr;
            view.image_idx = ifd_info.index;  // Use IFD index for nvImageCodec page selection
            view.region = nvregion;

            nvimgcodecCodeStream_t roi_stream_raw = nullptr;
            nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
                main_code_stream, &roi_stream_raw, &view);

            if (status != NVIMGCODEC_STATUS_SUCCESS)
            {
                #ifdef DEBUG
                fmt::print("  ⚠️  Failed to create ROI sub-stream for region {} (status: {})\n",
                          i, static_cast<int>(status));
                #endif
                roi_streams.emplace_back(nullptr);
                continue;
            }

            roi_streams.emplace_back(roi_stream_raw);

            #ifdef DEBUG
            fmt::print("  ✅ Created ROI sub-stream for region {} (IFD[{}] [{},{}] {}x{})\n",
                      i, ifd_info.index, region.x, region.y, region.width, region.height);
            #endif
        }

        // Step 2: Allocate output buffers and create image objects
        std::vector<DecodeBuffer> decode_buffers(batch_size);
        std::vector<UniqueImage> images;
        images.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i)
        {
            if (!roi_streams[i])
            {
                images.emplace_back(nullptr);
                continue;
            }

            const auto& region = regions[i];
            // Use num_channels from IFD info, fallback to 3 (RGB)
            uint32_t num_channels = (ifd_info.num_channels > 0) ? ifd_info.num_channels : 3;
            size_t row_stride = region.width * num_channels;
            size_t buffer_size = row_stride * region.height;

            if (!decode_buffers[i].allocate(buffer_size, use_device_memory))
            {
                #ifdef DEBUG
                fmt::print("  ⚠️  Failed to allocate buffer for region {}\n", i);
                #endif
                images.emplace_back(nullptr);
                continue;
            }

            nvimgcodecImageInfo_t output_image_info{};
            output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
            output_image_info.struct_next = nullptr;
            // Use UNCHANGED to preserve original format (RGB or grayscale)
            output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
            output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
            output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
            output_image_info.num_planes = 1;
            output_image_info.buffer_kind = buffer_kind;
            output_image_info.buffer = decode_buffers[i].get();
            output_image_info.plane_info[0].height = region.height;
            output_image_info.plane_info[0].width = region.width;
            output_image_info.plane_info[0].num_channels = num_channels;
            output_image_info.plane_info[0].row_stride = row_stride;
            output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            // Use provided CUDA stream, or default to 0 (default stream) if nullptr
            output_image_info.cuda_stream = (cuda_stream != nullptr) ? cuda_stream : cudaStream_t(0);

            nvimgcodecImage_t image_raw = nullptr;
            nvimgcodecStatus_t status = nvimgcodecImageCreate(manager.get_instance(), &image_raw, &output_image_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS)
            {
                #ifdef DEBUG
                fmt::print("  ⚠️  Failed to create image object for region {}\n", i);
                #endif
                images.emplace_back(nullptr);
                continue;
            }

            images.emplace_back(image_raw);
            results[i].buffer_size = buffer_size;

            #ifdef DEBUG
            fmt::print("  ✅ Created image object for region {} ({}x{}, {} bytes)\n",
                      i, region.width, region.height, buffer_size);
            #endif
        }

        // Step 3: Filter out invalid entries for batch decode
        std::vector<nvimgcodecCodeStream_t> valid_streams;
        std::vector<nvimgcodecImage_t> valid_images;
        std::vector<size_t> valid_indices;
        valid_streams.reserve(batch_size);
        valid_images.reserve(batch_size);
        valid_indices.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i)
        {
            if (roi_streams[i] && images[i])
            {
                valid_streams.push_back(roi_streams[i].get());
                valid_images.push_back(images[i].get());
                valid_indices.push_back(i);
            }
        }

        if (valid_streams.empty())
        {
            #ifdef DEBUG
            fmt::print("❌ No valid regions to decode\n");
            #endif
            return results;
        }

        #ifdef DEBUG
        fmt::print("  📦 Batch decoding {} valid regions (out of {})\n",
                  valid_streams.size(), batch_size);
        #endif

        // Step 4: Single batch decode call (the key optimization!)
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;

        nvimgcodecFuture_t decode_future_raw = nullptr;
        nvimgcodecStatus_t status = nvimgcodecDecoderDecode(
            decoder,
            valid_streams.data(),
            valid_images.data(),
            static_cast<int>(valid_streams.size()),
            &decode_params,
            &decode_future_raw);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule batch decoding (status: {})\n", static_cast<int>(status));
            #endif
            return results;
        }
        UniqueFuture decode_future(decode_future_raw);

        // Step 5: Get processing status for each image
        std::vector<nvimgcodecProcessingStatus_t> decode_statuses(valid_streams.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        size_t status_size = valid_streams.size();
        status = nvimgcodecFutureGetProcessingStatus(decode_future.get(), decode_statuses.data(), &status_size);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to get batch processing status (status: {})\n", static_cast<int>(status));
            #endif
            return results;
        }

        // Synchronize if using GPU (use stream sync instead of device sync for better performance)
        if (use_device_memory)
        {
            cudaStream_t stream_to_sync = (cuda_stream != nullptr) ? cuda_stream : cudaStream_t(0);
            cudaStreamSynchronize(stream_to_sync);
        }

        // Step 6: Transfer successful results to output
        for (size_t vi = 0; vi < valid_indices.size(); ++vi)
        {
            size_t i = valid_indices[vi];
            if (decode_statuses[vi] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
            {
                results[i].buffer = reinterpret_cast<uint8_t*>(decode_buffers[i].release());
                results[i].success = true;
                #ifdef DEBUG
                fmt::print("  ✅ Region {} decoded successfully\n", i);
                #endif
            }
            else
            {
                #ifdef DEBUG
                fmt::print("  ❌ Region {} decode failed (status: {})\n",
                          i, static_cast<int>(decode_statuses[vi]));
                #endif
            }
        }

        #ifdef DEBUG
        size_t success_count = 0;
        for (const auto& r : results) if (r.success) ++success_count;
        fmt::print("✅ Batch decode complete: {}/{} regions successful\n", success_count, batch_size);
        #endif

        return results;
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in batch decoding: {}\n", e.what());
        #endif
        return results;
    }
}

// ============================================================================
// Split Batch Decoding: Schedule and Wait Separately
// ============================================================================

BatchDecodeState schedule_batch_decode(
    const IfdInfo& ifd_info,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device,
    cudaStream_t cuda_stream)
{
    BatchDecodeState state;
    auto* impl = state.impl;
    impl->batch_size = regions.size();
    impl->use_device_memory = false;
    impl->cuda_stream = (cuda_stream != nullptr) ? cuda_stream : cudaStream_t(0);

    if (impl->batch_size == 0 || !main_code_stream)
    {
        return state;
    }

    try
    {
        auto& manager = NvImageCodecTiffParserManager::instance();
        if (!manager.is_available())
        {
            #ifdef DEBUG
            fmt::print("❌ Schedule batch: nvImageCodec manager not initialized\n");
            #endif
            return state;
        }

        // Determine target device
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);

        // Check GPU availability
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);

        nvimgcodecImageBufferKind_t buffer_kind;
        if (target_is_cpu)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            impl->use_device_memory = false;
        }
        else if (gpu_available)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            impl->use_device_memory = true;
        }
        else
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            impl->use_device_memory = false;
        }

        // Always use the primary GPU-enabled decoder
        nvimgcodecDecoder_t decoder = manager.get_decoder();

        // Step 1: Create ROI sub-streams for each region
        impl->roi_streams_raii.reserve(impl->batch_size);

        for (size_t i = 0; i < impl->batch_size; ++i)
        {
            const auto& region = regions[i];

            nvimgcodecRegion_t nvregion{};
            nvregion.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
            nvregion.struct_size = sizeof(nvimgcodecRegion_t);
            nvregion.struct_next = nullptr;
            nvregion.ndim = 2;
            nvregion.start[0] = region.y;
            nvregion.start[1] = region.x;
            nvregion.end[0] = region.y + region.height;
            nvregion.end[1] = region.x + region.width;

            nvimgcodecCodeStreamView_t view{};
            view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
            view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
            view.struct_next = nullptr;
            view.image_idx = ifd_info.index;
            view.region = nvregion;

            nvimgcodecCodeStream_t roi_stream_raw = nullptr;
            nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
                main_code_stream, &roi_stream_raw, &view);

            if (status != NVIMGCODEC_STATUS_SUCCESS)
            {
                impl->roi_streams_raii.emplace_back(nullptr);
                continue;
            }

            impl->roi_streams_raii.emplace_back(roi_stream_raw);
        }

        // Step 2: Allocate output buffers and create image objects
        std::vector<DecodeBuffer> decode_buffers(impl->batch_size);
        impl->images_raii.reserve(impl->batch_size);

        for (size_t i = 0; i < impl->batch_size; ++i)
        {
            if (!impl->roi_streams_raii[i])
            {
                impl->images_raii.emplace_back(nullptr);
                continue;
            }

            const auto& region = regions[i];
            uint32_t num_channels = (ifd_info.num_channels > 0) ? ifd_info.num_channels : 3;
            size_t row_stride = region.width * num_channels;
            size_t buffer_size = row_stride * region.height;

            if (!decode_buffers[i].allocate(buffer_size, impl->use_device_memory))
            {
                impl->images_raii.emplace_back(nullptr);
                continue;
            }

            nvimgcodecImageInfo_t output_image_info{};
            output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
            output_image_info.struct_next = nullptr;
            output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
            output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
            output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
            output_image_info.num_planes = 1;
            output_image_info.buffer_kind = buffer_kind;
            output_image_info.buffer = decode_buffers[i].get();
            output_image_info.plane_info[0].height = region.height;
            output_image_info.plane_info[0].width = region.width;
            output_image_info.plane_info[0].num_channels = num_channels;
            output_image_info.plane_info[0].row_stride = row_stride;
            output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            output_image_info.cuda_stream = impl->cuda_stream;

            nvimgcodecImage_t image_raw = nullptr;
            nvimgcodecStatus_t status = nvimgcodecImageCreate(manager.get_instance(), &image_raw, &output_image_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS)
            {
                impl->images_raii.emplace_back(nullptr);
                continue;
            }

            impl->images_raii.emplace_back(image_raw);
            impl->buffer_sizes.push_back(buffer_size);
        }

        // Step 3: Filter out invalid entries and prepare for batch decode
        for (size_t i = 0; i < impl->batch_size; ++i)
        {
            if (impl->roi_streams_raii[i] && impl->images_raii[i])
            {
                impl->roi_streams.push_back(impl->roi_streams_raii[i].get());
                impl->images.push_back(impl->images_raii[i].get());
                impl->valid_indices.push_back(i);
                impl->buffers.push_back(decode_buffers[i].release()); // Transfer ownership
            }
        }

        if (impl->roi_streams.empty())
        {
            return state;
        }

        // Step 4: Schedule batch decode (asynchronous)
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;

        nvimgcodecStatus_t status = nvimgcodecDecoderDecode(
            decoder,
            impl->roi_streams.data(),
            impl->images.data(),
            static_cast<int>(impl->roi_streams.size()),
            &decode_params,
            &impl->future);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule batch decoding (status: {})\n", static_cast<int>(status));
            #endif
            // Clean up on failure
            for (void* buf : impl->buffers)
            {
                if (buf)
                {
                    if (impl->use_device_memory)
                        cudaFree(buf);
                    else
                        cucim_free(buf);
                }
            }
            impl->buffers.clear();
            impl->roi_streams.clear();
            impl->images.clear();
            return state;
        }

        #ifdef DEBUG
        fmt::print("✅ Scheduled batch decode of {} regions\n", impl->roi_streams.size());
        #endif
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in schedule_batch_decode: {}\n", e.what());
        #endif
    }

    return state;
}

std::vector<BatchDecodeResult> wait_batch_decode(BatchDecodeState& state)
{
    auto* impl = state.impl;
    std::vector<BatchDecodeResult> results(impl->batch_size);

    // Initialize all results to failure
    for (auto& result : results)
    {
        result.buffer = nullptr;
        result.buffer_size = 0;
        result.success = false;
    }

    if (!impl->future || impl->roi_streams.empty())
    {
        return results;
    }

    try
    {
        // Wait for all decode operations to complete
        nvimgcodecStatus_t status = nvimgcodecFutureWaitForAll(impl->future);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to wait for batch decode (status: {})\n", static_cast<int>(status));
            #endif
            // Clean up resources
            nvimgcodecFutureDestroy(impl->future);
            for (void* buf : impl->buffers)
            {
                if (buf)
                {
                    if (impl->use_device_memory)
                        cudaFree(buf);
                    else
                        cucim_free(buf);
                }
            }
            return results;
        }

        // Synchronize stream if using GPU
        if (impl->use_device_memory)
        {
            cudaStreamSynchronize(impl->cuda_stream);
        }

        // Get processing status for each image
        std::vector<nvimgcodecProcessingStatus_t> decode_statuses(impl->roi_streams.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        size_t status_size = impl->roi_streams.size();
        status = nvimgcodecFutureGetProcessingStatus(impl->future, decode_statuses.data(), &status_size);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to get batch processing status (status: {})\n", static_cast<int>(status));
            #endif
        }

        // Process results
        for (size_t vi = 0; vi < impl->valid_indices.size(); ++vi)
        {
            size_t i = impl->valid_indices[vi];
            if (vi < decode_statuses.size() && decode_statuses[vi] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
            {
                if (vi < impl->buffers.size() && impl->buffers[vi])
                {
                    results[i].buffer = reinterpret_cast<uint8_t*>(impl->buffers[vi]);
                    results[i].buffer_size = (vi < impl->buffer_sizes.size()) ? impl->buffer_sizes[vi] : 0;
                    results[i].success = true;
                    // Transfer ownership - clear from state so it won't be freed
                    impl->buffers[vi] = nullptr;
                }
            }
        }

        // Clean up future
        nvimgcodecFutureDestroy(impl->future);
        impl->future = nullptr;

        // Clean up any remaining buffers (failed decodes)
        for (void* buf : impl->buffers)
        {
            if (buf)
            {
                if (impl->use_device_memory)
                    cudaFree(buf);
                else
                    cucim_free(buf);
            }
        }
        impl->buffers.clear();
        impl->roi_streams.clear();
        impl->images.clear();
        impl->roi_streams_raii.clear();
        impl->images_raii.clear();

        #ifdef DEBUG
        size_t success_count = 0;
        for (const auto& r : results) if (r.success) ++success_count;
        fmt::print("✅ Batch decode complete: {}/{} regions successful\n", success_count, impl->batch_size);
        #endif
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in wait_batch_decode: {}\n", e.what());
        #endif
    }

    return results;
}

#else // !CUCIM_HAS_NVIMGCODEC

// Fallback stub when nvImageCodec is not available
// cuslide2 plugin requires nvImageCodec, so this should never be called
bool decode_ifd_region_nvimgcodec(const IfdInfo&,
                                  nvimgcodecCodeStream_t,
                                  uint32_t, uint32_t,
                                  uint32_t, uint32_t,
                                  uint8_t*&,
                                  const cucim::io::Device&,
                                  cudaStream_t)
{
    throw std::runtime_error("cuslide2 plugin requires nvImageCodec to be enabled at compile time");
}

std::vector<BatchDecodeResult> decode_batch_regions_nvimgcodec(
    const IfdInfo&,
    nvimgcodecCodeStream_t,
    const std::vector<RoiRegion>&,
    const cucim::io::Device&,
    cudaStream_t)
{
    throw std::runtime_error("cuslide2 plugin requires nvImageCodec to be enabled at compile time");
}

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec
