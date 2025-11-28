/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvimgcodec_decoder.h"
#include "nvimgcodec_tiff_parser.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <memory>
#include <vector>
#include <cstring>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include <mutex>
#include <fmt/format.h>

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
// Per nvImageCodec team: each code stream (parent or sub) has its own state
// and MUST be explicitly destroyed. Sub-streams are NOT automatically cleaned
// up when the parent is destroyed.
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
        if (image) nvimgcodecImageDestroy(image);
    }
};
using UniqueImage = std::unique_ptr<std::remove_pointer_t<nvimgcodecImage_t>, ImageDeleter>;

// RAII wrapper for nvimgcodecFuture_t
struct FutureDeleter
{
    void operator()(nvimgcodecFuture_t future) const
    {
        if (future) nvimgcodecFutureDestroy(future);
    }
};
using UniqueFuture = std::unique_ptr<std::remove_pointer_t<nvimgcodecFuture_t>, FutureDeleter>;

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
            // Use pinned memory for faster GPU-to-host transfers when GPU backend is used
            cudaError_t status = cudaMallocHost(&buffer_, size);
            return status == cudaSuccess;
        }
    }

    void reset()
    {
        if (buffer_)
        {
            if (is_device_)
                cudaFree(buffer_);
            else
                cudaFreeHost(buffer_);  // Pinned memory must use cudaFreeHost
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
                                  uint8_t** output_buffer,
                                  const cucim::io::Device& out_device)
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
        
        // Select decoder based on target device
        // CPU-only backend can handle in-bounds ROI decoding for TIFF files
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);
        
        // Check if ROI is out of bounds (extends beyond image boundaries)
        // CPU decoder doesn't support out-of-bounds ROI decoding, must use hybrid decoder
        bool roi_out_of_bounds = (x + width > ifd_info.width) || (y + height > ifd_info.height);
        if (target_is_cpu && roi_out_of_bounds)
        {
            target_is_cpu = false;  // Force hybrid decoder for out-of-bounds ROI
            #ifdef DEBUG
            fmt::print("  ⚠️  ROI out of bounds (region ends at [{},{}] but image is {}x{}), using hybrid decoder\n",
                      x + width, y + height, ifd_info.width, ifd_info.height);
            #endif
        }
        
        nvimgcodecDecoder_t decoder;
        if (target_is_cpu && manager.has_cpu_decoder())
        {
            decoder = manager.get_cpu_decoder();
            #ifdef DEBUG
            fmt::print("  💡 Using CPU-only decoder for ROI\n");
            #endif
        }
        else
        {
            decoder = manager.get_decoder();
            #ifdef DEBUG
            fmt::print("  💡 Using hybrid decoder for ROI\n");
            #endif
        }
        
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
        
        // Get sub-code stream for this ROI (RAII managed)
        nvimgcodecCodeStream_t roi_stream_raw = nullptr;
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
            main_code_stream,
            &roi_stream_raw,
            &view
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create ROI sub-stream (status: {})\n",
                      static_cast<int>(status));
            #endif
            return false;
        }
        // RAII wrapper - sub-stream will be properly destroyed when scope exits
        UniqueCodeStream roi_stream(roi_stream_raw);
        
        // Step 2: Determine buffer kind based on target device and decoder
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);
        
        nvimgcodecImageBufferKind_t buffer_kind;
        if (target_is_cpu)
        {
            // CPU target: use host buffer directly
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            #ifdef DEBUG
            fmt::print("  ℹ️  Using CPU buffer for ROI decoding\n");
            #endif
        }
        else if (gpu_available)
        {
            // GPU target with GPU available: use device buffer
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        }
        else
        {
            // GPU target but no GPU available: fall back to host buffer
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
        
        // Use interleaved RGB format
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;
        output_image_info.buffer_kind = buffer_kind;
        
        // Calculate buffer requirements for the region
        uint32_t num_channels = 3;  // RGB
        size_t row_stride = width * num_channels;
        size_t buffer_size = row_stride * height;
        
        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        // Note: buffer_size removed in nvImageCodec v0.7.0 - size is inferred from plane_info
        output_image_info.cuda_stream = 0;
        
        #ifdef DEBUG
        fmt::print("  Buffer: {}x{} RGB, stride={}, size={} bytes\n",
                  width, height, row_stride, buffer_size);
        #endif
        
        // Step 4: Allocate output buffer (RAII managed)
        bool use_device_memory = (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE);
        DecodeBuffer decode_buffer;
        if (!decode_buffer.allocate(buffer_size, use_device_memory))
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to allocate {} memory\n", use_device_memory ? "GPU" : "host");
            #endif
            return false;
        }
        #ifdef DEBUG
        fmt::print("  Allocated {} buffer\n", use_device_memory ? "GPU" : "CPU");
        #endif
        
        output_image_info.buffer = decode_buffer.get();
        
        // Step 5: Create image object (RAII managed)
        nvimgcodecImage_t image_raw = nullptr;
        status = nvimgcodecImageCreate(
            manager.get_instance(),
            &image_raw,
            &output_image_info
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create image object (status: {})\n",
                      static_cast<int>(status));
            #endif
            return false;  // RAII handles cleanup
        }
        UniqueImage image(image_raw);
        
        // Step 6: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 7: Schedule decoding (RAII managed)
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
            fmt::print("❌ Failed to schedule decoding (status: {})\n",
                      static_cast<int>(status));
            #endif
            return false;  // RAII handles cleanup
        }
        UniqueFuture decode_future(decode_future_raw);
        
        // Step 8: Wait for completion
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        status = nvimgcodecFutureGetProcessingStatus(decode_future.get(), &decode_status, &status_size);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to get processing status (status: {})\n", static_cast<int>(status));
            #endif
            return false;  // RAII handles cleanup
        }
        
        if (use_device_memory)
        {
            cudaDeviceSynchronize();
        }
        
        // Step 9: Check decode status
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Decoding failed (status: {})\n", static_cast<int>(decode_status));
            #endif
            return false;  // RAII handles cleanup
        }
        
        #ifdef DEBUG
        fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
        #endif
        
        // Success: release buffer ownership to caller (RAII cleanup skipped for buffer)
        *output_buffer = reinterpret_cast<uint8_t*>(decode_buffer.release());
        #ifdef DEBUG
        fmt::print("✅ nvImageCodec ROI decode successful: {}x{} at ({}, {})\n", 
                  width, height, x, y);
        #endif
        return true;  // roi_stream, image, decode_future all cleaned up by RAII
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in ROI decoding: {}\n", e.what());
        #endif
        return false;
    }
}

#else // !CUCIM_HAS_NVIMGCODEC

// Fallback stub when nvImageCodec is not available
// cuslide2 plugin requires nvImageCodec, so this should never be called
// Forward declaration for types
struct IfdInfo;
typedef void* nvimgcodecCodeStream_t;

bool decode_ifd_region_nvimgcodec(const IfdInfo&,
                                  nvimgcodecCodeStream_t,
                                  uint32_t, uint32_t,
                                  uint32_t, uint32_t,
                                  uint8_t**,
                                  const cucim::io::Device&)
{
    throw std::runtime_error("cuslide2 plugin requires nvImageCodec to be enabled at compile time");
}

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec
