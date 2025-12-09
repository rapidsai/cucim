/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvimgcodec_decoder.h"
#include "nvimgcodec_tiff_parser.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <cucim/memory/memory_manager.h>
#include <memory>
#include <vector>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>
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

// RAII wrapper for nvimgcodecCodeStream_t
struct CodeStreamDeleter
{
    void operator()(nvimgcodecCodeStream_t stream) const
    {
        if (stream) { nvimgcodecCodeStreamDestroy(stream); }
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

// RAII wrapper for nvimgcodecFuture_t
struct FutureDeleter
{
    void operator()(nvimgcodecFuture_t future) const
    {
        if (future) { nvimgcodecFutureDestroy(future); }
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
            // Use standard malloc for CPU memory
            // NOTE: Must use malloc() (not cudaMallocHost()) because cuCIM's cleanup
            // code uses free(). Using cudaMallocHost() would require cudaFreeHost(),
            // causing "free(): invalid pointer" crash when cuCIM calls free().
            buffer_ = malloc(size);
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
                free(buffer_);  // Standard free (matches malloc)
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
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);

        // Always use hybrid decoder - it supports more codecs and can output to both CPU/GPU
        // The CPU-only decoder has limited codec support (e.g., no JPEG in some builds)
        nvimgcodecDecoder_t decoder = manager.get_decoder();

        #ifdef DEBUG
        if (target_is_cpu)
        {
            fmt::print("  💡 Using hybrid decoder for CPU output (better codec support)\n");
        }
        else
        {
            fmt::print("  💡 Using hybrid decoder for GPU output\n");
        }
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

        // Get sub-code stream for this ROI (RAII managed)
        nvimgcodecCodeStream_t roi_stream_raw = nullptr;
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
            main_code_stream,
            &roi_stream_raw,
            &view
        );

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("❌ Failed to create ROI sub-stream (status: {})\n",
                      static_cast<int>(status));
            return false;
        }
        UniqueCodeStream roi_stream(roi_stream_raw);

        // Step 2: Determine buffer kind
        // IMPORTANT: nvImageCodec v0.6.0 GPU JPEG decoder can ONLY output to device memory
        // For CPU requests, we must decode to GPU then copy D2H (device-to-host)

        // Verify GPU is available
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);

        if (!gpu_available)
        {
            fmt::print("❌ GPU not available but required for nvImageCodec JPEG decoding\n");
            throw std::runtime_error(
                "nvImageCodec GPU JPEG decoder requires CUDA device. "
                "No CUDA device found.");
        }

        // Always decode to GPU device memory (GPU codec requirement)
        nvimgcodecImageBufferKind_t buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

        #ifdef DEBUG
        if (target_is_cpu)
        {
            fmt::print("  ℹ️  Decoding to GPU (will copy to CPU after decode)\n");
        }
        else
        {
            fmt::print("  ℹ️  Decoding to GPU\n");
        }
        #endif

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
            fmt::print("❌ Failed to allocate {} memory ({} bytes)\n",
                      use_device_memory ? "GPU" : "host", buffer_size);
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
            fmt::print("❌ Failed to create image object (status: {})\n",
                      static_cast<int>(status));
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
            fmt::print("❌ Failed to schedule decoding (status: {}, buffer_kind: {}, size: {}x{})\n",
                      static_cast<int>(status),
                      use_device_memory ? "GPU" : "CPU",
                      width, height);
            return false;  // RAII handles cleanup
        }
        UniqueFuture decode_future(decode_future_raw);

        // Step 8: Wait for completion
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        status = nvimgcodecFutureGetProcessingStatus(decode_future.get(), &decode_status, &status_size);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("❌ Failed to get processing status (status: {})\n", static_cast<int>(status));
            return false;  // RAII handles cleanup
        }

        if (use_device_memory)
        {
            cudaStreamSynchronize(output_image_info.cuda_stream);
        }

        // Step 9: Check decode status
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            fmt::print("❌ Decoding failed (processing_status: {}, buffer_kind: {}, size: {}x{})\n",
                      static_cast<int>(decode_status),
                      use_device_memory ? "GPU" : "CPU",
                      width, height);
            return false;  // RAII handles cleanup
        }

        #ifdef DEBUG
        fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
        #endif

        // Step 10: Handle D2H copy if user requested CPU output
        if (target_is_cpu)
        {
            // User requested CPU but we decoded to GPU - copy D2H
            size_t buffer_size_bytes = width * height * 3;  // RGB

            #ifdef DEBUG
            fmt::print("  ℹ️  Copying decoded data from GPU to CPU ({} bytes)...\n", buffer_size_bytes);
            #endif

            // Allocate CPU memory using cucim_malloc (standard malloc)
            // IMPORTANT: Do NOT use cudaMallocHost here!
            // cuCIM will free this buffer with cucim_free/free(), not cudaFreeHost()
            uint8_t* cpu_buffer = static_cast<uint8_t*>(cucim_malloc(buffer_size_bytes));
            if (!cpu_buffer)
            {
                fmt::print("❌ Failed to allocate CPU memory\n");
                return false;
            }

            // Copy from GPU to CPU
            cuda_err = cudaMemcpy(cpu_buffer, decode_buffer.get(), buffer_size_bytes, cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess)
            {
                fmt::print("❌ D2H copy failed: {}\n", cudaGetErrorString(cuda_err));
                cucim_free(cpu_buffer);
                return false;
            }

            #ifdef DEBUG
            fmt::print("  ✅ D2H copy completed\n");
            #endif

            // Return CPU buffer (decode_buffer GPU memory will be freed by RAII)
            output_buffer = cpu_buffer;
        }
        else
        {
            // GPU output: release buffer ownership to caller (skip RAII cleanup)
            output_buffer = reinterpret_cast<uint8_t*>(decode_buffer.release());
        }

        #ifdef DEBUG
        fmt::print("✅ nvImageCodec ROI decode successful: {}x{} at ({}, {})\n",
                  width, height, x, y);
        #endif
        return true;  // roi_stream, image, decode_future cleaned up by RAII
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception in ROI decoding: {}\n", e.what());
        return false;
    }
}

#else // !CUCIM_HAS_NVIMGCODEC

// Fallback stub when nvImageCodec is not available
// cuslide2 plugin requires nvImageCodec, so this should never be called
bool decode_ifd_region_nvimgcodec(const IfdInfo&,
                                  nvimgcodecCodeStream_t,
                                  uint32_t, uint32_t,
                                  uint32_t, uint32_t,
                                  uint8_t*&,
                                  const cucim::io::Device&)
{
    throw std::runtime_error("cuslide2 plugin requires nvImageCodec to be enabled at compile time");
}

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec
