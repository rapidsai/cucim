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

#include "nvimgcodec_decoder.h"
#include "nvimgcodec_tiff_parser.h"
#include "nvimgcodec_manager.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

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
        fmt::print("❌ Invalid main_code_stream\n");
        return false;
    }
    
    fmt::print("🚀 Decoding IFD[{}] region: [{},{}] {}x{}, codec: {}\n",
              ifd_info.index, x, y, width, height, ifd_info.codec);
    
    try
    {
        // CRITICAL: Must use the same manager that created main_code_stream!
        // Using a decoder from a different nvImageCodec instance causes segfaults.
        auto& manager = NvImageCodecTiffParserManager::instance();
        if (!manager.is_available())
        {
            fmt::print("❌ nvImageCodec TIFF parser manager not initialized\n");
            return false;
        }
        
        // ROI decoding from TIFF requires nvTiff extension (not available in CPU-only backend)
        // Therefore, always use hybrid decoder for ROI operations
        // CPU-only decoder is only used for full IFD decoding (see decode_ifd_nvimgcodec)
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);
        
        nvimgcodecDecoder_t decoder = manager.get_decoder();  // Always use hybrid for ROI
        fmt::print("  💡 Using hybrid decoder for ROI (nvTiff required for TIFF sub-regions)\n");
        
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
        
        // Get sub-code stream for this ROI
        nvimgcodecCodeStream_t roi_stream;
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
            main_code_stream,
            &roi_stream,
            &view
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("❌ Failed to create ROI sub-stream (status: {})\n",
                      static_cast<int>(status));
            return false;
        }
        
        // Step 2: Determine buffer kind
        // For ROI decoding, hybrid decoder works best with GPU buffers (nvTiff uses GPU)
        // We'll decode to GPU and copy to CPU if needed
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);
        
        nvimgcodecImageBufferKind_t buffer_kind;
        if (gpu_available)
        {
            // Use GPU buffer for decoding (best performance with hybrid decoder)
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            if (target_is_cpu)
            {
                fmt::print("  ℹ️  Will decode to GPU then copy to CPU (ROI requires nvTiff)\n");
            }
        }
        else
        {
            // No GPU available, try CPU buffer (may not work for TIFF ROI)
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            fmt::print("  ⚠️  No GPU available, attempting CPU buffer (may fail for TIFF ROI)\n");
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
        output_image_info.buffer_size = buffer_size;
        output_image_info.cuda_stream = 0;
        
        fmt::print("  Buffer: {}x{} RGB, stride={}, size={} bytes\n",
                  width, height, row_stride, buffer_size);
        
        // Step 4: Allocate output buffer
        void* buffer = nullptr;
        if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaError_t cuda_status = cudaMalloc(&buffer, buffer_size);
            if (cuda_status != cudaSuccess)
            {
                fmt::print("❌ Failed to allocate GPU memory: {}\n", 
                          cudaGetErrorString(cuda_status));
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            fmt::print("  Allocated GPU buffer\n");
        }
        else
        {
            buffer = malloc(buffer_size);
            if (!buffer)
            {
                fmt::print("❌ Failed to allocate host memory\n");
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            fmt::print("  Allocated CPU buffer\n");
        }
        
        output_image_info.buffer = buffer;
        
        // Step 5: Create image object
        nvimgcodecImage_t image;
        status = nvimgcodecImageCreate(
            manager.get_instance(),
            &image,
            &output_image_info
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("❌ Failed to create image object (status: {})\n",
                      static_cast<int>(status));
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            nvimgcodecCodeStreamDestroy(roi_stream);
            return false;
        }
        
        // Step 6: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 7: Schedule decoding
        nvimgcodecFuture_t decode_future;
        status = nvimgcodecDecoderDecode(decoder,
                                        &roi_stream,
                                        &image,
                                        1,
                                        &decode_params,
                                        &decode_future);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("❌ Failed to schedule decoding (status: {})\n",
                      static_cast<int>(status));
            nvimgcodecImageDestroy(image);
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            nvimgcodecCodeStreamDestroy(roi_stream);
            return false;
        }
        
        // Step 8: Wait for completion
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        
        if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaDeviceSynchronize();
        }
        
        // Cleanup partial resources
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        
        // Step 9: Check decode status
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            fmt::print("❌ Decoding failed (status: {})\n", static_cast<int>(decode_status));
            
            // Decoding failed - clean up and return error
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            nvimgcodecCodeStreamDestroy(roi_stream);
            return false;
        }
        
        // Step 10: Handle GPU-to-CPU copy if target is CPU but we decoded to GPU
        if (target_is_cpu && buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            // Successful GPU decode, now copy to CPU
            fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
            fmt::print("  📥 Copying decoded data from GPU to CPU...\n");
            
            void* gpu_buffer = buffer;
            buffer = malloc(buffer_size);
            if (!buffer)
            {
                fmt::print("❌ Failed to allocate CPU memory for copy\n");
                cudaFree(gpu_buffer);
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            
            cudaError_t cuda_status = cudaMemcpy(buffer, gpu_buffer, buffer_size, cudaMemcpyDeviceToHost);
            cudaFree(gpu_buffer);  // Free GPU buffer after copy
            
            if (cuda_status != cudaSuccess)
            {
                fmt::print("❌ Failed to copy from GPU to CPU: {}\n", 
                          cudaGetErrorString(cuda_status));
                free(buffer);
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            fmt::print("  ✅ GPU-to-CPU copy completed\n");
        }
        else
        {
            fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
        }
        
        // Clean up
        nvimgcodecCodeStreamDestroy(roi_stream);
        
        // Assign output buffer
        *output_buffer = reinterpret_cast<uint8_t*>(buffer);
        fmt::print("✅ nvImageCodec ROI decode successful: {}x{} at ({}, {})\n", 
                  width, height, x, y);
        return true;
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
