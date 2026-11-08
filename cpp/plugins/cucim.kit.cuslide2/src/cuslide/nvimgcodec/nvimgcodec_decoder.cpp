/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvimgcodec_decoder.h"
#include "nvimgcodec_tiff_parser.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <memory>
#include <algorithm>
#include <array>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>
#include <mutex>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <cucim/memory/memory_manager.h>

#ifdef CUCIM_HAS_NVIMGCODEC
#include <cuda_runtime.h>
#endif

namespace cuslide2::nvimgcodec
{

DecodeBackend select_decode_backend(const std::vector<BackendProbe>& probes)
{
    for (const auto& probe : probes)
    {
        if (probe.decoder_available && probe.processing_status == 1)
        {
            return probe.backend;
        }
    }
    return DecodeBackend::kUnknown;
}

const char* decode_backend_name(DecodeBackend backend)
{
    switch (backend)
    {
    case DecodeBackend::kCpuOnly:
        return "cpu_only";
    case DecodeBackend::kGpuOnly:
        return "gpu_only";
    case DecodeBackend::kHybridCpuGpu:
        return "hybrid_cpu_gpu";
    case DecodeBackend::kHardwareGpuOnly:
        return "hardware_gpu_only";
    default:
        return "unknown";
    }
}

const char* decode_execution_name(DecodeBackend backend, bool success)
{
    if (!success)
    {
        return "decoder_failure";
    }
    switch (backend)
    {
    case DecodeBackend::kCpuOnly:
        return "cpu_fallback";
    case DecodeBackend::kGpuOnly:
    case DecodeBackend::kHardwareGpuOnly:
        return "gpu";
    case DecodeBackend::kHybridCpuGpu:
        return "hybrid_cpu_gpu";
    default:
        return "unknown";
    }
}

std::string processing_status_name(uint32_t status)
{
#ifdef CUCIM_HAS_NVIMGCODEC
    switch (status)
    {
    case NVIMGCODEC_PROCESSING_STATUS_UNKNOWN:
        return "unknown";
    case NVIMGCODEC_PROCESSING_STATUS_SUCCESS:
        return "success";
    case NVIMGCODEC_PROCESSING_STATUS_SATURATED:
        return "saturated";
    case NVIMGCODEC_PROCESSING_STATUS_FAIL:
        return "failure";
    case NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED:
        return "image_corrupted";
    case NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED:
        return "codec_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED:
        return "backend_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED:
        return "encoding_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED:
        return "resolution_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED:
        return "codestream_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED:
        return "color_spec_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED:
        return "orientation_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED:
        return "roi_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED:
        return "sampling_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED:
        return "sample_type_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED:
        return "sample_format_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED:
        return "num_images_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED:
        return "num_planes_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED:
        return "num_channels_unsupported";
    case NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED:
        return "tiling_unsupported";
    default:
        break;
    }
#endif
    return fmt::format("status_{}", status);
}

std::string decode_provenance_json(const std::vector<DecodeProvenance>& samples, size_t cache_hits, size_t cache_misses)
{
    using nlohmann::json;
    json sample_json = json::array();
    std::string execution =
        samples.empty() ? "unknown" : decode_execution_name(samples.front().backend, samples.front().success);
    bool mixed_execution = false;
    for (const auto& sample : samples)
    {
        if (execution != decode_execution_name(sample.backend, sample.success))
        {
            mixed_execution = true;
        }
        std::string cpu_work = "unknown";
        if (sample.backend == DecodeBackend::kCpuOnly || sample.backend == DecodeBackend::kHybridCpuGpu)
        {
            cpu_work = "present";
        }
        else if (sample.backend == DecodeBackend::kGpuOnly || sample.backend == DecodeBackend::kHardwareGpuOnly)
        {
            cpu_work = "not_in_decoder_backend";
        }

        sample_json.push_back({
            { "codec", sample.codec },
            { "plugin_id", sample.plugin_id.empty() ? json(nullptr) : json(sample.plugin_id) },
            { "plugin_id_status", sample.plugin_id.empty() ? "not_exposed_by_nvimagecodec_0_9" : "reported" },
            { "backend", decode_backend_name(sample.backend) },
            { "execution", decode_execution_name(sample.backend, sample.success) },
            { "execution_detail", sample.execution_detail.empty() ? json(nullptr) : json(sample.execution_detail) },
            { "success", sample.success },
            { "processing_status", processing_status_name(sample.processing_status) },
            { "processing_status_value", sample.processing_status },
            { "api_status", sample.api_status },
            { "failure_reason", sample.failure_reason.empty() ? json(nullptr) : json(sample.failure_reason) },
            { "boundaries",
              {
                  { "decoder_cpu_work", cpu_work },
                  { "output_memory", sample.output_memory },
                  { "output_cuda_stream", sample.cuda_stream },
                  { "future_wait_attempted", sample.future_wait_attempted },
                  { "future_wait_completed", sample.future_wait_completed },
                  { "cuda_stream_synchronize_attempted", sample.cuda_stream_synchronize_attempted },
                  { "cuda_stream_synchronize_completed", sample.cuda_stream_synchronize_completed },
              } },
        });
    }

    json report = {
        { "schema_version", 1 },
        { "selection_policy", "backend_restricted_gpu_then_hybrid_then_cpu" },
        { "execution", mixed_execution ? "unknown" : execution },
        { "execution_detail",
          mixed_execution ? json("mixed_sample_backends") :
                            (samples.empty() && cache_hits > 0 ? json("decoder_not_run_cache_hit") : json(nullptr)) },
        { "samples", std::move(sample_json) },
        { "cache", { { "hits", cache_hits }, { "misses", cache_misses } } },
        { "boundaries",
          {
              { "filesystem_io",
                {
                    { "owner", "nvImageCodec" },
                    { "api", "nvimgcodecCodeStreamCreateFromFile" },
                } },
              { "compressed_tile_host_staging",
                {
                    { "state", "unknown" },
                    { "reason", "nvImageCodec 0.9 does not expose compressed-input staging provenance" },
                } },
          } },
    };
    return json{ { "decoder_provenance", std::move(report) } }.dump();
}

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
    std::vector<DecodeProvenance> provenance; // One record per requested region
    size_t batch_size = 0;
    bool use_device_memory = false;
    bool owns_buffers = true;                         // false when caller provides output buffers
    cudaStream_t cuda_stream = nullptr;

    ~BatchDecodeStateImpl()
    {
        if (future)
        {
            nvimgcodecFutureDestroy(future);
            future = nullptr;
        }
        if (owns_buffers)
        {
            for (void* buf : buffers)
            {
                if (buf)
                {
                    if (use_device_memory)
                        cudaFree(buf);
                    else
                        cucim_free(buf);
                }
            }
        }
        buffers.clear();
    }
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

bool BatchDecodeState::is_valid() const
{
    return impl && impl->future;
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

struct DecodePixelSpec
{
    uint32_t num_channels = 3;
    uint32_t bytes_per_sample = 1;
    nvimgcodecSampleDataType_t sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    nvimgcodecColorSpec_t color_spec = NVIMGCODEC_COLORSPEC_SRGB;
};

DecodePixelSpec resolve_decode_pixel_spec(const IfdInfo& ifd_info)
{
    DecodePixelSpec spec;
    spec.num_channels = ifd_info.num_channels > 0 ? ifd_info.num_channels : 3;

    const uint32_t bits = ifd_info.bits_per_sample > 0 ? ifd_info.bits_per_sample : 8;
    if (bits > 8)
    {
        spec.bytes_per_sample = 2;
        spec.sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
    }
    else
    {
        spec.bytes_per_sample = 1;
        spec.sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    }

    int64_t photometric = -1;
    auto tag_it = ifd_info.tiff_tags.find("PHOTOMETRIC");
    if (tag_it != ifd_info.tiff_tags.end())
    {
        std::visit(
            [&](const auto& v)
            {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_integral_v<T>)
                {
                    photometric = static_cast<int64_t>(v);
                }
            },
            tag_it->second);
    }

    if (spec.num_channels == 1 || photometric == 0 || photometric == 1)
    {
        spec.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
    }
    else
    {
        spec.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    }

    return spec;
}

struct DecoderChoice
{
    nvimgcodecDecoder_t decoder = nullptr;
    DecodeBackend backend = DecodeBackend::kUnknown;
    std::vector<nvimgcodecProcessingStatus_t> statuses;
    nvimgcodecStatus_t api_status = NVIMGCODEC_STATUS_SUCCESS;
};

static DecodeBackend decode_backend_from_nvimgcodec(nvimgcodecBackendKind_t backend)
{
    switch (backend)
    {
    case NVIMGCODEC_BACKEND_KIND_CPU_ONLY:
        return DecodeBackend::kCpuOnly;
    case NVIMGCODEC_BACKEND_KIND_GPU_ONLY:
        return DecodeBackend::kGpuOnly;
    case NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU:
        return DecodeBackend::kHybridCpuGpu;
    case NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY:
        return DecodeBackend::kHardwareGpuOnly;
    default:
        return DecodeBackend::kUnknown;
    }
}

static DecoderChoice choose_decoder(NvImageCodecTiffParserManager& manager,
                                    const nvimgcodecCodeStream_t* streams,
                                    const nvimgcodecImage_t* images,
                                    size_t sample_count,
                                    const nvimgcodecDecodeParams_t& decode_params)
{
    DecoderChoice choice;
    choice.statuses.assign(sample_count, NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);

    // Keep the policy explicit and stable. Dedicated hardware and GPU-only
    // decoders are preferred, followed by hybrid work and finally CPU fallback.
    const std::array<nvimgcodecBackendKind_t, 4> candidates = {
        NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
        NVIMGCODEC_BACKEND_KIND_GPU_ONLY,
        NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
        NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
    };

    bool any_probe_completed = false;
    std::vector<bool> supported_by_any_backend(sample_count, false);
    for (const auto backend : candidates)
    {
        nvimgcodecDecoder_t decoder = manager.get_backend_decoder(backend);
        if (!decoder)
        {
            continue;
        }

        std::vector<nvimgcodecProcessingStatus_t> statuses(sample_count, NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        const nvimgcodecStatus_t api_status = nvimgcodecDecoderCanDecode(
            decoder, streams, images, static_cast<int>(sample_count), &decode_params, statuses.data(), 1);
        if (api_status != NVIMGCODEC_STATUS_SUCCESS)
        {
            choice.api_status = api_status;
            continue;
        }

        any_probe_completed = true;
        bool all_supported = true;
        for (size_t i = 0; i < sample_count; ++i)
        {
            const bool supported = statuses[i] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
            supported_by_any_backend[i] = supported_by_any_backend[i] || supported;
            all_supported = all_supported && supported;
            if (choice.statuses[i] == NVIMGCODEC_PROCESSING_STATUS_UNKNOWN)
            {
                choice.statuses[i] = statuses[i];
            }
        }
        if (all_supported)
        {
            choice.decoder = decoder;
            choice.backend = decode_backend_from_nvimgcodec(backend);
            choice.statuses = std::move(statuses);
            choice.api_status = NVIMGCODEC_STATUS_SUCCESS;
            return choice;
        }
    }

    // A heterogeneous batch may need multiple backend kinds. Preserve the
    // existing generic-dispatch behavior in that case, but mark provenance as
    // unknown because nvImageCodec 0.9 cannot disclose its per-sample choice.
    const bool all_supported_somewhere = std::all_of(
        supported_by_any_backend.begin(), supported_by_any_backend.end(), [](bool supported) { return supported; });
    if (!any_probe_completed || all_supported_somewhere)
    {
        choice.decoder = manager.get_decoder();
        choice.backend = DecodeBackend::kUnknown;
    }
    return choice;
}

static DecodeProvenance make_provenance(const IfdInfo& ifd_info,
                                        DecodeBackend backend,
                                        bool device_output,
                                        bool dedicated_stream)
{
    DecodeProvenance provenance;
    provenance.codec = ifd_info.compression_codec.empty() ? ifd_info.codec : ifd_info.compression_codec;
    provenance.backend = backend;
    provenance.processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    provenance.api_status = NVIMGCODEC_STATUS_SUCCESS;
    provenance.output_memory = device_output ? "cuda" : "host";
    provenance.cuda_stream = device_output ? (dedicated_stream ? "dedicated" : "default") : "none";
    return provenance;
}

// ============================================================================
// IFD-Level Region Decoding (Primary Decode Function)
// ============================================================================

bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x,
                                  uint32_t y,
                                  uint32_t width,
                                  uint32_t height,
                                  uint8_t*& output_buffer,
                                  const cucim::io::Device& out_device,
                                  cudaStream_t cuda_stream,
                                  DecodeProvenance* provenance)
{
    const bool dedicated_stream = cuda_stream != nullptr;
    const bool requested_device_output = std::string(out_device).find("cuda") != std::string::npos;
    if (provenance)
    {
        *provenance = make_provenance(ifd_info, DecodeBackend::kUnknown, requested_device_output, dedicated_stream);
    }
    auto fail = [&](std::string reason, int api_status = NVIMGCODEC_STATUS_SUCCESS,
                    uint32_t processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN) {
        if (provenance)
        {
            provenance->success = false;
            provenance->failure_reason = std::move(reason);
            provenance->api_status = api_status;
            provenance->processing_status = processing_status;
        }
        return false;
    };

    // Normalize: treat nullptr as the default CUDA stream
    if (!cuda_stream) cuda_stream = cudaStream_t(0);

    if (!main_code_stream)
    {
        #ifdef DEBUG
        fmt::print("❌ Invalid main_code_stream\n");
        #endif
        return fail("invalid_main_codestream");
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
            return fail("nvimagecodec_manager_unavailable");
        }

        // Caller-provided output buffer (optional). If non-null, we decode into it.
        const bool caller_provided_buffer = (output_buffer != nullptr);

        // Determine target device for buffer placement
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);

        #ifdef DEBUG
        fmt::print("  💡 Decoding with default decoder (output buffer: {})\n", target_is_cpu ? "host" : "device");
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
            return fail("roi_substream_creation_failed", static_cast<int>(status));
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

        const DecodePixelSpec decode_spec = resolve_decode_pixel_spec(ifd_info);
        // Use UNCHANGED to preserve original planar packing and channel ordering.
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
        output_image_info.color_spec = decode_spec.color_spec;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;
        output_image_info.buffer_kind = buffer_kind;

        uint32_t num_channels = decode_spec.num_channels;
        size_t row_stride = static_cast<size_t>(width) * num_channels * decode_spec.bytes_per_sample;
        size_t buffer_size = row_stride * height;

        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = decode_spec.sample_type;
        // Buffer size is inferred from plane_info rather than set explicitly.
        output_image_info.cuda_stream = cuda_stream;

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
                return fail("output_allocation_failed");
            }
            output_image_info.buffer = decode_buffer.get();
        }
        if (provenance)
        {
            provenance->output_memory = use_device_memory ? "cuda" : "host";
            provenance->cuda_stream = use_device_memory ? (dedicated_stream ? "dedicated" : "default") : "none";
        }

        // Step 5: Create image
        nvimgcodecImage_t image_raw = nullptr;
        status = nvimgcodecImageCreate(manager.get_instance(), &image_raw, &output_image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create image object (status: {})\n", static_cast<int>(status));
            #endif
            return fail("output_image_creation_failed", static_cast<int>(status));
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
        DecoderChoice decoder_choice = choose_decoder(manager, &roi_stream_ptr, &image_ptr, 1, decode_params);
        if (provenance)
        {
            provenance->backend = decoder_choice.backend;
            if (decoder_choice.decoder && decoder_choice.backend == DecodeBackend::kUnknown)
            {
                provenance->execution_detail = "generic_dispatch_backend_not_exposed";
            }
        }
        if (!decoder_choice.decoder)
        {
            const uint32_t processing_status = decoder_choice.statuses.empty() ?
                                                   static_cast<uint32_t>(NVIMGCODEC_PROCESSING_STATUS_UNKNOWN) :
                                                   decoder_choice.statuses.front();
            return fail("no_available_decoder_backend", static_cast<int>(decoder_choice.api_status), processing_status);
        }

        nvimgcodecFuture_t decode_future_raw = nullptr;
        status = nvimgcodecDecoderDecode(
            decoder_choice.decoder, &roi_stream_ptr, &image_ptr, 1, &decode_params, &decode_future_raw);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule decoding (status: {})\n", static_cast<int>(status));
            #endif
            return fail("decode_scheduling_failed", static_cast<int>(status));
        }
        UniqueFuture decode_future(decode_future_raw);

        if (provenance)
        {
            provenance->future_wait_attempted = true;
        }
        status = nvimgcodecFutureWaitForAll(decode_future.get());
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            return fail("decode_wait_failed", static_cast<int>(status));
        }
        if (provenance)
        {
            provenance->future_wait_completed = true;
        }

        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        status = nvimgcodecFutureGetProcessingStatus(decode_future.get(), &decode_status, &status_size);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to get processing status (status: {})\n", static_cast<int>(status));
            #endif
            return fail("processing_status_query_failed", static_cast<int>(status));
        }

        if (use_device_memory)
        {
            if (provenance)
            {
                provenance->cuda_stream_synchronize_attempted = true;
            }
            const cudaError_t sync_status = cudaStreamSynchronize(cuda_stream);
            if (sync_status != cudaSuccess)
            {
                return fail("cuda_stream_synchronize_failed", static_cast<int>(sync_status), decode_status);
            }
            if (provenance)
            {
                provenance->cuda_stream_synchronize_completed = true;
            }
        }

        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Decoding failed (status: {})\n", static_cast<int>(decode_status));
            #endif
            return fail("decoder_processing_failed", NVIMGCODEC_STATUS_SUCCESS, decode_status);
        }

        // Success: return buffer (only if we allocated it)
        if (!caller_provided_buffer)
        {
            output_buffer = reinterpret_cast<uint8_t*>(decode_buffer.release());
        }
        if (provenance)
        {
            provenance->success = true;
            provenance->processing_status = decode_status;
            provenance->api_status = NVIMGCODEC_STATUS_SUCCESS;
        }
        return true;
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in ROI decoding: {}\n", e.what());
        #endif
        return fail(fmt::format("exception: {}", e.what()));
    }
}

// ============================================================================
// Batch ROI Decoding
// ============================================================================

std::vector<BatchDecodeResult> decode_batch_regions_nvimgcodec(
    const IfdInfo& ifd_info,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device,
    cudaStream_t cuda_stream)
{
    // Normalize: treat nullptr as the default CUDA stream
    if (!cuda_stream) cuda_stream = cudaStream_t(0);

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

        for (auto& result : results)
        {
            result.provenance =
                make_provenance(ifd_info, DecodeBackend::kUnknown, use_device_memory, cuda_stream != cudaStream_t(0));
        }

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
            const DecodePixelSpec decode_spec = resolve_decode_pixel_spec(ifd_info);
            uint32_t num_channels = decode_spec.num_channels;
            size_t row_stride = static_cast<size_t>(region.width) * num_channels * decode_spec.bytes_per_sample;
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
            // Use UNCHANGED to preserve original planar packing and channel ordering.
            output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
            output_image_info.color_spec = decode_spec.color_spec;
            output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
            output_image_info.num_planes = 1;
            output_image_info.buffer_kind = buffer_kind;
            output_image_info.buffer = decode_buffers[i].get();
            output_image_info.plane_info[0].height = region.height;
            output_image_info.plane_info[0].width = region.width;
            output_image_info.plane_info[0].num_channels = num_channels;
            output_image_info.plane_info[0].row_stride = row_stride;
            output_image_info.plane_info[0].sample_type = decode_spec.sample_type;
            output_image_info.cuda_stream = cuda_stream;

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

        DecoderChoice decoder_choice =
            choose_decoder(manager, valid_streams.data(), valid_images.data(), valid_streams.size(), decode_params);
        for (size_t vi = 0; vi < valid_indices.size(); ++vi)
        {
            auto& provenance = results[valid_indices[vi]].provenance;
            provenance.backend = decoder_choice.backend;
            if (decoder_choice.decoder && decoder_choice.backend == DecodeBackend::kUnknown)
            {
                provenance.execution_detail = "generic_dispatch_backend_not_exposed";
            }
            provenance.api_status = decoder_choice.api_status;
            if (vi < decoder_choice.statuses.size())
                provenance.processing_status = decoder_choice.statuses[vi];
        }
        if (!decoder_choice.decoder)
        {
            for (const size_t index : valid_indices)
            {
                results[index].provenance.failure_reason = "no_available_decoder_backend";
            }
            return results;
        }

        nvimgcodecFuture_t decode_future_raw = nullptr;
        nvimgcodecStatus_t status =
            nvimgcodecDecoderDecode(decoder_choice.decoder, valid_streams.data(), valid_images.data(),
                                    static_cast<int>(valid_streams.size()), &decode_params, &decode_future_raw);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule batch decoding (status: {})\n", static_cast<int>(status));
            #endif
            return results;
        }
        UniqueFuture decode_future(decode_future_raw);

        for (const size_t index : valid_indices)
        {
            results[index].provenance.future_wait_attempted = true;
        }
        status = nvimgcodecFutureWaitForAll(decode_future.get());
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            for (const size_t index : valid_indices)
            {
                results[index].provenance.api_status = status;
                results[index].provenance.failure_reason = "decode_wait_failed";
            }
            return results;
        }
        for (const size_t index : valid_indices)
        {
            results[index].provenance.future_wait_completed = true;
        }

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
            for (const size_t index : valid_indices)
            {
                results[index].provenance.cuda_stream_synchronize_attempted = true;
            }
            const cudaError_t sync_status = cudaStreamSynchronize(cuda_stream);
            if (sync_status != cudaSuccess)
            {
                for (const size_t index : valid_indices)
                {
                    results[index].provenance.api_status = sync_status;
                    results[index].provenance.failure_reason = "cuda_stream_synchronize_failed";
                }
                return results;
            }
            for (const size_t index : valid_indices)
            {
                results[index].provenance.cuda_stream_synchronize_completed = true;
            }
        }

        // Step 6: Transfer successful results to output
        for (size_t vi = 0; vi < valid_indices.size(); ++vi)
        {
            size_t i = valid_indices[vi];
            results[i].provenance.processing_status = decode_statuses[vi];
            if (decode_statuses[vi] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
            {
                results[i].buffer = reinterpret_cast<uint8_t*>(decode_buffers[i].release());
                results[i].success = true;
                results[i].provenance.success = true;
#    ifdef DEBUG
                fmt::print("  ✅ Region {} decoded successfully\n", i);
#endif
            }
            else
            {
                results[i].provenance.failure_reason = "decoder_processing_failed";
#    ifdef DEBUG
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
    cudaStream_t cuda_stream,
    const std::vector<uint8_t*>& output_buffers)
{
    // Normalize: treat nullptr as the default CUDA stream
    if (!cuda_stream) cuda_stream = cudaStream_t(0);

    BatchDecodeState state;
    auto* impl = state.impl;
    impl->batch_size = regions.size();
    impl->use_device_memory = false;
    impl->cuda_stream = cuda_stream;
    impl->provenance.resize(impl->batch_size);

    // When caller-provided output buffers are supplied, the decoder writes
    // directly into them and the state does NOT own (or free) those buffers.
    bool use_caller_buffers = !output_buffers.empty();
    impl->owns_buffers = !use_caller_buffers;

    if (impl->batch_size == 0 || !main_code_stream)
    {
        return state;
    }

    // No try-catch: nvImageCodec is a C API and does not throw.
    // The only possible exception is std::bad_alloc from vector allocations,
    // which should propagate to the caller.  ~BatchDecodeStateImpl provides
    // RAII cleanup regardless.
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

        for (auto& sample_provenance : impl->provenance)
        {
            sample_provenance = make_provenance(
                ifd_info, DecodeBackend::kUnknown, impl->use_device_memory, cuda_stream != cudaStream_t(0));
        }

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

        // Step 2: Set up output buffers and create image objects.
        // Two modes:
        //   (a) use_caller_buffers == true  → decode directly into caller-provided
        //       raster pointers. The state does NOT own these buffers. This says
        //       nothing about compressed-input staging inside nvImageCodec.
        //   (b) use_caller_buffers == false → allocate DecodeBuffers internally
        //       (ownership tracked in impl->buffers, freed by ~BatchDecodeStateImpl).
        std::vector<DecodeBuffer> decode_buffers;
        if (!use_caller_buffers)
        {
            decode_buffers.resize(impl->batch_size);
        }
        impl->images_raii.reserve(impl->batch_size);

        // Per-region buffer sizes, indexed by original batch index.
        // Populated in step 2, consumed in step 3 to keep impl->buffer_sizes
        // aligned with impl->valid_indices / impl->buffers.
        std::vector<size_t> per_region_buffer_sizes(impl->batch_size, 0);

        for (size_t i = 0; i < impl->batch_size; ++i)
        {
            if (!impl->roi_streams_raii[i])
            {
                impl->images_raii.emplace_back(nullptr);
                continue;
            }

            const auto& region = regions[i];
            const DecodePixelSpec decode_spec = resolve_decode_pixel_spec(ifd_info);
            uint32_t num_channels = decode_spec.num_channels;
            size_t row_stride = static_cast<size_t>(region.width) * num_channels * decode_spec.bytes_per_sample;
            size_t buffer_size = row_stride * region.height;

            void* buffer_ptr = nullptr;
            if (use_caller_buffers)
            {
                // Use the caller's pre-allocated raster buffer
                if (i < output_buffers.size() && output_buffers[i])
                {
                    buffer_ptr = output_buffers[i];
                }
                else
                {
                    impl->images_raii.emplace_back(nullptr);
                    continue;
                }
            }
            else
            {
                // Allocate a fresh decode buffer
                if (!decode_buffers[i].allocate(buffer_size, impl->use_device_memory))
                {
                    impl->images_raii.emplace_back(nullptr);
                    continue;
                }
                buffer_ptr = decode_buffers[i].get();
            }

            nvimgcodecImageInfo_t output_image_info{};
            output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
            output_image_info.struct_next = nullptr;
            output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
            output_image_info.color_spec = decode_spec.color_spec;
            output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
            output_image_info.num_planes = 1;
            output_image_info.buffer_kind = buffer_kind;
            output_image_info.buffer = buffer_ptr;
            output_image_info.plane_info[0].height = region.height;
            output_image_info.plane_info[0].width = region.width;
            output_image_info.plane_info[0].num_channels = num_channels;
            output_image_info.plane_info[0].row_stride = row_stride;
            output_image_info.plane_info[0].sample_type = decode_spec.sample_type;
            output_image_info.cuda_stream = impl->cuda_stream;

            nvimgcodecImage_t image_raw = nullptr;
            nvimgcodecStatus_t status = nvimgcodecImageCreate(manager.get_instance(), &image_raw, &output_image_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS)
            {
                impl->images_raii.emplace_back(nullptr);
                continue;
            }

            impl->images_raii.emplace_back(image_raw);
            per_region_buffer_sizes[i] = buffer_size;
        }

        // Step 3: Filter out invalid entries and prepare for batch decode.
        // All impl-> vectors (roi_streams, images, buffers, buffer_sizes,
        // valid_indices) are populated here so their indices stay aligned.
        for (size_t i = 0; i < impl->batch_size; ++i)
        {
            if (impl->roi_streams_raii[i] && impl->images_raii[i])
            {
                impl->roi_streams.push_back(impl->roi_streams_raii[i].get());
                impl->images.push_back(impl->images_raii[i].get());
                impl->valid_indices.push_back(i);
                impl->buffer_sizes.push_back(per_region_buffer_sizes[i]);
                if (impl->owns_buffers)
                {
                    impl->buffers.push_back(decode_buffers[i].release()); // Transfer ownership
                }
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

        DecoderChoice decoder_choice = choose_decoder(
            manager, impl->roi_streams.data(), impl->images.data(), impl->roi_streams.size(), decode_params);
        for (size_t vi = 0; vi < impl->valid_indices.size(); ++vi)
        {
            auto& sample_provenance = impl->provenance[impl->valid_indices[vi]];
            sample_provenance.backend = decoder_choice.backend;
            if (decoder_choice.decoder && decoder_choice.backend == DecodeBackend::kUnknown)
            {
                sample_provenance.execution_detail = "generic_dispatch_backend_not_exposed";
            }
            sample_provenance.api_status = decoder_choice.api_status;
            if (vi < decoder_choice.statuses.size())
            {
                sample_provenance.processing_status = decoder_choice.statuses[vi];
            }
        }
        if (!decoder_choice.decoder)
        {
            for (const size_t index : impl->valid_indices)
            {
                impl->provenance[index].failure_reason = "no_available_decoder_backend";
            }
            return state;
        }

        nvimgcodecStatus_t status =
            nvimgcodecDecoderDecode(decoder_choice.decoder, impl->roi_streams.data(), impl->images.data(),
                                    static_cast<int>(impl->roi_streams.size()), &decode_params, &impl->future);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            for (const size_t index : impl->valid_indices)
            {
                impl->provenance[index].api_status = status;
                impl->provenance[index].failure_reason = "decode_scheduling_failed";
            }
#    ifdef DEBUG
            fmt::print("❌ Failed to schedule batch decoding (status: {})\n", static_cast<int>(status));
#endif
            // No manual cleanup needed — ~BatchDecodeStateImpl handles buffer
            // and future cleanup when `state` is destroyed by the caller.
            return state;
        }

#ifdef DEBUG
        fmt::print("✅ Scheduled batch decode of {} regions\n", impl->roi_streams.size());
#endif
    }

    return state;
}

std::vector<BatchDecodeResult> wait_batch_decode(BatchDecodeState& state)
{
    if (!state.impl) return {};
    auto* impl = state.impl;
    std::vector<BatchDecodeResult> results(impl->batch_size);

    // Initialize all results to failure
    for (auto& result : results)
    {
        result.buffer = nullptr;
        result.buffer_size = 0;
        result.success = false;
    }
    for (size_t i = 0; i < results.size() && i < impl->provenance.size(); ++i)
    {
        results[i].provenance = impl->provenance[i];
    }

    if (!impl->future || impl->roi_streams.empty())
    {
        return results;
    }

    // Wait for all decode operations to complete
    for (const size_t index : impl->valid_indices)
    {
        results[index].provenance.future_wait_attempted = true;
    }
    nvimgcodecStatus_t status = nvimgcodecFutureWaitForAll(impl->future);
    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        for (auto& result : results)
        {
            result.provenance.api_status = status;
            result.provenance.failure_reason = "decode_wait_failed";
        }
#    ifdef DEBUG
        fmt::print("❌ Failed to wait for batch decode (status: {})\n", static_cast<int>(status));
#endif
        // No manual cleanup — ~BatchDecodeStateImpl handles it.
        return results;
    }
    for (const size_t index : impl->valid_indices)
    {
        results[index].provenance.future_wait_completed = true;
    }

    // Synchronize stream if using GPU
    if (impl->use_device_memory)
    {
        for (const size_t index : impl->valid_indices)
        {
            results[index].provenance.cuda_stream_synchronize_attempted = true;
        }
        const cudaError_t sync_status = cudaStreamSynchronize(impl->cuda_stream);
        for (const size_t index : impl->valid_indices)
        {
            auto& provenance = results[index].provenance;
            provenance.cuda_stream_synchronize_completed = sync_status == cudaSuccess;
            if (sync_status != cudaSuccess)
            {
                provenance.api_status = sync_status;
                provenance.failure_reason = "cuda_stream_synchronize_failed";
            }
        }
        if (sync_status != cudaSuccess)
        {
            return results;
        }
    }

    // Get processing status for each image
    std::vector<nvimgcodecProcessingStatus_t> decode_statuses(impl->roi_streams.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
    size_t status_size = impl->roi_streams.size();
    status = nvimgcodecFutureGetProcessingStatus(impl->future, decode_statuses.data(), &status_size);

    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        for (auto& result : results)
        {
            result.provenance.api_status = status;
            result.provenance.failure_reason = "processing_status_query_failed";
        }
#    ifdef DEBUG
        fmt::print("❌ Failed to get batch processing status (status: {})\n", static_cast<int>(status));
#endif
        return results;  // All results already initialized to failure
    }

    // Process results
    for (size_t vi = 0; vi < impl->valid_indices.size(); ++vi)
    {
        size_t i = impl->valid_indices[vi];
        results[i].provenance.processing_status = vi < decode_statuses.size() ?
                                                      decode_statuses[vi] :
                                                      static_cast<uint32_t>(NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        if (vi < decode_statuses.size() && decode_statuses[vi] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            if (impl->owns_buffers)
            {
                // Internally-allocated buffers: transfer ownership to result
                if (vi < impl->buffers.size() && impl->buffers[vi])
                {
                    results[i].buffer = reinterpret_cast<uint8_t*>(impl->buffers[vi]);
                    results[i].buffer_size = impl->buffer_sizes[vi];
                    results[i].success = true;
                    results[i].provenance.success = true;
                    impl->buffers[vi] = nullptr;
                }
            }
            else
            {
                // Caller-provided buffers: data is already in place
                results[i].buffer = nullptr;  // Caller owns the buffer
                results[i].buffer_size = impl->buffer_sizes[vi];
                results[i].success = true;
                results[i].provenance.success = true;
            }
        }
        else
        {
            results[i].provenance.failure_reason = "decoder_processing_failed";
        }
    }

    #ifdef DEBUG
    size_t success_count = 0;
    for (const auto& r : results) if (r.success) ++success_count;
    fmt::print("✅ Batch decode complete: {}/{} regions successful\n", success_count, impl->batch_size);
    #endif

    return results;
}

#else // !CUCIM_HAS_NVIMGCODEC

// Fallback stub when nvImageCodec is not available
// cuslide2 plugin requires nvImageCodec, so this should never be called
bool decode_ifd_region_nvimgcodec(const IfdInfo&,
                                  nvimgcodecCodeStream_t,
                                  uint32_t,
                                  uint32_t,
                                  uint32_t,
                                  uint32_t,
                                  uint8_t*&,
                                  const cucim::io::Device&,
                                  cudaStream_t,
                                  DecodeProvenance*)
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
