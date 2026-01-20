# Batch Decoding Implementation - Line-by-Line Code Description

This document provides a detailed line-by-line explanation of the code changes for implementing batch ROI decoding using nvImageCodec v0.7.0+ API.

---

## Table of Contents

1. [nvimgcodec_decoder.h - Header Additions](#1-nvimgcodec_decoderh---header-additions)
2. [nvimgcodec_decoder.cpp - Batch Decode Implementation](#2-nvimgcodec_decodercpp---batch-decode-implementation)
3. [nvimgcodec_processor.h - Processor Class Definition](#3-nvimgcodec_processorh---processor-class-definition)
4. [nvimgcodec_processor.cpp - Processor Implementation](#4-nvimgcodec_processorcpp---processor-implementation)
5. [ifd.cpp - Integration with Batch Loading](#5-ifdcpp---integration-with-batch-loading)
6. [CMakeLists.txt - Build Configuration](#6-cmakeliststxt---build-configuration)
7. [test_batch_decoding.py - Python Tests](#7-test_batch_decodingpy---python-tests)

---

## 1. nvimgcodec_decoder.h - Header Additions

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.h`

### Lines 118-125: RoiRegion Struct

```cpp
struct RoiRegion
{
    uint32_t ifd_index;  // IFD index (resolution level)
    uint32_t x;          // Starting x coordinate (column)
    uint32_t y;          // Starting y coordinate (row)
    uint32_t width;      // Width of region in pixels
    uint32_t height;     // Height of region in pixels
};
```

**Purpose:** Defines a Region of Interest (ROI) specification for batch decoding.

| Field | Description |
|-------|-------------|
| `ifd_index` | Which IFD (resolution pyramid level) to decode from |
| `x`, `y` | Top-left corner coordinates of the ROI in pixels |
| `width`, `height` | Size of the ROI to decode |

### Lines 130-135: BatchDecodeResult Struct

```cpp
struct BatchDecodeResult
{
    uint8_t* buffer;     // Decoded pixel data (caller must free)
    size_t buffer_size;  // Size of buffer in bytes
    bool success;        // Whether decoding succeeded
};
```

**Purpose:** Holds the result of decoding a single ROI in a batch operation.

| Field | Description |
|-------|-------------|
| `buffer` | Pointer to decoded pixel data (RGB, 8-bit per channel). Caller owns this memory. |
| `buffer_size` | Total size in bytes: `width × height × 3` |
| `success` | `true` if decode succeeded, `false` otherwise |

### Lines 157-161: decode_batch_regions_nvimgcodec Function Declaration

```cpp
std::vector<BatchDecodeResult> decode_batch_regions_nvimgcodec(
    const std::vector<const IfdInfo*>& ifd_infos,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device);
```

**Purpose:** Main batch decoding function that decodes multiple ROIs in a single `decoder.decode()` call.

| Parameter | Description |
|-----------|-------------|
| `ifd_infos` | Vector of IFD metadata (one per unique IFD used) |
| `main_code_stream` | The main TIFF CodeStream from TiffFileParser |
| `regions` | Vector of ROI specifications to decode |
| `out_device` | Target device ("cpu" or "cuda") |
| **Returns** | Vector of results, same order as input regions |

---

## 2. nvimgcodec_decoder.cpp - Batch Decode Implementation

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp`

### Lines 419-424: Function Signature and Initialization

```cpp
std::vector<BatchDecodeResult> decode_batch_regions_nvimgcodec(
    const std::vector<const IfdInfo*>& ifd_infos,
    nvimgcodecCodeStream_t main_code_stream,
    const std::vector<RoiRegion>& regions,
    const cucim::io::Device& out_device)
{
    const size_t batch_size = regions.size();
    std::vector<BatchDecodeResult> results(batch_size);
```

**Explanation:**
- Creates a results vector with one entry per input region
- `batch_size` determines how many ROIs we'll decode in one call

### Lines 428-434: Initialize Results to Failure

```cpp
    // Initialize all results to failure
    for (auto& result : results)
    {
        result.buffer = nullptr;
        result.buffer_size = 0;
        result.success = false;
    }
```

**Explanation:**
- Safe initialization - all results start as failed
- Only successful decodes will be marked `success = true`

### Lines 453-503: Device Selection Logic

```cpp
    auto& manager = NvImageCodecTiffParserManager::instance();
    
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
    }

    // Select decoder
    nvimgcodecDecoder_t decoder;
    if (target_is_cpu && manager.has_cpu_decoder())
    {
        decoder = manager.get_cpu_decoder();
    }
    else
    {
        decoder = manager.get_decoder();
    }
```

**Explanation:**
- Gets the singleton nvImageCodec manager instance
- Determines if output should go to CPU or GPU memory
- Selects appropriate decoder (CPU-only or hybrid)
- Falls back to CPU if GPU not available

### Lines 505-566: Step 1 - Create ROI Sub-Streams

```cpp
    // Step 1: Create ROI sub-streams for each region
    // Per nvImageCodec team guidance: call get_sub_code_stream() for each ROI
    std::vector<UniqueCodeStream> roi_streams;
    std::vector<nvimgcodecCodeStream_t> roi_stream_ptrs;
    roi_streams.reserve(batch_size);
    roi_stream_ptrs.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i)
    {
        const auto& region = regions[i];
        const IfdInfo* ifd_info = (region.ifd_index < ifd_infos.size()) 
                                   ? ifd_infos[region.ifd_index] : nullptr;

        if (!ifd_info)
        {
            roi_streams.emplace_back(nullptr);
            roi_stream_ptrs.push_back(nullptr);
            continue;
        }

        // Create nvImageCodec region specification
        nvimgcodecRegion_t nvregion{};
        nvregion.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
        nvregion.struct_size = sizeof(nvimgcodecRegion_t);
        nvregion.struct_next = nullptr;
        nvregion.ndim = 2;
        nvregion.start[0] = region.y;  // row (Y comes first in nvImageCodec)
        nvregion.start[1] = region.x;  // col
        nvregion.end[0] = region.y + region.height;
        nvregion.end[1] = region.x + region.width;

        // Create CodeStreamView with ROI
        nvimgcodecCodeStreamView_t view{};
        view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
        view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
        view.struct_next = nullptr;
        view.image_idx = region.ifd_index;
        view.region = nvregion;

        // Get sub-code stream with this ROI
        nvimgcodecCodeStream_t roi_stream_raw = nullptr;
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
            main_code_stream, &roi_stream_raw, &view);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            roi_streams.emplace_back(nullptr);
            roi_stream_ptrs.push_back(nullptr);
            continue;
        }

        roi_streams.emplace_back(roi_stream_raw);  // RAII wrapper
        roi_stream_ptrs.push_back(roi_stream_raw);
    }
```

**Key Points:**
- **This is the core of batch decoding** - calls `nvimgcodecCodeStreamGetSubCodeStream()` for each ROI
- Creates a "view" into the main CodeStream with specific region bounds
- `nvregion.start[0]` = row (Y), `nvregion.start[1]` = column (X)
- Uses RAII wrapper (`UniqueCodeStream`) to ensure cleanup

### Lines 568-636: Step 2 - Allocate Buffers and Create Images

```cpp
    // Step 2: Allocate output buffers and create image objects
    std::vector<DecodeBuffer> decode_buffers(batch_size);
    std::vector<UniqueImage> images;
    std::vector<nvimgcodecImage_t> image_ptrs;

    for (size_t i = 0; i < batch_size; ++i)
    {
        if (!roi_stream_ptrs[i])
        {
            images.emplace_back(nullptr);
            image_ptrs.push_back(nullptr);
            continue;
        }

        const auto& region = regions[i];
        uint32_t num_channels = 3;  // RGB output
        size_t row_stride = region.width * num_channels;
        size_t buffer_size = row_stride * region.height;

        // Allocate buffer (CPU or GPU memory)
        if (!decode_buffers[i].allocate(buffer_size, use_device_memory))
        {
            images.emplace_back(nullptr);
            image_ptrs.push_back(nullptr);
            continue;
        }

        // Configure output image
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
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
        output_image_info.cuda_stream = 0;

        // Create nvImageCodec image object
        nvimgcodecImage_t image_raw = nullptr;
        nvimgcodecStatus_t status = nvimgcodecImageCreate(
            manager.get_instance(), &image_raw, &output_image_info);
        
        images.emplace_back(image_raw);
        image_ptrs.push_back(image_raw);
        results[i].buffer_size = buffer_size;
    }
```

**Explanation:**
- Allocates output buffers using `DecodeBuffer` RAII class
- CPU: uses `malloc()` (compatible with cuCIM's `free()`)
- GPU: uses `cudaMalloc()`
- Configures interleaved RGB output format (NVIMGCODEC_SAMPLEFORMAT_I_RGB)

### Lines 638-665: Step 3 - Filter Valid Entries

```cpp
    // Step 3: Filter out invalid entries for batch decode
    std::vector<nvimgcodecCodeStream_t> valid_streams;
    std::vector<nvimgcodecImage_t> valid_images;
    std::vector<size_t> valid_indices;

    for (size_t i = 0; i < batch_size; ++i)
    {
        if (roi_stream_ptrs[i] && image_ptrs[i])
        {
            valid_streams.push_back(roi_stream_ptrs[i]);
            valid_images.push_back(image_ptrs[i]);
            valid_indices.push_back(i);
        }
    }

    if (valid_streams.empty())
    {
        return results;
    }
```

**Explanation:**
- Only valid entries (successful sub-stream + image creation) are sent to decoder
- Tracks original indices to map results back correctly

### Lines 666-689: Step 4 - Single Batch Decode Call (THE KEY OPTIMIZATION)

```cpp
    // Step 4: Single batch decode call (the key optimization!)
    nvimgcodecDecodeParams_t decode_params{};
    decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
    decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
    decode_params.struct_next = nullptr;
    decode_params.apply_exif_orientation = 1;

    nvimgcodecFuture_t decode_future_raw = nullptr;
    nvimgcodecStatus_t status = nvimgcodecDecoderDecode(
        decoder,
        valid_streams.data(),      // Array of code streams
        valid_images.data(),       // Array of output images  
        static_cast<int>(valid_streams.size()),  // Batch size
        &decode_params,
        &decode_future_raw);

    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        return results;
    }
    UniqueFuture decode_future(decode_future_raw);
```

**This is the KEY line - `nvimgcodecDecoderDecode()` with arrays:**
- Instead of N separate decode calls, we make ONE call
- nvImageCodec internally batches GPU operations
- Amortizes kernel launch overhead across all regions
- Enables parallel decoding on GPU

### Lines 691-729: Step 5-6 - Get Status and Transfer Results

```cpp
    // Step 5: Get processing status for each image
    std::vector<nvimgcodecProcessingStatus_t> decode_statuses(
        valid_streams.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
    size_t status_size = valid_streams.size();
    status = nvimgcodecFutureGetProcessingStatus(
        decode_future.get(), decode_statuses.data(), &status_size);

    // Synchronize if using GPU
    if (use_device_memory)
    {
        cudaDeviceSynchronize();
    }

    // Step 6: Transfer successful results to output
    for (size_t vi = 0; vi < valid_indices.size(); ++vi)
    {
        size_t i = valid_indices[vi];
        if (decode_statuses[vi] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            results[i].buffer = reinterpret_cast<uint8_t*>(
                decode_buffers[i].release());  // Transfer ownership
            results[i].success = true;
        }
    }

    return results;
```

**Explanation:**
- Gets per-image decode status from the batch future
- Syncs GPU if using device memory
- Transfers buffer ownership to results (prevents double-free)

---

## 3. nvimgcodec_processor.h - Processor Class Definition

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/loader/nvimgcodec_processor.h`

### Lines 32-40: RoiDecodeRequest Struct

```cpp
struct RoiDecodeRequest
{
    uint64_t location_index;  // Index in the location array
    uint32_t ifd_index;       // IFD (resolution level) index
    uint32_t x;               // ROI start x
    uint32_t y;               // ROI start y
    uint32_t width;           // ROI width
    uint32_t height;          // ROI height
};
```

**Purpose:** Internal request structure for queuing ROI decode operations.

### Lines 58-103: NvImageCodecProcessor Class

```cpp
class NvImageCodecProcessor : public cucim::loader::BatchDataProcessor
{
public:
    NvImageCodecProcessor(cuslide2::nvimgcodec::TiffFileParser* tiff_parser,
                          const int64_t* request_location,
                          const int64_t* request_size,
                          uint64_t location_len,
                          uint32_t batch_size,
                          uint32_t ifd_index,
                          const cucim::io::Device& out_device);

    ~NvImageCodecProcessor() override;

    // BatchDataProcessor interface
    uint32_t request(...) override;
    uint32_t wait_batch(...) override;
    std::shared_ptr<cucim::cache::ImageCacheValue> wait_for_processing(uint32_t index) override;
    void shutdown() override;

    // Custom methods
    uint32_t preferred_loader_prefetch_factor() const;
    uint8_t* get_decoded_data(uint64_t location_index) const;

private:
    bool decode_roi_batch(const std::vector<RoiDecodeRequest>& requests);
    
    // ... member variables ...
};
```

**Key Design Points:**
- Inherits from `cucim::loader::BatchDataProcessor` (cuslide1 infrastructure)
- Integrates with `ThreadBatchDataLoader` for multi-threaded loading
- Wraps `decode_batch_regions_nvimgcodec()` for the actual decoding

---

## 4. nvimgcodec_processor.cpp - Processor Implementation

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/loader/nvimgcodec_processor.cpp`

### Lines 25-95: Constructor

```cpp
NvImageCodecProcessor::NvImageCodecProcessor(
    cuslide2::nvimgcodec::TiffFileParser* tiff_parser,
    const int64_t* request_location,
    const int64_t* request_size,
    uint64_t location_len,
    uint32_t batch_size,
    uint32_t ifd_index,
    const cucim::io::Device& out_device)
    : cucim::loader::BatchDataProcessor(batch_size),
      tiff_parser_(tiff_parser),
      ifd_index_(ifd_index),
      out_device_(out_device),
      request_location_(request_location),
      location_len_(location_len)
{
    // Get ROI dimensions
    roi_width_ = static_cast<uint32_t>(request_size[0]);
    roi_height_ = static_cast<uint32_t>(request_size[1]);

    // Calculate buffer size (RGB, 8-bit)
    const auto& ifd_info = tiff_parser_->get_ifd(ifd_index_);
    uint32_t num_channels = ifd_info.num_channels > 0 ? ifd_info.num_channels : 3;
    roi_size_bytes_ = static_cast<size_t>(roi_width_) * roi_height_ * num_channels;

    // Determine output memory type
    std::string device_str = std::string(out_device_);
    use_device_memory_ = (device_str.find("cuda") != std::string::npos);

    // Calculate nvImageCodec batch size (up to 64)
    cuda_batch_size_ = std::min(
        static_cast<uint32_t>(location_len),
        std::min(batch_size * 2, MAX_NVIMGCODEC_BATCH_SIZE));

    // Create CUDA stream for async operations
    if (use_device_memory_)
    {
        CUDA_ERROR(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
}
```

**Key Points:**
- Stores reference to TIFF parser (has main_code_stream)
- Calculates optimal nvImageCodec batch size (max 64)
- Creates non-blocking CUDA stream for GPU operations

### Lines 134-179: request() Method

```cpp
uint32_t NvImageCodecProcessor::request(
    std::deque<uint32_t>& batch_item_counts, 
    uint32_t num_remaining_patches)
{
    // Build batch of ROI decode requests
    std::vector<RoiDecodeRequest> batch_requests;

    {
        std::lock_guard<std::mutex> lock(request_mutex_);

        while (next_decode_index_ < location_len_ && 
               batch_requests.size() < cuda_batch_size_)
        {
            RoiDecodeRequest req;
            req.location_index = next_decode_index_;
            req.ifd_index = ifd_index_;
            req.x = static_cast<uint32_t>(request_location_[next_decode_index_ * 2]);
            req.y = static_cast<uint32_t>(request_location_[next_decode_index_ * 2 + 1]);
            req.width = roi_width_;
            req.height = roi_height_;

            batch_requests.push_back(req);
            ++next_decode_index_;
        }
    }

    if (batch_requests.empty())
    {
        return 0;
    }

    // Decode the batch
    if (!decode_roi_batch(batch_requests))
    {
        return 0;
    }

    return static_cast<uint32_t>(batch_requests.size());
}
```

**Explanation:**
- Called by ThreadBatchDataLoader to request next batch
- Builds requests from location array (`[x0,y0,x1,y1,...]`)
- Limits batch size to `cuda_batch_size_` (max 64)
- Calls `decode_roi_batch()` for actual decoding

### Lines 251-331: decode_roi_batch() Method

```cpp
bool NvImageCodecProcessor::decode_roi_batch(
    const std::vector<RoiDecodeRequest>& requests)
{
    if (requests.empty() || !tiff_parser_)
    {
        return false;
    }

    // Build regions for batch decode
    std::vector<cuslide2::nvimgcodec::RoiRegion> regions;
    regions.reserve(requests.size());

    for (const auto& req : requests)
    {
        cuslide2::nvimgcodec::RoiRegion region;
        region.ifd_index = req.ifd_index;
        region.x = req.x;
        region.y = req.y;
        region.width = req.width;
        region.height = req.height;
        regions.push_back(region);
    }

    // Get IFD info
    const auto& ifd_info = tiff_parser_->get_ifd(ifd_index_);
    std::vector<const cuslide2::nvimgcodec::IfdInfo*> ifd_infos = { &ifd_info };

    // Perform batch decode using nvImageCodec API
    auto results = cuslide2::nvimgcodec::decode_batch_regions_nvimgcodec(
        ifd_infos,
        tiff_parser_->get_main_code_stream(),
        regions,
        out_device_);

    // Store results in cache
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        for (size_t i = 0; i < requests.size(); ++i)
        {
            uint64_t loc_idx = requests[i].location_index;

            if (results[i].success && results[i].buffer)
            {
                if (use_device_memory_)
                {
                    // Store GPU pointer directly
                    decoded_data_gpu_[loc_idx] = results[i].buffer;
                }
                else
                {
                    // Copy to internal storage and free original
                    decoded_data_cpu_[loc_idx].resize(roi_size_bytes_);
                    std::memcpy(decoded_data_cpu_[loc_idx].data(), 
                               results[i].buffer, roi_size_bytes_);
                    free(results[i].buffer);
                }
                decode_complete_[loc_idx] = true;
                ++completed_decode_count_;
            }
        }
    }

    // Notify waiters
    cache_cond_.notify_all();

    return true;
}
```

**Key Points:**
- Converts internal request format to `RoiRegion` structs
- Calls `decode_batch_regions_nvimgcodec()` - the actual batch decode
- Stores results in internal cache (CPU map or GPU pointer map)
- Notifies waiting threads via condition variable

---

## 5. ifd.cpp - Integration with Batch Loading

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp`

### Line 29: Include New Processor

```cpp
#include "cuslide/loader/nvimgcodec_processor.h"
```

### Lines 272-277: Batch Path Decision

```cpp
    // BATCH DECODING PATH: Multiple locations or batch_size > 1
    // Uses ThreadBatchDataLoader with NvImageCodecProcessor for parallel ROI decoding
    if (location_len > 1 || batch_size > 1 || request->num_workers > 0)
    {
        if (batch_size > 1)
        {
```

**Logic:** Uses batch path when:
- Multiple locations requested
- Batch size > 1
- num_workers > 0 (explicit parallel request)

### Lines 370-383: Create NvImageCodecProcessor for GPU

```cpp
        if (out_device.type() == cucim::io::DeviceType::kCUDA)
        {
            raster_type = cucim::io::DeviceType::kCUDA;

            // Create NvImageCodecProcessor for GPU-accelerated batch decoding
            auto nvimgcodec_processor = std::make_unique<cuslide2::loader::NvImageCodecProcessor>(
                tiff->nvimgcodec_parser_.get(),
                request_location->data(),
                request_size->data(),
                adjusted_location_len,
                batch_size,
                ifd->ifd_index_,
                out_device);

            prefetch_factor = nvimgcodec_processor->preferred_loader_prefetch_factor();

            batch_processor = std::move(nvimgcodec_processor);
        }
```

**Explanation:**
- For CUDA output, creates `NvImageCodecProcessor`
- Processor wraps nvImageCodec batch decoding
- Gets recommended prefetch factor from processor

### Lines 386-390: Create ThreadBatchDataLoader

```cpp
        // Create ThreadBatchDataLoader
        auto loader = std::make_unique<cucim::loader::ThreadBatchDataLoader>(
            load_func, std::move(batch_processor), out_device,
            std::move(request_location), std::move(request_size),
            adjusted_location_len, one_raster_size, batch_size, prefetch_factor, num_workers);
```

**Explanation:**
- Creates the multi-threaded loader
- Passes the batch processor for GPU acceleration
- Loader coordinates workers and batch processing

---

## 6. CMakeLists.txt - Build Configuration

**File:** `cpp/plugins/cucim.kit.cuslide2/CMakeLists.txt`

### Lines 164-166: Add New Source Files

```cmake
    # ThreadBatchDataLoader integration with nvImageCodec
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/loader/nvimgcodec_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuslide/loader/nvimgcodec_processor.h
)
```

**Purpose:** Adds the new processor files to the build.

---

## 7. test_batch_decoding.py - Python Tests

**File:** `python/cucim/tests/unit/clara/test_batch_decoding.py`

### Test Classes

| Class | Purpose | Tests |
|-------|---------|-------|
| `TestBatchDecoding` | CPU batch decoding | 7 tests |
| `TestBatchDecodingCUDA` | GPU batch decoding (JPEG only) | 2 tests |
| `TestBatchDecodingPerformance` | Scaling tests | 8 tests |

### Key Test: test_batch_read_multiple_locations (Lines 24-47)

```python
def test_batch_read_multiple_locations(self, testimg_tiff_stripe_4096x4096_256):
    """Test reading multiple locations in a single call."""
    with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
        # Define multiple locations
        locations = [
            (0, 0),
            (256, 256),
            (512, 512),
            (768, 768),
        ]
        size = (256, 256)
        level = 0

        # Read with multiple workers (triggers batch decoding path)
        gen = img.read_region(locations, size, level, num_workers=2)

        results = list(gen)
        assert len(results) == len(locations)

        for i, result in enumerate(results):
            arr = np.asarray(result)
            assert arr.shape == (256, 256, 3), f"Region {i} has wrong shape"
            assert np.any(arr > 0), f"Region {i} is all zeros"
```

**Verification:**
- Reads 4 locations in one call
- Triggers batch path via `num_workers=2`
- Verifies all regions decoded with correct shape
- Confirms actual image data (not zeros)

---

## Architecture Summary

```
Python: img.read_region(locations, size, level, num_workers=4)
           │
           ▼
     ┌─────────────────────────────────────────────────────────┐
     │                    IFD::read()                          │
     │  • Detects location_len > 1 or batch_size > 1           │
     │  • Creates NvImageCodecProcessor                        │
     │  • Creates ThreadBatchDataLoader                        │
     └─────────────────────────────────────────────────────────┘
           │
           ▼
     ┌─────────────────────────────────────────────────────────┐
     │              NvImageCodecProcessor::request()           │
     │  • Builds RoiDecodeRequest batch                        │
     │  • Calls decode_roi_batch()                             │
     └─────────────────────────────────────────────────────────┘
           │
           ▼
     ┌─────────────────────────────────────────────────────────┐
     │           decode_batch_regions_nvimgcodec()             │
     │                                                         │
     │  1. nvimgcodecCodeStreamGetSubCodeStream() × N          │
     │     (Create ROI views for each region)                  │
     │                                                         │
     │  2. nvimgcodecDecoderDecode()                           │
     │     (SINGLE CALL with arrays - THE KEY OPTIMIZATION)    │
     │                                                         │
     │  3. nvimgcodecFutureGetProcessingStatus()               │
     │     (Get per-image results)                             │
     └─────────────────────────────────────────────────────────┘
           │
           ▼
     ┌─────────────────────────────────────────────────────────┐
     │              Results returned to Python                 │
     │  • Each region as numpy/cupy array                      │
     │  • Shape: [height, width, 3]                            │
     └─────────────────────────────────────────────────────────┘
```

---

## Performance Benefits

1. **Single Decode Call**: Instead of N separate `decoder.decode()` calls, ONE batched call
2. **GPU Kernel Amortization**: Overhead of kernel launches spread across N regions
3. **Parallel Decoding**: nvImageCodec can decode multiple regions in parallel on GPU
4. **Memory Efficiency**: Sub-code streams share the main CodeStream's buffer
5. **Thread Pool**: ThreadBatchDataLoader provides additional parallelism for CPU operations

---

## Memory Management

| Operation | CPU Memory | GPU Memory |
|-----------|------------|------------|
| Allocation | `malloc()` | `cudaMalloc()` |
| Deallocation | `free()` | `cudaFree()` |
| RAII Wrapper | `DecodeBuffer` class | Same |

**Important:** Uses `malloc()` (not `cudaMallocHost()`) for CPU buffers because cuCIM's cleanup code uses `free()`.

