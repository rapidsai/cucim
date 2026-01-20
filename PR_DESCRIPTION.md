# Add batch ROI decoding using nvImageCodec v0.7.0+ API

## Description

This PR implements batch ROI decoding for cuslide2 using nvImageCodec v0.7.0+'s native batch decoding API, following guidance from the nvImageCodec team.

### Background

Per nvImageCodec team (Michal Kepa):
> "Read image into CodeStream, then call multiple `get_sub_code_stream()` on this main CodeStream with different ROI and decode them all in a single `decoder.decode()` call"

This approach provides significant performance improvements by:
- Amortizing GPU kernel launch overhead across multiple regions
- Enabling parallel decoding of multiple ROIs
- Reducing memory allocation overhead through batching

## Changes

### New Files

- `cpp/plugins/cucim.kit.cuslide2/src/cuslide/loader/nvimgcodec_processor.h`
- `cpp/plugins/cucim.kit.cuslide2/src/cuslide/loader/nvimgcodec_processor.cpp`
  - `NvImageCodecProcessor` class inheriting from `BatchDataProcessor`
  - Integrates with existing `ThreadBatchDataLoader` infrastructure
  - Supports both CPU and CUDA output devices

- `python/cucim/tests/unit/clara/test_batch_decoding.py`
  - Comprehensive test suite with 47 tests

### Modified Files

- `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.h`
  - Added `RoiRegion` and `BatchDecodeResult` structs
  - Added `decode_batch_regions_nvimgcodec()` function declaration

- `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp`
  - Implemented `decode_batch_regions_nvimgcodec()` using:
    1. `nvimgcodecCodeStreamGetSubCodeStream()` with ROI for each region
    2. Single `nvimgcodecDecoderDecode()` call with all streams
    3. Batch result processing

- `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp`
  - Updated `IFD::read()` to use `ThreadBatchDataLoader` with `NvImageCodecProcessor`
  - Supports `num_workers`, `batch_size`, `prefetch_factor`, `shuffle`, `drop_last` parameters

- `cpp/plugins/cucim.kit.cuslide2/CMakeLists.txt`
  - Added new loader source files to build

## Architecture

```
IFD::read()
    |
    +-- Single Location (location_len=1)
    |   +-- decode_ifd_region_nvimgcodec()
    |
    +-- Multiple Locations (location_len>1 or batch_size>1)
        +-- ThreadBatchDataLoader + NvImageCodecProcessor
            +-- decode_batch_regions_nvimgcodec()
                +-- nvimgcodecCodeStreamGetSubCodeStream() x N
                +-- nvimgcodecDecoderDecode() (single batch call)
```

## Test Results

All 47 tests passing:

| Test Category | Compression Types | Count | Status |
|---------------|-------------------|-------|--------|
| TestBatchDecoding (CPU) | JPEG, Deflate, Raw | 21 | PASS |
| TestBatchDecodingCUDA | JPEG | 2 | PASS |
| TestBatchDecodingPerformance | JPEG, Deflate, Raw | 24 | PASS |

**Note:** CUDA output is only supported for JPEG compression. Deflate and Raw use CPU decoding with optional GPU memory transfer.

## How to Build

```bash
cd cucim/cpp/plugins/cucim.kit.cuslide2
mkdir -p build-release && cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUCIM_HAS_NVIMGCODEC=ON
make -j$(nproc)
```

## How to Run Tests

```bash
# Run all batch decoding tests
cd cucim
pytest python/cucim/tests/unit/clara/test_batch_decoding.py -v

# Run specific test categories
pytest python/cucim/tests/unit/clara/test_batch_decoding.py::TestBatchDecoding -v
pytest python/cucim/tests/unit/clara/test_batch_decoding.py::TestBatchDecodingCUDA -v
pytest python/cucim/tests/unit/clara/test_batch_decoding.py::TestBatchDecodingPerformance -v
```

## Example Usage

```python
from cucim import CuImage
import numpy as np

# Open TIFF file
img = CuImage("slide.tiff")

# Batch decode multiple locations
locations = [(0, 0), (256, 256), (512, 512), (768, 768)]
size = (256, 256)

# CPU output with parallel workers
for region in img.read_region(locations, size, level=0, num_workers=4):
    arr = np.asarray(region)
    print(f"Decoded: {arr.shape}")

# CUDA output (JPEG only)
import cupy as cp
for region in img.read_region(locations, size, level=0, num_workers=4, device="cuda"):
    arr = cp.asarray(region)
    print(f"GPU decoded: {arr.shape}")
```

## Related Issues

- Ref: gh-998

