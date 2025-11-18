# Segfault Debug Changes

## Summary
Fixed uninitialized variables and added extensive debug output to trace the segmentation fault occurring during JPEG2000 decoding with nvImageCodec.

## Files Modified

### 1. `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp`

#### Bug Fixes - Uninitialized Variables

**Issue:** Three instances of uninitialized `status_size` and `decode_status` variables were causing undefined behavior when passed to nvImageCodec API functions.

**Locations Fixed:**

1. **Line 622-623** - `decode_jpeg2k_nvimgcodec()`:
```cpp
// BEFORE (BUG):
size_t status_size;
nvimgcodecProcessingStatus_t decode_status;

// AFTER (FIXED):
size_t status_size = 1;
nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
```

2. **Line 824-825** - `decode_ifd_nvimgcodec()`:
```cpp
// BEFORE (BUG):
nvimgcodecProcessingStatus_t decode_status;
size_t status_size;

// AFTER (FIXED):
nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
size_t status_size = 1;
```

3. **Line 1050-1051** - `decode_ifd_region_nvimgcodec()`:
```cpp
// BEFORE (BUG):
nvimgcodecProcessingStatus_t decode_status;
size_t status_size;

// AFTER (FIXED):
nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
size_t status_size = 1;
```

#### Debug Output Added

**Purpose:** Trace exactly where the crash occurs during decoding.

**Line 444-449** - Entry point debug:
```cpp
fmt::print("🔍 decode_jpeg2k_nvimgcodec: ENTRY - fd={}, offset={}, size={}\n", fd, offset, size);
fmt::print("🔍 decode_jpeg2k_nvimgcodec: Getting manager instance...\n");
auto& manager = NvImageCodecManager::instance();
fmt::print("🔍 decode_jpeg2k_nvimgcodec: Got manager instance\n");
```

### 2. `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp`

#### Debug Output Added

**Line 806** - Task execution entry point:
```cpp
auto decode_func = [=, &image_cache]() {
    fmt::print("🔍 decode_func: START - index={}, compression={}\n", index, compression_method);
    // ...
```

## Expected Debug Output Flow

When the code runs successfully, you should see output in this order:

```
📍 location_len=1, batch_size=1, num_workers=1
📍 Entering multi-location/batch/worker path
📍 Creating ThreadBatchDataLoader
📍 ThreadBatchDataLoader created
📍 Calling loader->request(1)
📍 loader->request() completed
📍 Calling loader->next_data()
🔍 decode_func: START - index=X, compression=33005
🔍 Decoding JPEG2000 tile (RGB) at offset Y, size Z
🔍 About to call decode_jpeg2k_nvimgcodec from ifd.cpp...
🔍 decode_jpeg2k_nvimgcodec: ENTRY - fd=A, offset=B, size=C
🔍 decode_jpeg2k_nvimgcodec: Getting manager instance...
🔍 decode_jpeg2k_nvimgcodec: Got manager instance
🚀 nvImageCodec JPEG2000 decode: Starting, size=X bytes, device=cpu
[... rest of decode process ...]
✅ JPEG2000 tile decoded successfully
```

## What To Look For

1. **If crash occurs before "🔍 decode_func: START":**
   - Issue is in ThreadBatchDataLoader setup or task enqueueing

2. **If crash occurs after "🔍 decode_func: START" but before "🔍 decode_jpeg2k_nvimgcodec: ENTRY":**
   - Issue is in lambda capture or ifd.cpp logic before decode call

3. **If crash occurs after "ENTRY" but before "Getting manager instance":**
   - Issue with function parameter passing or stack corruption

4. **If crash occurs during "Getting manager instance":**
   - Issue with singleton initialization or thread safety

5. **If crash occurs after "Got manager instance":**
   - Issue within nvImageCodec API calls or buffer allocation

## Potential Root Causes

Based on the segfault occurring during `loader->next_data()`:

1. ✅ **Uninitialized variables** - Fixed in this commit
2. **Thread safety issue** - NvImageCodecManager singleton accessed from worker thread
3. **Memory corruption** - Buffer allocation/deallocation mismatch
4. **nvImageCodec library issue** - Internal crash within nvImageCodec itself
5. **File descriptor issue** - fd being used across threads

## Next Steps

1. Rebuild the plugin with these changes
2. Run the test and observe which debug message is the last one printed
3. The crash location will tell us exactly where the problem is
4. Based on the crash location, we can implement a targeted fix

## Rebuild Command

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2
./build.sh
```

## Test Command

```bash
./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs
```

