# Single-Threaded Test - Diagnostic Bypass

## Purpose

After all previous fixes failed, I've added a diagnostic bypass to force **single-threaded execution**. This will tell us definitively whether:

1. **The lambda can execute** - If it works single-threaded, the lambda is valid
2. **The decode code works** - If JPEG2000 decoding succeeds, the decoder is fine  
3. **The problem is threading** - If single-threaded works but multi-threaded crashes, it's a thread pool issue

## The Change

Added at line ~1246 in `ifd.cpp`:

```cpp
// TEMPORARY: Force single-threaded execution to test if decode works
bool force_single_threaded = true;

if (force_single_threaded || !loader || !(*loader))
{
    fmt::print("🔍 Executing decode_func directly (FORCED SINGLE-THREADED TEST)\n");
    fflush(stdout);
    decode_func();  // Execute directly, no thread pool!
    fmt::print("🔍 decode_func completed successfully!\n");
    fflush(stdout);
}
else
{
    // Normal multi-threaded path (currently disabled)
    loader->enqueue(std::move(decode_func), ...);
}
```

## Expected Outcomes

### Scenario A: Single-Threaded Works ✅
```
🔍 Executing decode_func directly (FORCED SINGLE-THREADED TEST)
🔍🔍🔍 decode_func: LAMBDA INVOKED! index=0
🔍 decode_func: START - ...
🔍 Decoding JPEG2000 tile (RGB) ...
✅ JPEG2000 tile decoded successfully
🔍 decode_func completed successfully!
```

**Conclusion**: The lambda and decode code are FINE. Problem is:
- Thread pool implementation (taskflow)
- Worker thread stack size
- Some thread-specific issue

**Next Steps**:
1. Increase worker thread stack size
2. Try different thread pool backend
3. Use different taskflow settings

### Scenario B: Single-Threaded Also Crashes ❌
```
🔍 Executing decode_func directly (FORCED SINGLE-THREADED TEST)
🔍🔍🔍 decode_func: LAMBDA INVOKED! index=0
Segmentation fault
```

**Conclusion**: Problem is in the decode logic itself, not threading.

**Next Steps**:
1. Debug the decode path (likely nvImageCodec call)
2. Check pointer validity
3. Check buffer sizes

### Scenario C: Lambda Never Executes ❌
```
🔍 Executing decode_func directly (FORCED SINGLE-THREADED TEST)
Segmentation fault
```

**Conclusion**: Lambda construction itself is broken.

**Next Steps**:
1. Simplify shared_ptr struct
2. Check if some captured value is invalid
3. Try even simpler lambda

## Rebuild and Test

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim
./setup_and_build.sh
./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs
```

## Analysis

This test will give us definitive information:
- **If it works**: Thread pool is the problem → investigate taskflow/threading
- **If it crashes**: Decode logic is the problem → investigate nvImageCodec/buffers
- **If lambda fails**: Lambda construction is the problem → simplify further

This is a critical diagnostic that will point us in the right direction!

## After Testing

Once we know the result, we can:
1. **If single-threaded works**: Focus on fixing the thread pool issue
2. **If single-threaded fails**: Focus on fixing the decode logic
3. Either way, we'll know exactly where the problem is

## Reverting

After testing, if single-threaded works, we can:
1. Change `force_single_threaded = false` to re-enable multi-threading
2. Apply thread-pool-specific fixes
3. Or leave it single-threaded if performance is acceptable

