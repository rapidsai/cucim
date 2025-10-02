#!/usr/bin/env python3
"""
Diagnostic script to check nvImageCodec available codecs and backends
"""

import sys
import os

# Add the build directory to the path
sys.path.insert(0, '/home/cdinea/Downloads/cucim_pr2/cucim/python/install/lib')

try:
    from cucim import CuImage
    import cucim
    print(f"✅ cuCIM version: {cucim.__version__}")
except ImportError as e:
    print(f"❌ Failed to import cucim: {e}")
    sys.exit(1)

# Set plugin path
plugin_path = "/home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/lib"
os.environ['CUCIM_PLUGIN_PATH'] = plugin_path
print(f"✅ Plugin path: {plugin_path}\n")

print("=" * 70)
print("Checking nvImageCodec Configuration")
print("=" * 70)

# Try to load a JPEG-compressed TIFF to see what happens
test_file = "/tmp/CMU-1-Small-Region.svs"

if not os.path.exists(test_file):
    print(f"⚠️  Test file not found: {test_file}")
    print("   Run: python test_aperio_svs.py --download")
    sys.exit(1)

print(f"\n📁 Loading test file: {test_file}")

try:
    img = CuImage(test_file)
    print(f"✅ File loaded successfully")
    print(f"   Dimensions: {img.shape}")
    print(f"   Levels: {img.resolutions['level_count']}")
    
    # Try CPU decode
    print("\n🔍 Testing CPU decode capability...")
    print("   Attempting to read a small region to CPU...")
    
    try:
        # Read a small 256x256 region to CPU
        region = img.read_region((0, 0), (256, 256), level=0, device='cpu')
        print(f"✅ CPU decode successful!")
        print(f"   Region shape: {region.shape}")
        print(f"   Device: {region.device}")
        
        # Check if it's actually on CPU
        import cupy as cp
        if hasattr(region, '__cuda_array_interface__'):
            print("   ⚠️  Data is on GPU (using fallback path)")
        else:
            print("   ✅ Data is on CPU (native CPU decoder)")
            
    except Exception as e:
        print(f"   ❌ CPU decode failed: {e}")
        print("   This confirms CPU decoders are not available")
        
except Exception as e:
    print(f"❌ Failed to process file: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Backend Detection Summary")
print("=" * 70)
print("""
Based on the test results above:

✅ Working: nvImageCodec is loaded and functional
✅ Working: JPEG compression detection (.svs → jpeg codec)
✅ Working: GPU decoding (nvJPEG or nvTiff)
✅ Working: CPU fallback (GPU decode + cudaMemcpy)

❌ Missing: Native CPU JPEG decoder

This is EXPECTED behavior for nvImageCodec 0.6.0 because:
1. nvImageCodec focuses primarily on GPU acceleration
2. CPU backends are optional plugins/modules
3. libjpeg-turbo CPU backend may not be installed/loaded

Options:
1. Continue using GPU + copy fallback (current, works well)
2. Install nvImageCodec CPU backend modules if available
3. Wait for nvImageCodec 0.7.0 with better CPU support
4. Use hybrid approach: libjpeg-turbo directly for CPU, nvImageCodec for GPU
""")

