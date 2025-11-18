#!/usr/bin/env python3
"""Test Philips TIFF support in cuslide2"""

import sys
import cucim
from cucim.clara import _set_plugin_root
import numpy as np
import time

def test_philips_tiff(file_path, plugin_lib):
    """Test Philips TIFF loading and decoding"""
    
    print("=" * 60)
    print("🔬 Testing Philips TIFF with cuslide2")
    print("=" * 60)
    print(f"📁 File: {file_path}")
    
    # Set plugin root to use cuslide2
    _set_plugin_root(str(plugin_lib))
    print(f"✅ Plugin root set: {plugin_lib}")
    print()
    
    # Load image
    print("📂 Loading Philips TIFF file...")
    start = time.time()
    img = cucim.CuImage(file_path)
    load_time = time.time() - start
    print(f"✅ Loaded in {load_time:.3f}s")
    print()
    
    # Check detection
    print("📊 Image Information:")
    print(f"  Format: Philips TIFF")
    print(f"  Dimensions: {img.shape}")
    level_count = img.resolutions['level_count']
    print(f"  Levels: {level_count}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Device: {img.device}")
    print()
    
    # Display resolution levels
    print("🔍 Resolution Levels:")
    level_count = img.resolutions['level_count']
    level_dimensions = img.resolutions['level_dimensions']
    level_downsamples = img.resolutions['level_downsamples']
    for level in range(level_count):
        dims = level_dimensions[level]
        downsample = level_downsamples[level]
        print(f"  Level {level}: {dims[0]}x{dims[1]} (downsample: {downsample:.1f}x)")
    print()
    
    # Check for Philips metadata
    print("📋 Philips Metadata:")
    metadata = img.metadata
    if 'philips' in metadata:
        philips_data = metadata['philips']
        print(f"  ✅ Found {len(philips_data)} Philips metadata entries")
        # Show some important keys
        important_keys = [
            'DICOM_PIXEL_SPACING',
            'DICOM_MANUFACTURER', 
            'PIM_DP_IMAGE_TYPE',
            'DICOM_SOFTWARE_VERSIONS',
            'PIM_DP_IMAGE_ROWS',
            'PIM_DP_IMAGE_COLUMNS'
        ]
        for key in important_keys:
            if key in philips_data:
                print(f"    {key}: {philips_data[key]}")
        print(f"    ... and {len(philips_data) - len(important_keys)} more entries")
    else:
        print("  ⚠️  No Philips metadata found")
    print()
    
    # Check MPP (microns per pixel)
    print(f"📏 Pixel Spacing:")
    if 'philips' in metadata and 'DICOM_PIXEL_SPACING' in metadata['philips']:
        spacing = metadata['philips']['DICOM_PIXEL_SPACING']
        print(f"  DICOM Pixel Spacing: {spacing[0]*1000:.4f} x {spacing[1]*1000:.4f} μm/pixel")
    if 'openslide.mpp-x' in metadata:
        print(f"  OpenSlide MPP-X: {metadata['openslide.mpp-x']} μm/pixel")
        print(f"  OpenSlide MPP-Y: {metadata['openslide.mpp-y']} μm/pixel")
    print()
    
    # Test GPU decode
    print("🚀 Testing GPU decode (nvImageCodec)...")
    try:
        start = time.time()
        region = img.read_region((0, 0), (512, 512), level=0, device="cuda")
        decode_time = time.time() - start
        print(f"✅ GPU decode successful!")
        print(f"  Time: {decode_time:.4f}s")
        print(f"  Shape: {region.shape}")
        print(f"  Device: {region.device}")
        
        # Check pixel values
        if hasattr(region, 'get'):
            region_cpu = region.get()
            print(f"  Pixel range: [{region_cpu.min()}, {region_cpu.max()}]")
            print(f"  Mean value: {region_cpu.mean():.2f}")
        print()
    except Exception as e:
        print(f"❌ GPU decode failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test CPU decode (expected to fail for cuslide2)
    print("🖥️  Testing CPU decode...")
    try:
        start = time.time()
        region = img.read_region((0, 0), (512, 512), level=0, device="cpu")
        decode_time = time.time() - start
        
        # Check if we got actual data
        if hasattr(region, '__array_interface__') or hasattr(region, '__cuda_array_interface__'):
            import numpy as np
            if hasattr(region, 'get'):  # CuPy array
                region_cpu = region.get()
            else:
                region_cpu = np.asarray(region)
            
            if region_cpu.size > 0:
                pixel_sum = region_cpu.sum()
                pixel_mean = region_cpu.mean()
                print(f"⚠️  CPU decode returned data (unexpected for cuslide2!):")
                print(f"  Time: {decode_time:.4f}s")
                print(f"  Shape: {region_cpu.shape}")
                print(f"  Pixel sum: {pixel_sum}, mean: {pixel_mean:.2f}")
                print(f"  ⚠️  This suggests CPU fallback is active!")
            else:
                print(f"⚠️  CPU decode returned empty data:")
                print(f"  Time: {decode_time:.4f}s (likely returning cached/empty)")
        else:
            print(f"⚠️  CPU decode returned unknown type: {type(region)}")
        print()
    except Exception as e:
        print(f"✅ CPU decode failed (as expected for cuslide2 - GPU only):")
        print(f"  {e}")
        print()
    
    # Test associated images
    print("🖼️  Testing associated images...")
    try:
        label = img.associated_image('label')
        print(f"  ✅ Label: {label.shape}")
    except Exception as e:
        print(f"  ⚠️  Label not found: {e}")
    
    try:
        macro = img.associated_image('macro')
        print(f"  ✅ Macro: {macro.shape}")
    except Exception as e:
        print(f"  ⚠️  Macro not found: {e}")
    
    try:
        thumbnail = img.associated_image('thumbnail')
        print(f"  ✅ Thumbnail: {thumbnail.shape}")
    except Exception as e:
        print(f"  ⚠️  Thumbnail not found: {e}")
    print()
    
    # Test larger tile
    print("📏 Testing larger tile (2048x2048)...")
    try:
        start = time.time()
        region = img.read_region((0, 0), (2048, 2048), level=0, device="cuda")
        decode_time = time.time() - start
        print(f"  ✅ GPU: {decode_time:.4f}s")
        print(f"     Shape: {region.shape}")
    except Exception as e:
        print(f"  ⚠️  Large tile failed: {e}")
    print()
    
    # Test multi-level reads
    print("🔀 Testing multi-level reads...")
    level_count = img.resolutions['level_count']
    level_dimensions = img.resolutions['level_dimensions']
    for level in range(min(3, level_count)):
        try:
            start = time.time()
            dims = level_dimensions[level]
            read_size = (min(512, dims[0]), min(512, dims[1]))
            region = img.read_region((0, 0), read_size, level=level, device="cuda")
            decode_time = time.time() - start
            print(f"  ✅ Level {level}: {decode_time:.4f}s ({region.shape})")
        except Exception as e:
            print(f"  ❌ Level {level} failed: {e}")
    print()
    
    print("✅ Philips TIFF test completed!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_philips_tiff.py <philips_tiff_file> [plugin_lib_path]")
        print()
        print("Example:")
        print("  python test_philips_tiff.py /tmp/philips_sample.tiff")
        print()
        print("Download test data from:")
        print("  https://openslide.cs.cmu.edu/download/openslide-testdata/Philips-TIFF/")
        sys.exit(1)
    
    file_path = sys.argv[1]
    plugin_lib = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/lib"
    
    test_philips_tiff(file_path, plugin_lib)

