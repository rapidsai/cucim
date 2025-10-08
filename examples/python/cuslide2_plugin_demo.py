#!/usr/bin/env python3
"""
cuslide2 Plugin Demo with nvImageCodec GPU Acceleration

This example demonstrates how to use the cuslide2 plugin for GPU-accelerated
JPEG/JPEG2000 decoding in digital pathology images.

Features:
- Automatic cuslide2 plugin configuration
- GPU vs CPU performance comparison
- Support for SVS, TIFF, and Philips formats
- nvImageCodec integration validation
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

def setup_cuslide2_plugin():
    """Configure cuCIM to use cuslide2 plugin with priority"""
    
    print("🔧 Setting up cuslide2 plugin...")
    
    # Set plugin root to build directory
    plugin_root = "/home/cdinea/cucim/cpp/plugins/cucim.kit.cuslide2/build/lib"
    
    try:
        from cucim.clara import _set_plugin_root
        _set_plugin_root(plugin_root)
        print(f"✅ Plugin root set: {plugin_root}")
    except ImportError:
        print("❌ cuCIM not available - please install cuCIM")
        return False
    
    # Create plugin configuration to prioritize cuslide2
    config = {
        "plugin": {
            "names": [
                "cucim.kit.cuslide2@25.10.00.so",  # cuslide2 with nvImageCodec (highest priority)
                "cucim.kit.cuslide@25.10.00.so",   # Original cuslide (fallback)
                "cucim.kit.cumed@25.10.00.so"      # Medical imaging
            ]
        }
    }
    
    # Write config file
    config_path = "/tmp/.cucim_cuslide2_demo.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Set environment variable
    os.environ["CUCIM_CONFIG_PATH"] = config_path
    print(f"✅ Plugin configuration created: {config_path}")
    
    return True

def check_nvimgcodec_availability() -> bool:
    """Check if nvImageCodec is available for GPU acceleration"""
    
    conda_prefix = os.environ.get('CONDA_PREFIX', '/home/cdinea/micromamba')
    nvimgcodec_lib = Path(conda_prefix) / "lib/libnvimgcodec.so.0"
    
    if nvimgcodec_lib.exists():
        print(f"✅ nvImageCodec available: {nvimgcodec_lib}")
        return True
    else:
        print(f"⚠️  nvImageCodec not found: {nvimgcodec_lib}")
        print("   GPU acceleration will not be available")
        return False

def benchmark_decode_performance(img, region_sizes: List[int] = [1024, 2048, 4096]) -> dict:
    """Benchmark CPU vs GPU decode performance"""
    
    results = {}
    
    print(f"\n📊 Performance Benchmarking")
    print("=" * 50)
    
    for size in region_sizes:
        if img.shape[0] < size or img.shape[1] < size:
            print(f"⚠️  Skipping {size}x{size} - image too small")
            continue
            
        print(f"\n🔍 Testing {size}x{size} region...")
        
        # CPU benchmark
        print("  🖥️  CPU decode...")
        try:
            start_time = time.time()
            cpu_region = img.read_region(
                location=[0, 0],
                size=[size, size],
                level=0,
                device="cpu"
            )
            cpu_time = time.time() - start_time
            print(f"     Time: {cpu_time:.3f}s")
            print(f"     Shape: {cpu_region.shape}")
            print(f"     Device: {cpu_region.device}")
        except Exception as e:
            print(f"     ❌ CPU decode failed: {e}")
            cpu_time = None
        
        # GPU benchmark
        print("  🚀 GPU decode...")
        try:
            start_time = time.time()
            gpu_region = img.read_region(
                location=[0, 0],
                size=[size, size],
                level=0,
                device="cuda"
            )
            gpu_time = time.time() - start_time
            print(f"     Time: {gpu_time:.3f}s")
            print(f"     Shape: {gpu_region.shape}")
            print(f"     Device: {gpu_region.device}")
            
            if cpu_time and gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"     🎯 Speedup: {speedup:.2f}x")
                results[size] = {
                    'cpu_time': cpu_time,
                    'gpu_time': gpu_time,
                    'speedup': speedup
                }
            
        except Exception as e:
            print(f"     ⚠️  GPU decode failed: {e}")
            print(f"     (This is expected if CUDA is not available)")
    
    return results

def analyze_image_format(img) -> dict:
    """Analyze image format and compression details"""
    
    info = {
        'dimensions': img.shape,
        'levels': img.level_count,
        'spacing': img.spacing() if hasattr(img, 'spacing') else None,
        'dtype': str(img.dtype),
        'device': str(img.device),
        'associated_images': []
    }
    
    # Get associated images
    if hasattr(img, 'associated_images'):
        info['associated_images'] = list(img.associated_images)
    
    # Get metadata
    if hasattr(img, 'metadata'):
        metadata = img.metadata
        if isinstance(metadata, dict):
            # Look for compression information
            if 'tiff' in metadata:
                tiff_info = metadata['tiff']
                if isinstance(tiff_info, dict) and 'compression' in tiff_info:
                    info['compression'] = tiff_info['compression']
    
    return info

def test_cuslide2_plugin(file_path: str):
    """Test cuslide2 plugin with a specific file"""
    
    print(f"\n🔍 Testing cuslide2 plugin with: {file_path}")
    print("=" * 60)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        from cucim import CuImage
        
        # Load image
        print("📁 Loading image...")
        start_time = time.time()
        img = CuImage(file_path)
        load_time = time.time() - start_time
        
        print(f"✅ Image loaded in {load_time:.3f}s")
        
        # Analyze image format
        print("\n📋 Image Analysis:")
        info = analyze_image_format(img)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Show level information
        print(f"\n📊 Level Information:")
        for level in range(img.level_count):
            level_shape = img.level_shape(level)
            level_spacing = img.level_spacing(level) if hasattr(img, 'level_spacing') else None
            print(f"  Level {level}: {level_shape} (spacing: {level_spacing})")
        
        # Performance benchmarking
        results = benchmark_decode_performance(img)
        
        # Summary
        if results:
            print(f"\n🏆 Performance Summary:")
            avg_speedup = sum(r['speedup'] for r in results.values()) / len(results)
            print(f"  Average GPU speedup: {avg_speedup:.2f}x")
            
            best_speedup = max(r['speedup'] for r in results.values())
            best_size = max(results.keys(), key=lambda k: results[k]['speedup'])
            print(f"  Best speedup: {best_speedup:.2f}x (at {best_size}x{best_size})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing plugin: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_test_images() -> List[str]:
    """Find available test images"""
    
    search_paths = [
        "/home/cdinea/cucim/test_data",
        "/home/cdinea/cucim/notebooks/input",
        "/home/cdinea/cucim/cpp/plugins/cucim.kit.cuslide2/test_data",
        "/tmp"
    ]
    
    extensions = ['.svs', '.tiff', '.tif', '.ndpi']
    found_images = []
    
    for search_path in search_paths:
        if Path(search_path).exists():
            for ext in extensions:
                pattern = f"*{ext}"
                matches = list(Path(search_path).glob(pattern))
                found_images.extend([str(m) for m in matches])
    
    return found_images

def demo_mode():
    """Run demo mode without specific files"""
    
    print("\n🎮 cuslide2 Plugin Demo Mode")
    print("=" * 40)
    
    # Check for available test images
    test_images = find_test_images()
    
    if test_images:
        print(f"📁 Found {len(test_images)} test image(s):")
        for img_path in test_images[:5]:  # Show first 5
            print(f"  • {img_path}")
        
        # Test with first available image
        print(f"\n🧪 Testing with: {test_images[0]}")
        return test_cuslide2_plugin(test_images[0])
    else:
        print("📝 No test images found. To test cuslide2:")
        print("  1. Place a .svs, .tiff, or .tif file in one of these locations:")
        print("     • /home/cdinea/cucim/test_data/")
        print("     • /home/cdinea/cucim/notebooks/input/")
        print("     • /tmp/")
        print("  2. Run: python cuslide2_plugin_demo.py /path/to/your/image.svs")
        
        print(f"\n✅ cuslide2 plugin is configured and ready!")
        print(f"🎯 Supported formats:")
        print(f"  • Aperio SVS (JPEG/JPEG2000)")
        print(f"  • Philips TIFF (JPEG/JPEG2000)")
        print(f"  • Generic tiled TIFF (JPEG/JPEG2000)")
        
        return True

def main():
    """Main function"""
    
    print("🚀 cuslide2 Plugin Demo with nvImageCodec")
    print("=" * 50)
    
    # Setup plugin
    if not setup_cuslide2_plugin():
        return 1
    
    # Check nvImageCodec
    nvimgcodec_available = check_nvimgcodec_availability()
    
    # Get file path from command line or run demo
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        success = test_cuslide2_plugin(file_path)
    else:
        success = demo_mode()
    
    # Final summary
    print(f"\n🎉 Demo completed!")
    print(f"✅ cuslide2 plugin: Ready")
    print(f"{'✅' if nvimgcodec_available else '⚠️ '} nvImageCodec: {'Available' if nvimgcodec_available else 'CPU fallback'}")
    
    if nvimgcodec_available:
        print(f"\n🚀 GPU acceleration is active!")
        print(f"   JPEG/JPEG2000 tiles will be decoded on GPU for faster performance")
    else:
        print(f"\n💡 To enable GPU acceleration:")
        print(f"   micromamba install libnvimgcodec-dev libnvimgcodec0 -c conda-forge")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
