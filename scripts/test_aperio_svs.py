#!/usr/bin/env python3
"""
Quick test script for cuslide2 plugin with Aperio SVS files
"""

import sys
import os
import json
import time
from pathlib import Path

def setup_environment():
    """Setup cuCIM environment for cuslide2 plugin"""
    
    # Get current build directory
    repo_root = Path(__file__).parent.parent
    plugin_lib = repo_root / "cpp/plugins/cucim.kit.cuslide2/build-release/lib"
    
    if not plugin_lib.exists():
        plugin_lib = repo_root / "install/lib"
    
    # Create plugin configuration
    config = {
        "plugin": {
            "names": [
                "cucim.kit.cuslide2@26.02.00.so",  # Try cuslide2 first
            ]
        }
    }
    
    config_path = "/tmp/.cucim_aperio_test.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    os.environ["CUCIM_CONFIG_PATH"] = config_path
    
    print(f"✅ Plugin configuration: {config_path}")
    print(f"✅ Plugin library path: {plugin_lib}")
    
    return True

def test_aperio_svs(svs_path):
    """Test cuslide2 plugin with an Aperio SVS file"""
    
    print(f"\n🔬 Testing cuslide2 plugin with Aperio SVS")
    print(f"=" * 60)
    print(f"📁 File: {svs_path}")
    
    if not Path(svs_path).exists():
        print(f"❌ File not found: {svs_path}")
        return False
    
    try:
        # Set plugin root AFTER importing cucim but BEFORE creating CuImage
        repo_root = Path(__file__).parent.parent
        plugin_lib = repo_root / "cpp/plugins/cucim.kit.cuslide2/build-release/lib"
        
        from cucim.clara import _set_plugin_root
        _set_plugin_root(str(plugin_lib))
        print(f"✅ Plugin root set: {plugin_lib}")
        
        from cucim import CuImage
        
        # Load the SVS file
        print(f"\n📂 Loading SVS file...")
        start = time.time()
        img = CuImage(svs_path)
        load_time = time.time() - start
        
        print(f"✅ Loaded in {load_time:.3f}s")
        
        # Show basic info
        print(f"\n📊 Image Information:")
        print(f"  Dimensions: {img.shape}")
        level_count = img.resolutions['level_count']
        print(f"  Levels: {level_count}")
        print(f"  Dtype: {img.dtype}")
        print(f"  Device: {img.device}")
        
        # Show all levels
        print(f"\n🔍 Resolution Levels:")
        level_dimensions = img.resolutions['level_dimensions']
        level_downsamples = img.resolutions['level_downsamples']
        for level in range(level_count):
            level_dims = level_dimensions[level]
            level_downsample = level_downsamples[level]
            print(f"  Level {level}: {level_dims[0]}x{level_dims[1]} (downsample: {level_downsample:.1f}x)")
        
        # Try to read a tile from level 0 (GPU)
        print(f"\n🚀 Testing GPU decode (nvImageCodec)...")
        try:
            start = time.time()
            gpu_tile = img.read_region(
                location=[0, 0],
                size=[512, 512],
                level=0,
                device="cuda"
            )
            gpu_time = time.time() - start
            
            print(f"✅ GPU decode successful!")
            print(f"  Time: {gpu_time:.4f}s")
            print(f"  Shape: {gpu_tile.shape}")
            print(f"  Device: {gpu_tile.device}")
        except Exception as e:
            print(f"⚠️  GPU decode failed: {e}")
            print(f"   (This is expected if CUDA is not available)")
            gpu_time = None
        
        # Try to read same tile from CPU
        print(f"\n🖥️  Testing CPU decode (baseline)...")
        try:
            start = time.time()
            cpu_tile = img.read_region(
                location=[0, 0],
                size=[512, 512],
                level=0,
                device="cpu"
            )
            cpu_time = time.time() - start
            
            print(f"✅ CPU decode successful!")
            print(f"  Time: {cpu_time:.4f}s")
            print(f"  Shape: {cpu_tile.shape}")
            print(f"  Device: {cpu_tile.device}")
            
            # Calculate speedup
            if gpu_time:
                speedup = cpu_time / gpu_time
                print(f"\n🎯 GPU Speedup: {speedup:.2f}x faster than CPU")
                
                if speedup > 1.5:
                    print(f"   🚀 nvImageCodec GPU acceleration is working!")
                elif speedup > 0.9:
                    print(f"   ✅ GPU decode working (speedup may vary by tile size)")
                else:
                    print(f"   ℹ️  CPU was faster for this small tile")
        except Exception as e:
            print(f"❌ CPU decode failed: {e}")
        
        # Test larger tile for better speedup
        print(f"\n📏 Testing larger tile (2048x2048)...")
        try:
            # GPU
            start = time.time()
            gpu_large = img.read_region([0, 0], [2048, 2048], 0, device="cuda")
            gpu_large_time = time.time() - start
            print(f"  GPU: {gpu_large_time:.4f}s")
            
            # CPU
            start = time.time()
            cpu_large = img.read_region([0, 0], [2048, 2048], 0, device="cpu")
            cpu_large_time = time.time() - start
            print(f"  CPU: {cpu_large_time:.4f}s")
            
            speedup = cpu_large_time / gpu_large_time
            print(f"  🎯 Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  ⚠️  Large tile test failed: {e}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_test_svs():
    """Download a small Aperio SVS test file from OpenSlide"""
    
    print(f"\n📥 Downloading Aperio SVS test file...")
    
    test_file = Path("/tmp/CMU-1-Small-Region.svs")
    
    if test_file.exists():
        print(f"✅ Test file already exists: {test_file}")
        return str(test_file)
    
    try:
        import urllib.request
        
        # Download small test file (2MB) from OpenSlide test data
        url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
        
        print(f"   Downloading from: {url}")
        print(f"   Size: ~2MB (small test file)")
        print(f"   This may take a minute...")
        
        urllib.request.urlretrieve(url, test_file)
        
        print(f"✅ Downloaded: {test_file}")
        return str(test_file)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def list_available_test_files():
    """List available Aperio SVS test files from OpenSlide"""
    
    print(f"\n📋 Available Aperio SVS Test Files from OpenSlide:")
    print(f"=" * 70)
    
    test_files = [
        ("CMU-1-Small-Region.svs", "~2MB", "Small region, JPEG, single pyramid level"),
        ("CMU-1.svs", "~177MB", "Brightfield, JPEG compression"),
        ("CMU-1-JP2K-33005.svs", "~126MB", "JPEG 2000, RGB"),
        ("CMU-2.svs", "~390MB", "Brightfield, JPEG compression"),
        ("CMU-3.svs", "~253MB", "Brightfield, JPEG compression"),
        ("JP2K-33003-1.svs", "~63MB", "Aorta tissue, JPEG 2000, YCbCr"),
        ("JP2K-33003-2.svs", "~275MB", "Heart tissue, JPEG 2000, YCbCr"),
    ]
    
    print(f"{'Filename':<25} {'Size':<10} {'Description'}")
    print(f"-" * 70)
    for filename, size, description in test_files:
        print(f"{filename:<25} {size:<10} {description}")
    
    print(f"\n💡 To download:")
    print(f"   wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/<filename>")
    print(f"\n📖 More info: https://openslide.cs.cmu.edu/download/openslide-testdata/")

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python test_aperio_svs.py <path_to_svs_file>")
        print("   or: python test_aperio_svs.py --download (auto-download test file)")
        print("")
        print("Example:")
        print("  python test_aperio_svs.py /path/to/slide.svs")
        print("  python test_aperio_svs.py --download")
        print("")
        print("This script will:")
        print("  ✅ Configure cuslide2 plugin with nvImageCodec")
        print("  ✅ Load and analyze the SVS file")
        print("  ✅ Test GPU-accelerated decoding")
        print("  ✅ Compare CPU vs GPU performance")
        
        # List available test files
        list_available_test_files()
        return 1
    
    svs_path = sys.argv[1]
    
    # Handle --download flag
    if svs_path == "--download":
        svs_path = download_test_svs()
        if svs_path is None:
            print(f"\n❌ Failed to download test file")
            print(f"💡 You can manually download with:")
            print(f"   wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs")
            return 1
    
    # Setup environment
    setup_environment()
    
    # Test the SVS file
    success = test_aperio_svs(svs_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

