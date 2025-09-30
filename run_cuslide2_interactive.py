#!/usr/bin/env python3
"""
Interactive cuslide2 plugin runner
Usage: python run_cuslide2_interactive.py [path_to_tiff_file]
"""

import sys
import os
import json
import time
from pathlib import Path

def setup_cuslide2():
    """Setup cuslide2 plugin environment"""
    from cucim.clara import _set_plugin_root
    
    # Set plugin root
    _set_plugin_root("/home/cdinea/cucim/build-release/lib")
    
    # Configure cuslide2 priority
    config = {
        "plugin": {
            "names": [
                "cucim.kit.cuslide2@25.10.00.so",
                "cucim.kit.cuslide@25.10.00.so", 
                "cucim.kit.cumed@25.10.00.so"
            ]
        }
    }
    
    config_path = "/tmp/.cucim_cuslide2.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    os.environ["CUCIM_CONFIG_PATH"] = config_path
    
    print("✓ cuslide2 plugin configured")

def benchmark_decode(img, region_size=2048):
    """Benchmark CPU vs GPU decode performance"""
    
    print(f"\n📊 Benchmarking {region_size}x{region_size} region decode...")
    
    # CPU benchmark
    print("🖥️  CPU decode...")
    start_time = time.time()
    cpu_region = img.read_region(
        location=[0, 0],
        size=[region_size, region_size],
        level=0,
        device="cpu"
    )
    cpu_time = time.time() - start_time
    print(f"   CPU time: {cpu_time:.3f}s")
    
    # GPU benchmark
    try:
        print("🚀 GPU decode...")
        start_time = time.time()
        gpu_region = img.read_region(
            location=[0, 0],
            size=[region_size, region_size], 
            level=0,
            device="cuda"
        )
        gpu_time = time.time() - start_time
        print(f"   GPU time: {gpu_time:.3f}s")
        
        speedup = cpu_time / gpu_time
        print(f"   🎯 Speedup: {speedup:.2f}x")
        
        return speedup
        
    except Exception as e:
        print(f"   ⚠️  GPU decode failed: {e}")
        return None

def main():
    """Main interactive runner"""
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to TIFF/SVS file (or press Enter for demo): ").strip()
        if not file_path:
            print("No file specified - running in demo mode")
            return demo_mode()
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return 1
    
    print(f"🔍 Loading: {file_path}")
    
    # Setup cuslide2
    setup_cuslide2()
    
    # Import cuCIM
    from cucim import CuImage
    
    # Load image
    try:
        start_time = time.time()
        img = CuImage(file_path)
        load_time = time.time() - start_time
        
        print(f"✅ Loaded in {load_time:.3f}s")
        print(f"   📐 Dimensions: {img.shape}")
        print(f"   📊 Levels: {img.level_count}")
        print(f"   🔬 Spacing: {img.spacing}")
        
        # Show associated images
        if hasattr(img, 'associated_image_names'):
            assoc_images = img.associated_image_names
            if assoc_images:
                print(f"   🖼️  Associated images: {assoc_images}")
        
        # Benchmark performance
        speedups = []
        for size in [1024, 2048, 4096]:
            if img.shape[0] >= size and img.shape[1] >= size:
                speedup = benchmark_decode(img, size)
                if speedup:
                    speedups.append(speedup)
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"\n🏆 Average GPU speedup: {avg_speedup:.2f}x")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return 1

def demo_mode():
    """Demo mode without actual files"""
    print("\n🎮 Demo Mode - cuslide2 Plugin")
    print("=" * 40)
    
    setup_cuslide2()
    
    from cucim import CuImage
    print("✅ cuCIM with cuslide2 ready!")
    print("\n📝 To test with your files:")
    print("   python run_cuslide2_interactive.py /path/to/your/slide.svs")
    print("\n🎯 Supported formats:")
    print("   • Aperio SVS (JPEG/JPEG2000)")
    print("   • Philips TIFF (JPEG/JPEG2000)")
    print("   • Generic tiled TIFF (JPEG/JPEG2000)")
    print("\n🚀 GPU acceleration automatically enabled for supported formats!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
