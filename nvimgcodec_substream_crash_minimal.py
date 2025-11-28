#!/usr/bin/env python3
"""
Minimal reproducer for nvImageCodec 0.7.0 sub-code stream destruction crash.

This script demonstrates a "free(): invalid pointer" crash that occurs when
destroying sub-code streams obtained via nvimgcodecCodeStreamGetSubCodeStream().

CRASH LOCATION:
  cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_tiff_parser.cpp
  Line 342: nvimgcodecCodeStreamDestroy(ifd_info.sub_code_stream)
  
EXPECTED BEHAVIOR:
  Script should complete cleanly and exit normally.
  
ACTUAL BEHAVIOR:
  After "Exiting Python..." message, crash occurs during interpreter shutdown:
    free(): invalid pointer
    Aborted

CRASH TRIGGER:
  The crash is triggered by the combination of:
  1. Plugin configuration via _set_plugin_root() (like test_aperio_svs.py)
  2. Multiple read_region() calls with different devices (GPU and CPU)
  3. Python interpreter shutdown destroying objects in specific order

REQUIREMENTS:
  - nvImageCodec 0.7.0
  - cuslide2 plugin built with nvImageCodec support
  - Test TIFF file (Aperio SVS)
"""

import json
import os
import sys
from pathlib import Path

def setup_plugin():
    """Setup cuslide2 plugin configuration (like test_aperio_svs.py does)"""
    repo_root = Path(__file__).parent
    plugin_lib = repo_root / "cpp/plugins/cucim.kit.cuslide2/build-release/lib"
    
    if not plugin_lib.exists():
        plugin_lib = repo_root / "install/lib"
    
    # Create plugin configuration
    config = {
        "plugin": {
            "names": [
                "cucim.kit.cuslide2@25.12.00.so",
            ]
        }
    }
    
    config_path = "/tmp/.cucim_crash_reproducer.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    os.environ["CUCIM_CONFIG_PATH"] = config_path
    print(f"  Plugin config: {config_path}")
    print(f"  Plugin lib: {plugin_lib}")
    
    return str(plugin_lib)

def main():
    # Check for test file
    test_file = "/tmp/CMU-1-Small-Region.svs"
    
    if not Path(test_file).exists():
        print(f"\n❌ Test file not found: {test_file}")
        print("\nDownload with:")
        print("  wget -O /tmp/CMU-1-Small-Region.svs \\")
        print("    'https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs'")
        return 1
    
    print(f"\nTest file: {test_file}")
    
    print("\n[Setup] Configuring plugin...")
    plugin_lib = setup_plugin()
    
    try:
        # Import and set plugin root BEFORE creating CuImage (like test_aperio_svs.py)
        from cucim.clara import _set_plugin_root
        _set_plugin_root(plugin_lib)
        print(f"  ✓ Plugin root set: {plugin_lib}")
        
        from cucim import CuImage
        print("  ✓ cucim imported")
    except ImportError as e:
        print(f"✗ Failed to import cucim: {e}")
        return 1
    
    print("\n[Step 1] Opening slide...")
    print("  (This creates TiffFileParser with sub-code streams)")
    slide = CuImage(test_file)
    print(f"  ✓ Opened: {slide.shape}")
    
    print("\n[Step 2] Reading regions (mimicking test_aperio_svs.py)...")
    print("  (This creates multiple sub-code streams)")
    
    # Read GPU tile 512x512 (like test_aperio_svs.py line 98)
    print("\n  [2a] GPU decode 512x512...")
    region1 = slide.read_region((0, 0), (512, 512), level=0, device="cuda")
    print(f"    ✓ GPU tile: {region1.shape}")
    
    # Read CPU tile 512x512 (like test_aperio_svs.py line 116)
    print("  [2b] CPU decode 512x512...")
    region2 = slide.read_region((0, 0), (512, 512), level=0, device="cpu")
    print(f"    ✓ CPU tile: {region2.shape}")
    
    # Read larger GPU tile 2048x2048 (like test_aperio_svs.py line 145)
    print("  [2c] GPU decode 2048x2048...")
    region3 = slide.read_region((0, 0), (2048, 2048), level=0, device="cuda")
    print(f"    ✓ Large GPU tile: {region3.shape}")
    
    # Read larger CPU tile 2048x2048 (like test_aperio_svs.py line 151)
    print("  [2d] CPU decode 2048x2048...")
    region4 = slide.read_region((0, 0), (2048, 2048), level=0, device="cpu")
    print(f"    ✓ Large CPU tile: {region4.shape}")
    
    print("\n[Step 3] Letting slide go out of scope...")
    print("  (Natural destruction - no explicit 'del')")
    print("  (This will destroy TiffFileParser and call:")
    print("   nvimgcodecCodeStreamDestroy() on sub-code streams)")
    print()
    
    # Let slide go out of scope naturally instead of explicit del
    # THE CRASH HAPPENS during Python shutdown when slide is finally destroyed
    # (Not here, but when main() returns and Python cleans up)
    
    print("=" * 70)
    print("✓ Function completed - slide going out of scope now...")
    print("=" * 70)
    print("\nNOTE: The crash may occur AFTER this message")
    print("      during Python interpreter shutdown.")
    print("\nWatching for: 'free(): invalid pointer'")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING TEST - nvImageCodec Sub-Stream Crash Reproducer")
    print("=" * 70)
    result = main()
    print("\n" + "=" * 70)
    print("Exiting Python... (crash may occur during cleanup)")
    print("=" * 70)
    sys.exit(result)

